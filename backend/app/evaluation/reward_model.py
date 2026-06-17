"""
Feedback-driven reward model for retrieval re-ranking.

This module learns from user feedback (1-5 ratings) to predict
which retrieved chunks lead to good answers. Over time, it shifts
the retrieval ranking toward chunks the users have found helpful.

Architecture:
- Feature extraction: converts (query, chunk) pairs into numerical
  features (embedding similarity, chunk length, source type, etc.)
- Gradient Boosted Classifier: lightweight model that predicts
  whether a chunk will receive positive feedback
- Re-ranking: multiplies original retrieval scores by the predicted
  reward to boost high-quality chunks

The model retrains periodically (via a cron job or manual trigger)
as new feedback accumulates.
"""

from sklearn.ensemble import GradientBoostingClassifier  # Scikit-learn GradientBoostingClassifier for reward model trainingfrom sklearn.model_selection import cross_val_score
from sentence_transformers import SentenceTransformer  # sentence-transformers for embedding generationimport numpy as np
import joblib
import os
import structlog

from app.config import get_settings
from app.models.schemas import FusedResult, FeedbackRecord

logger = structlog.get_logger()

MODEL_PATH = "data/reward_model.joblib"


class RewardModel:
    """
    Learns to predict chunk quality from user feedback.

    The model takes features of a (query, chunk) pair and predicts
    a binary label: 1 = user rated the response ≥4, 0 = rated ≤2.
    Responses rated 3 are excluded from training as ambiguous.
    """

    def __init__(self):
        settings = get_settings()
        self.encoder = SentenceTransformer(settings.embedding_model)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load a previously trained model from disk if available."""
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            logger.info("loaded_reward_model", path=MODEL_PATH)

    def extract_features(
        self,
        query: str,
        chunk_content: str,
        rrf_score: float,
        num_retrievers: int,
    ) -> np.ndarray:
        """
        Build a feature vector for a (query, chunk) pair.

        Features:
        1. Cosine similarity between query and chunk embeddings
        2. RRF score from hybrid retrieval
        3. Number of retrievers that found this chunk (1-3)
        4. Chunk length in words
        5. Query length in words
        6. Ratio of query words found in chunk (lexical overlap)
        """
        # embedding similarity
        q_emb = self.encoder.encode(query)
        c_emb = self.encoder.encode(chunk_content[:512])  # truncate long chunks
        cosine_sim = float(np.dot(q_emb, c_emb) / (
            np.linalg.norm(q_emb) * np.linalg.norm(c_emb) + 1e-8
        ))

        # lexical overlap
        q_words = set(query.lower().split())
        c_words = set(chunk_content.lower().split())
        overlap = len(q_words & c_words) / max(len(q_words), 1)

        features = np.array([
            cosine_sim,
            rrf_score,
            num_retrievers,
            len(chunk_content.split()),
            len(query.split()),
            overlap,
        ])

        return features

    def train(self, feedback_records: list[FeedbackRecord], chunks_by_query: dict):
        """
        Train the reward model on accumulated feedback.

        feedback_records: list of FeedbackRecord from PostgreSQL
        chunks_by_query: dict mapping query_id to list of (content, rrf_score, num_retrievers)

        Only uses strong signals: rating ≥4 → positive, rating ≤2 → negative.
        Rating 3 is excluded as it's ambiguous.
        """
        X_samples = []
        y_labels = []

        for record in feedback_records:
            # skip neutral ratings
            if record.rating == 3:
                continue

            label = 1 if record.rating >= 4 else 0
            query_chunks = chunks_by_query.get(record.query_id, [])

            for content, rrf_score, num_retrievers in query_chunks:
                features = self.extract_features(
                    record.question, content, rrf_score, num_retrievers
                )
                X_samples.append(features)
                y_labels.append(label)

        if len(X_samples) < 20:
            logger.warning("insufficient_feedback", count=len(X_samples))
            return

        X = np.array(X_samples)
        y = np.array(y_labels)

        # gradient boosting handles small datasets well and doesn't need
        # the elaborate hyperparameter tuning that neural nets require
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

        # cross-validate to estimate generalization performance
        scores = cross_val_score(self.model, X, y, cv=min(5, len(X_samples) // 5 + 1))
        logger.info(
            "reward_model_cv",
            mean_accuracy=round(float(scores.mean()), 4),
            std=round(float(scores.std()), 4),
        )

        # train on full dataset
        self.model.fit(X, y)

        # persist to disk
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        logger.info("trained_reward_model", samples=len(X_samples), path=MODEL_PATH)

    def re_rank(
        self,
        query: str,
        fused_results: list[FusedResult],
    ) -> list[FusedResult]:
        """
        Re-rank retrieval results using the reward model.

        If no model is trained yet, return the original ranking.
        Otherwise, multiply each chunk's RRF score by the reward
        model's confidence that it will lead to a good answer.
        """
        if self.model is None:
            return fused_results

        scored = []
        for fused in fused_results:
            features = self.extract_features(
                query,
                fused.chunk.content,
                fused.rrf_score,
                len(fused.contributing_retrievers),
            )

            # predict_proba returns [P(bad), P(good)]
            reward_score = self.model.predict_proba(features.reshape(1, -1))[0][1]

            # blend original RRF score with reward prediction
            # 70% original ranking + 30% learned preference
            blended = 0.7 * fused.rrf_score + 0.3 * reward_score

            scored.append((fused, blended))

        # sort by blended score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        re_ranked = [item[0] for item in scored]

        logger.info(
            "reward_reranked",
            query=query[:80],
            top_original=fused_results[0].chunk.chunk_id if fused_results else None,
            top_reranked=re_ranked[0].chunk.chunk_id if re_ranked else None,
        )

        return re_ranked
