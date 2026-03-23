"""
Periodic reward model training script.
Run via cron or manually: python scripts/train_reward_model.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.database import DatabaseManager
from app.evaluation.reward_model import RewardModel
from app.models.schemas import FeedbackRecord


def main():
    db = DatabaseManager()
    reward = RewardModel()
    feedback = db.get_all_feedback()
    print(f"Found {len(feedback)} feedback records")

    if len(feedback) < 20:
        print(f"Need at least 20 records. Currently have {len(feedback)}.")
        return

    records = [
        FeedbackRecord(
            feedback_id=f["feedback_id"], query_id=f["query_id"],
            question=f["question"], answer=f["answer"],
            retrieved_chunk_ids=f["retrieved_chunk_ids"],
            rating=f["rating"], comment=f.get("comment"),
        )
        for f in feedback
    ]

    chunks_by_query = db.get_chunks_by_query()
    reward.train(records, chunks_by_query)
    print("Reward model trained and saved.")


if __name__ == "__main__":
    main()
