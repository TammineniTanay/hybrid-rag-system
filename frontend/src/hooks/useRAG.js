import { useState, useCallback } from "react";
import { submitQuery, submitFeedback } from "../utils/api";

export function useRAG() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);
  const [history, setHistory] = useState([]);

  const query = useCallback(async (question, options = {}) => {
    setLoading(true);
    setError(null);
    try {
      const result = await submitQuery(question, options);
      setResponse(result);
      setHistory((prev) => [...prev, {
        question, answer: result.answer, queryId: result.query_id,
        sources: result.sources, cragAction: result.crag_action_taken,
        totalTime: result.total_time_ms, timestamp: new Date().toISOString(),
      }]);
      return result;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const rate = useCallback(async (queryId, rating, comment) => {
    try {
      return await submitFeedback(queryId, rating, comment);
    } catch (err) {
      setError(err.message);
      return null;
    }
  }, []);

  return { query, rate, loading, error, response, history };
}
