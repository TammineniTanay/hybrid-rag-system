const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

export async function submitQuery(question, options = {}) {
  const response = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question, top_k: options.topK || 5,
      use_crag: options.useCrag !== false,
      use_web_fallback: options.useWebFallback !== false,
    }),
  });
  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || "Query failed");
  }
  return response.json();
}

export async function submitFeedback(queryId, rating, comment = null) {
  const response = await fetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query_id: queryId, rating, comment }),
  });
  return response.json();
}

export async function fetchEvalMetrics() {
  const response = await fetch(`${API_BASE}/eval/metrics`);
  return response.json();
}

export async function fetchEvalHistory(limit = 50) {
  const response = await fetch(`${API_BASE}/eval/history?limit=${limit}`);
  return response.json();
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/ingest`, { method: "POST", body: formData });
  return response.json();
}

export async function triggerRewardTraining() {
  const response = await fetch(`${API_BASE}/reward/train`, { method: "POST" });
  return response.json();
}
