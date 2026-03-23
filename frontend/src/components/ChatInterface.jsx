import { useState } from "react";
import { Send, Loader2, Zap, GitBranch } from "lucide-react";

export default function ChatInterface({ onSubmit, loading, response, history }) {
  const [input, setInput] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    if (!input.trim() || loading) return;
    onSubmit(input.trim());
    setInput("");
  }

  function retrieverBadge(source) {
    const map = {
      dense: { label: "Semantic", color: "bg-blue-100 text-blue-700" },
      sparse: { label: "Keyword", color: "bg-green-100 text-green-700" },
      graph: { label: "Graph", color: "bg-purple-100 text-purple-700" },
      web: { label: "Web", color: "bg-orange-100 text-orange-700" },
    };
    const info = map[source] || { label: source, color: "bg-gray-100 text-gray-700" };
    return <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${info.color}`}>{info.label}</span>;
  }

  function cragBadge(action) {
    if (!action || action === "proceed") return null;
    const map = {
      rewrite: { label: "Query Rewritten", color: "bg-yellow-100 text-yellow-800" },
      web_search: { label: "Web Fallback", color: "bg-orange-100 text-orange-800" },
      decompose: { label: "Decomposed", color: "bg-red-100 text-red-800" },
    };
    const info = map[action] || { label: action, color: "bg-gray-100" };
    return (
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${info.color}`}>
        <Zap size={12} /> CRAG: {info.label}
      </span>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {history.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            <GitBranch size={48} className="mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium">Hybrid RAG System</p>
            <p className="text-sm mt-1">Ask a question about your ingested documents.</p>
          </div>
        )}
        {history.map((item, idx) => (
          <div key={idx} className="space-y-3">
            <div className="flex justify-end">
              <div className="bg-blue-600 text-white px-4 py-2 rounded-2xl rounded-br-sm max-w-[70%]">{item.question}</div>
            </div>
            <div className="flex justify-start">
              <div className="bg-gray-50 border border-gray-200 px-4 py-3 rounded-2xl rounded-bl-sm max-w-[85%]">
                <div className="flex items-center gap-2 mb-2">
                  {cragBadge(item.cragAction)}
                  <span className="text-xs text-gray-400">{Math.round(item.totalTime)}ms</span>
                </div>
                <p className="text-gray-800 whitespace-pre-wrap text-sm leading-relaxed">{item.answer}</p>
                {item.sources?.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <p className="text-xs font-medium text-gray-500 mb-2">Sources:</p>
                    <div className="space-y-1.5">
                      {item.sources.map((src, i) => (
                        <div key={i} className="flex items-start gap-2 text-xs text-gray-600">
                          {retrieverBadge(src.retriever)}
                          <span className="line-clamp-1">{src.content_preview}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-50 border px-4 py-3 rounded-2xl">
              <Loader2 className="animate-spin text-gray-400" size={20} />
            </div>
          </div>
        )}
      </div>
      <div className="border-t bg-white p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input type="text" value={input} onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your documents..."
            className="flex-1 px-4 py-2.5 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
            disabled={loading} />
          <button type="submit" disabled={loading || !input.trim()}
            className="px-4 py-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-40 transition-colors">
            <Send size={18} />
          </button>
        </form>
      </div>
    </div>
  );
}
