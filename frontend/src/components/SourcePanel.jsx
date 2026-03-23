import { Database, Search, Share2, Globe } from "lucide-react";

const RETRIEVER_META = {
  dense: { icon: Database, label: "Semantic Search", color: "text-blue-600", bg: "bg-blue-50", description: "Qdrant" },
  sparse: { icon: Search, label: "Keyword Search", color: "text-green-600", bg: "bg-green-50", description: "Elasticsearch" },
  graph: { icon: Share2, label: "Graph Traversal", color: "text-purple-600", bg: "bg-purple-50", description: "Neo4j" },
  web: { icon: Globe, label: "Web Search", color: "text-orange-600", bg: "bg-orange-50", description: "Tavily" },
};

export default function SourcePanel({ sources }) {
  if (!sources?.length) {
    return <div className="p-4 text-center text-gray-400 text-sm">No sources yet. Ask a question to see retrieval results.</div>;
  }

  return (
    <div className="p-4 space-y-3">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">Retrieved Sources</h3>
      {sources.map((src, idx) => {
        const meta = RETRIEVER_META[src.retriever] || RETRIEVER_META.dense;
        const Icon = meta.icon;
        return (
          <div key={idx} className={`p-3 rounded-lg border border-gray-100 ${meta.bg} hover:shadow-sm transition-all`}>
            <div className="flex items-center gap-2 mb-1.5">
              <Icon size={14} className={meta.color} />
              <span className={`text-xs font-medium ${meta.color}`}>{meta.label}</span>
              <span className="text-xs text-gray-400 ml-auto">Score: {src.relevance_score.toFixed(4)}</span>
            </div>
            <p className="text-xs text-gray-600 leading-relaxed">{src.content_preview}</p>
            <span className="text-xs text-gray-400 truncate mt-1.5 block">{src.source}</span>
          </div>
        );
      })}
    </div>
  );
}
