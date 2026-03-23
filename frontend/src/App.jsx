import { useState } from "react";
import { MessageSquare, BarChart3, Upload, Settings } from "lucide-react";
import ChatInterface from "./components/ChatInterface";
import SourcePanel from "./components/SourcePanel";
import FeedbackWidget from "./components/FeedbackWidget";
import EvalDashboard from "./components/EvalDashboard";
import { useRAG } from "./hooks/useRAG";
import { uploadDocument } from "./utils/api";

const TABS = [
  { id: "chat", label: "Query", icon: MessageSquare },
  { id: "eval", label: "Evaluation", icon: BarChart3 },
  { id: "ingest", label: "Ingest", icon: Upload },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("chat");
  const { query, rate, loading, error, response, history } = useRAG();
  const [uploadStatus, setUploadStatus] = useState(null);

  async function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadStatus({ status: "uploading", filename: file.name });
    try {
      const result = await uploadDocument(file);
      setUploadStatus({
        status: "success",
        filename: file.name,
        chunks: result.chunks_created,
        entities: result.entities_extracted,
      });
    } catch (err) {
      setUploadStatus({ status: "error", filename: file.name, error: err.message });
    }
  }

  return (
    <div className="h-screen flex flex-col bg-white">
      {/* Top navigation bar */}
      <header className="border-b border-gray-200 bg-white px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
            <span className="text-white text-sm font-bold">H</span>
          </div>
          <div>
            <h1 className="text-sm font-bold text-gray-900">HybridRAG</h1>
            <p className="text-xs text-gray-400">Dense + Sparse + Graph + CRAG</p>
          </div>
        </div>

        <nav className="flex gap-1">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  isActive
                    ? "bg-blue-50 text-blue-700"
                    : "text-gray-500 hover:text-gray-700 hover:bg-gray-50"
                }`}
              >
                <Icon size={14} />
                {tab.label}
              </button>
            );
          })}
        </nav>

        <div className="text-xs text-gray-400">
          v1.0.0
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 overflow-hidden">
        {activeTab === "chat" && (
          <div className="h-full flex">
            {/* Chat panel (left) */}
            <div className="flex-1 flex flex-col border-r border-gray-100">
              <ChatInterface
                onSubmit={(q) => query(q)}
                loading={loading}
                response={response}
                history={history}
              />

              {/* Feedback widget for the latest response */}
              {response && (
                <div className="px-4 pb-2">
                  <FeedbackWidget queryId={response.query_id} onSubmit={rate} />
                </div>
              )}
            </div>

            {/* Sources panel (right) */}
            <div className="w-80 overflow-y-auto border-l border-gray-100 hidden lg:block">
              <SourcePanel sources={response?.sources} />
            </div>
          </div>
        )}

        {activeTab === "eval" && <EvalDashboard />}

        {activeTab === "ingest" && (
          <div className="p-8 max-w-2xl mx-auto">
            <h2 className="text-xl font-bold text-gray-900 mb-2">
              Document Ingestion
            </h2>
            <p className="text-sm text-gray-500 mb-6">
              Upload PDFs or text files. They'll be chunked, embedded, indexed in
              Qdrant + Elasticsearch, and entities extracted into Neo4j.
            </p>

            <label className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-gray-300 rounded-xl hover:border-blue-400 hover:bg-blue-50 transition-colors cursor-pointer">
              <Upload size={32} className="text-gray-400 mb-2" />
              <span className="text-sm text-gray-500">
                Drop a file here or click to browse
              </span>
              <span className="text-xs text-gray-400 mt-1">
                Supports: PDF, TXT, MD, RST
              </span>
              <input
                type="file"
                accept=".pdf,.txt,.md,.rst"
                className="hidden"
                onChange={handleFileUpload}
              />
            </label>

            {uploadStatus && (
              <div
                className={`mt-4 p-4 rounded-lg text-sm ${
                  uploadStatus.status === "uploading"
                    ? "bg-blue-50 text-blue-700"
                    : uploadStatus.status === "success"
                    ? "bg-green-50 text-green-700"
                    : "bg-red-50 text-red-700"
                }`}
              >
                {uploadStatus.status === "uploading" && (
                  <p>Ingesting {uploadStatus.filename}...</p>
                )}
                {uploadStatus.status === "success" && (
                  <div>
                    <p className="font-medium">
                      {uploadStatus.filename} ingested successfully
                    </p>
                    <p className="text-xs mt-1">
                      {uploadStatus.chunks} chunks created,{" "}
                      {uploadStatus.entities} entities extracted
                    </p>
                  </div>
                )}
                {uploadStatus.status === "error" && (
                  <p>Failed to ingest {uploadStatus.filename}: {uploadStatus.error}</p>
                )}
              </div>
            )}

            {/* Pipeline explanation */}
            <div className="mt-8 bg-gray-50 rounded-xl p-5">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">
                Ingestion Pipeline
              </h3>
              <div className="space-y-3 text-xs text-gray-600">
                <div className="flex items-start gap-3">
                  <span className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-xs font-bold shrink-0">
                    1
                  </span>
                  <div>
                    <p className="font-medium text-gray-700">Parse Document</p>
                    <p>Extract text from PDF pages or read plain text files</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="w-6 h-6 rounded-full bg-green-100 text-green-600 flex items-center justify-center text-xs font-bold shrink-0">
                    2
                  </span>
                  <div>
                    <p className="font-medium text-gray-700">Chunk with Overlap</p>
                    <p>Split into 512-char chunks with 64-char overlap using recursive splitting</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="w-6 h-6 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center text-xs font-bold shrink-0">
                    3
                  </span>
                  <div>
                    <p className="font-medium text-gray-700">Index Everywhere</p>
                    <p>Embed chunks → Qdrant, keyword index → Elasticsearch</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="w-6 h-6 rounded-full bg-orange-100 text-orange-600 flex items-center justify-center text-xs font-bold shrink-0">
                    4
                  </span>
                  <div>
                    <p className="font-medium text-gray-700">Extract Entities → Graph</p>
                    <p>LLM extracts authors, methods, datasets, concepts → Neo4j</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Error toast */}
      {error && (
        <div className="fixed bottom-4 right-4 bg-red-50 border border-red-200 text-red-700 px-4 py-2 rounded-lg text-sm shadow-lg">
          {error}
        </div>
      )}
    </div>
  );
}
