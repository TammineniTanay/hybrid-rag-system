import { useState } from "react";
import { Star, MessageSquare } from "lucide-react";

export default function FeedbackWidget({ queryId, onSubmit }) {
  const [rating, setRating] = useState(0);
  const [hoveredStar, setHoveredStar] = useState(0);
  const [comment, setComment] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [showComment, setShowComment] = useState(false);

  async function handleSubmit() {
    if (rating === 0) return;
    await onSubmit(queryId, rating, comment || null);
    setSubmitted(true);
  }

  if (submitted) {
    return <div className="inline-flex items-center gap-1.5 text-xs text-green-600 bg-green-50 px-3 py-1.5 rounded-full">Thanks for your feedback!</div>;
  }

  return (
    <div className="flex flex-col gap-2 mt-2">
      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-500">Rate this answer:</span>
        <div className="flex gap-0.5">
          {[1, 2, 3, 4, 5].map((n) => (
            <button key={n} onClick={() => setRating(n)} onMouseEnter={() => setHoveredStar(n)} onMouseLeave={() => setHoveredStar(0)} className="p-0.5 transition-colors">
              <Star size={16} className={n <= (hoveredStar || rating) ? "fill-yellow-400 text-yellow-400" : "text-gray-300"} />
            </button>
          ))}
        </div>
        {rating > 0 && (
          <>
            <button onClick={() => setShowComment(!showComment)} className="text-xs text-gray-400 hover:text-gray-600"><MessageSquare size={14} /></button>
            <button onClick={handleSubmit} className="text-xs bg-blue-600 text-white px-3 py-1 rounded-full hover:bg-blue-700">Submit</button>
          </>
        )}
      </div>
      {showComment && (
        <input type="text" value={comment} onChange={(e) => setComment(e.target.value)}
          placeholder="Optional comment..." className="text-xs px-3 py-1.5 border rounded-lg w-64 focus:outline-none focus:ring-1 focus:ring-blue-400" />
      )}
    </div>
  );
}
