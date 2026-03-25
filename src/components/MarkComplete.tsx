"use client";

import { useState } from "react";

export function MarkComplete({
  lessonSlug,
  section,
  initialCompleted = false,
}: {
  lessonSlug: string;
  section: "concepts" | "lab";
  initialCompleted?: boolean;
}) {
  const [completed, setCompleted] = useState(initialCompleted);
  const [saving, setSaving] = useState(false);

  const handleToggle = async () => {
    setSaving(true);
    const newValue = !completed;
    try {
      await fetch("/api/progress", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lessonSlug,
          [section === "concepts" ? "conceptsCompleted" : "labCompleted"]: newValue,
        }),
      });
      setCompleted(newValue);
    } catch (e) {
      console.error("Failed to update progress:", e);
    } finally {
      setSaving(false);
    }
  };

  return (
    <button
      onClick={handleToggle}
      disabled={saving}
      className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
        completed
          ? "bg-emerald-100 text-emerald-700 hover:bg-emerald-200"
          : "bg-slate-100 text-slate-600 hover:bg-slate-200"
      }`}
    >
      {completed ? (
        <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
            clipRule="evenodd"
          />
        </svg>
      ) : (
        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <circle cx="12" cy="12" r="10" strokeWidth={2} />
        </svg>
      )}
      {completed
        ? `${section === "concepts" ? "Concepts" : "Lab"} completed`
        : `Mark ${section} complete`}
    </button>
  );
}
