"use client";

import { useState } from "react";

interface QuizQuestion {
  id: string;
  type: string;
  question: string;
  options?: string[];
  correct: string;
  explanation: string;
}

interface QuizProps {
  questions: QuizQuestion[];
  lessonSlug: string;
}

export function Quiz({ questions, lessonSlug }: QuizProps) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [submitted, setSubmitted] = useState(false);
  const [showAll, setShowAll] = useState(true);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [saving, setSaving] = useState(false);

  const score = submitted
    ? questions.filter((q) => answers[q.id] === q.correct).length
    : 0;

  const handleSelect = (questionId: string, value: string) => {
    if (submitted) return;
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  };

  const handleSubmit = async () => {
    setSubmitted(true);
    setSaving(true);

    const answersPayload: Record<string, { selected: string; correct: boolean }> = {};
    questions.forEach((q) => {
      answersPayload[q.id] = {
        selected: answers[q.id] || "",
        correct: answers[q.id] === q.correct,
      };
    });

    try {
      await fetch("/api/quiz", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lessonSlug,
          score: questions.filter((q) => answers[q.id] === q.correct).length,
          totalQuestions: questions.length,
          answers: answersPayload,
        }),
      });
    } catch (e) {
      console.error("Failed to save quiz results:", e);
    } finally {
      setSaving(false);
    }
  };

  const handleRetake = () => {
    setAnswers({});
    setSubmitted(false);
    setCurrentIdx(0);
  };

  const renderQuestion = (q: QuizQuestion, idx: number) => {
    const isCorrect = answers[q.id] === q.correct;

    return (
      <div key={q.id} className="mb-6 rounded-lg border border-slate-200 bg-white p-5">
        <p className="mb-3 font-medium text-slate-800">
          {idx + 1}. {q.question}
        </p>

        {q.options && (
          <div className="space-y-2">
            {q.options.map((option, optIdx) => {
              const letter = String.fromCharCode(65 + optIdx);
              const isSelected = answers[q.id] === letter;
              const isCorrectOption = letter === q.correct;

              let optionClass =
                "flex cursor-pointer items-center rounded-md border p-3 transition-colors";
              if (submitted) {
                if (isCorrectOption) {
                  optionClass += " border-emerald-500 bg-emerald-50";
                } else if (isSelected && !isCorrectOption) {
                  optionClass += " border-red-500 bg-red-50";
                } else {
                  optionClass += " border-slate-200 opacity-60";
                }
              } else {
                optionClass += isSelected
                  ? " border-blue-500 bg-blue-50"
                  : " border-slate-200 hover:border-slate-300 hover:bg-slate-50";
              }

              return (
                <label key={letter} className={optionClass}>
                  <input
                    type="radio"
                    name={q.id}
                    value={letter}
                    checked={isSelected}
                    onChange={() => handleSelect(q.id, letter)}
                    disabled={submitted}
                    className="mr-3"
                  />
                  <span className="mr-2 font-mono text-sm font-semibold text-slate-500">
                    {letter})
                  </span>
                  <span className="text-slate-700">{option}</span>
                </label>
              );
            })}
          </div>
        )}

        {q.type === "true_false" && (
          <div className="space-y-2">
            {["True", "False"].map((val) => {
              const isSelected = answers[q.id] === val;
              const isCorrectOption = val === q.correct;

              let optionClass =
                "flex cursor-pointer items-center rounded-md border p-3 transition-colors";
              if (submitted) {
                if (isCorrectOption) {
                  optionClass += " border-emerald-500 bg-emerald-50";
                } else if (isSelected && !isCorrectOption) {
                  optionClass += " border-red-500 bg-red-50";
                } else {
                  optionClass += " border-slate-200 opacity-60";
                }
              } else {
                optionClass += isSelected
                  ? " border-blue-500 bg-blue-50"
                  : " border-slate-200 hover:border-slate-300 hover:bg-slate-50";
              }

              return (
                <label key={val} className={optionClass}>
                  <input
                    type="radio"
                    name={q.id}
                    value={val}
                    checked={isSelected}
                    onChange={() => handleSelect(q.id, val)}
                    disabled={submitted}
                    className="mr-3"
                  />
                  <span className="text-slate-700">{val}</span>
                </label>
              );
            })}
          </div>
        )}

        {submitted && (
          <div
            className={`mt-3 rounded-md p-3 text-sm ${
              isCorrect ? "bg-emerald-50 text-emerald-800" : "bg-red-50 text-red-800"
            }`}
          >
            <span className="font-semibold">{isCorrect ? "Correct!" : "Incorrect."}</span>{" "}
            {q.explanation}
          </div>
        )}
      </div>
    );
  };

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-xl font-bold text-slate-800">Quiz</h2>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowAll(!showAll)}
            className="rounded-md bg-slate-100 px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-200"
          >
            {showAll ? "One at a time" : "Show all"}
          </button>
          {submitted && (
            <button
              onClick={handleRetake}
              className="rounded-md bg-blue-100 px-3 py-1.5 text-sm text-blue-700 hover:bg-blue-200"
            >
              Retake
            </button>
          )}
        </div>
      </div>

      {showAll ? (
        questions.map((q, i) => renderQuestion(q, i))
      ) : (
        <div>
          {renderQuestion(questions[currentIdx], currentIdx)}
          <div className="flex items-center justify-between">
            <button
              onClick={() => setCurrentIdx((p) => Math.max(0, p - 1))}
              disabled={currentIdx === 0}
              className="rounded-md bg-slate-100 px-4 py-2 text-sm text-slate-600 hover:bg-slate-200 disabled:opacity-40"
            >
              Previous
            </button>
            <span className="text-sm text-slate-500">
              {currentIdx + 1} / {questions.length}
            </span>
            <button
              onClick={() =>
                setCurrentIdx((p) => Math.min(questions.length - 1, p + 1))
              }
              disabled={currentIdx === questions.length - 1}
              className="rounded-md bg-slate-100 px-4 py-2 text-sm text-slate-600 hover:bg-slate-200 disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {!submitted && (
        <button
          onClick={handleSubmit}
          disabled={Object.keys(answers).length < questions.length}
          className="mt-4 w-full rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white transition-colors hover:bg-blue-700 disabled:opacity-40"
        >
          Submit Quiz
        </button>
      )}

      {submitted && (
        <div className="mt-4 rounded-lg bg-slate-50 p-4 text-center">
          <p className="text-lg font-semibold text-slate-800">
            Score: {score} / {questions.length}
          </p>
          <p className="text-sm text-slate-500">
            {score === questions.length
              ? "Perfect score!"
              : score >= questions.length * 0.7
                ? "Good job! Review the explanations above."
                : "Keep studying and try again."}
          </p>
          {saving && <p className="mt-1 text-xs text-slate-400">Saving results...</p>}
        </div>
      )}
    </div>
  );
}
