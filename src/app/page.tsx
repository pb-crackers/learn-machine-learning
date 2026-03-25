import Link from "next/link";
import { modules, getAllLessons } from "@/lib/course-structure";
import { ProgressBar } from "@/components/ProgressBar";
import { prisma } from "@/lib/db";

export const dynamic = "force-dynamic";

async function getProgress() {
  try {
    const progress = await prisma.lessonProgress.findMany();
    const quizzes = await prisma.quizResult.findMany({
      orderBy: { completedAt: "desc" },
      take: 5,
    });
    return { progress, quizzes };
  } catch {
    return { progress: [], quizzes: [] };
  }
}

export default async function Dashboard() {
  const { progress, quizzes } = await getProgress();
  const allLessons = getAllLessons();

  const completedSlugs = new Set(
    progress
      .filter((p) => p.conceptsCompleted && p.labCompleted)
      .map((p) => p.lessonSlug)
  );

  const overallCompleted = completedSlugs.size;
  const overallTotal = allLessons.length;

  // Find next incomplete lesson
  const nextLesson = allLessons.find((l) => !completedSlugs.has(l.slug));

  return (
    <div className="mx-auto max-w-4xl px-8 py-10">
      <h1 className="mb-2 text-3xl font-bold text-slate-900">
        Machine Learning Foundations
      </h1>
      <p className="mb-8 text-slate-500">
        From Python to transformers — understand ML from the ground up.
      </p>

      {/* Overall progress */}
      <div className="mb-8 rounded-xl border border-slate-200 bg-slate-50 p-6">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="font-semibold text-slate-700">Course Progress</h2>
          <span className="text-sm text-slate-500">
            {overallCompleted} / {overallTotal} lessons
          </span>
        </div>
        <ProgressBar value={overallCompleted} max={overallTotal} size="lg" />

        {nextLesson && (
          <Link
            href={`/lesson/${nextLesson.slug}`}
            className="mt-4 inline-block rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-blue-700"
          >
            {overallCompleted === 0
              ? "Start Learning"
              : `Continue: ${nextLesson.slug} ${nextLesson.title}`}
          </Link>
        )}
      </div>

      {/* Module cards */}
      <div className="mb-10 grid gap-4 sm:grid-cols-2">
        {modules.map((mod) => {
          const modCompleted = mod.lessons.filter((l) =>
            completedSlugs.has(l.slug)
          ).length;

          return (
            <div
              key={mod.number}
              className="rounded-xl border border-slate-200 p-5 hover:border-slate-300"
            >
              <div className="mb-1 flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Module {mod.number}
                </span>
                <span className="text-xs text-slate-400">
                  {mod.lessons.length} lessons
                </span>
              </div>
              <h3 className="mb-1 font-semibold text-slate-800">{mod.title}</h3>
              <p className="mb-3 text-sm text-slate-500">{mod.description}</p>
              <ProgressBar
                value={modCompleted}
                max={mod.lessons.length}
                size="sm"
              />
            </div>
          );
        })}
      </div>

      {/* Recent quiz scores */}
      {quizzes.length > 0 && (
        <div>
          <h2 className="mb-3 font-semibold text-slate-700">Recent Quizzes</h2>
          <div className="space-y-2">
            {quizzes.map((q) => (
              <div
                key={q.id}
                className="flex items-center justify-between rounded-lg border border-slate-200 px-4 py-3"
              >
                <span className="text-sm text-slate-600">
                  Lesson {q.lessonSlug}
                </span>
                <span className="text-sm font-medium text-slate-800">
                  {q.score} / {q.totalQuestions}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
