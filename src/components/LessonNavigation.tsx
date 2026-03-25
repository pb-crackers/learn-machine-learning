import Link from "next/link";
import type { Lesson } from "@/lib/course-structure";

export function LessonNavigation({
  prev,
  next,
}: {
  prev: Lesson | null;
  next: Lesson | null;
}) {
  return (
    <div className="mt-12 flex items-center justify-between border-t border-slate-200 pt-6">
      {prev ? (
        <Link
          href={`/lesson/${prev.slug}`}
          className="group flex items-center gap-2 text-sm text-slate-600 hover:text-blue-600"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          <div>
            <p className="text-xs text-slate-400">Previous</p>
            <p className="font-medium">{prev.title}</p>
          </div>
        </Link>
      ) : (
        <div />
      )}
      {next ? (
        <Link
          href={`/lesson/${next.slug}`}
          className="group flex items-center gap-2 text-right text-sm text-slate-600 hover:text-blue-600"
        >
          <div>
            <p className="text-xs text-slate-400">Next</p>
            <p className="font-medium">{next.title}</p>
          </div>
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </Link>
      ) : (
        <div />
      )}
    </div>
  );
}
