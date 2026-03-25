"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { modules } from "@/lib/course-structure";

interface ProgressMap {
  [slug: string]: {
    conceptsCompleted: boolean;
    labCompleted: boolean;
  };
}

export function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState<Record<number, boolean>>({});
  const [progress, setProgress] = useState<ProgressMap>({});

  useEffect(() => {
    fetch("/api/progress")
      .then((r) => r.json())
      .then((data) => {
        const map: ProgressMap = {};
        for (const p of data) {
          map[p.lessonSlug] = {
            conceptsCompleted: p.conceptsCompleted,
            labCompleted: p.labCompleted,
          };
        }
        setProgress(map);
      })
      .catch(() => {});
  }, [pathname]);

  const toggleModule = (num: number) => {
    setCollapsed((prev) => ({ ...prev, [num]: !prev[num] }));
  };

  const getStatusIcon = (slug: string) => {
    const p = progress[slug];
    if (!p) return <span className="h-2 w-2 rounded-full bg-slate-500" />;
    if (p.conceptsCompleted && p.labCompleted)
      return <span className="h-2 w-2 rounded-full bg-emerald-400" />;
    if (p.conceptsCompleted || p.labCompleted)
      return <span className="h-2 w-2 rounded-full bg-amber-400" />;
    return <span className="h-2 w-2 rounded-full bg-slate-500" />;
  };

  return (
    <aside className="flex h-screen w-64 flex-col overflow-y-auto bg-slate-900 text-slate-300">
      <Link
        href="/"
        className="border-b border-slate-700 px-4 py-4 text-lg font-bold text-white hover:text-blue-400"
      >
        ML Foundations
      </Link>

      <nav className="flex-1 py-2">
        {modules.map((mod) => {
          const isCollapsed = collapsed[mod.number];
          const completedCount = mod.lessons.filter((l) => {
            const p = progress[l.slug];
            return p?.conceptsCompleted && p?.labCompleted;
          }).length;

          return (
            <div key={mod.number}>
              <button
                onClick={() => toggleModule(mod.number)}
                className="flex w-full items-center justify-between px-4 py-2.5 text-left text-sm font-semibold text-slate-400 hover:text-white"
              >
                <span>
                  Module {mod.number}: {mod.title}
                </span>
                <span className="flex items-center gap-2">
                  <span className="text-xs text-slate-500">
                    {completedCount}/{mod.lessons.length}
                  </span>
                  <svg
                    className={`h-3 w-3 transition-transform ${isCollapsed ? "" : "rotate-90"}`}
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path d="M6 6L14 10L6 14V6Z" />
                  </svg>
                </span>
              </button>

              {!isCollapsed && (
                <div className="pb-1">
                  {mod.lessons.map((lesson) => {
                    const href = `/lesson/${lesson.slug}`;
                    const isActive = pathname === href;

                    return (
                      <Link
                        key={lesson.slug}
                        href={href}
                        className={`flex items-center gap-2.5 px-6 py-1.5 text-sm transition-colors ${
                          isActive
                            ? "bg-slate-800 text-blue-400"
                            : "text-slate-400 hover:bg-slate-800 hover:text-white"
                        }`}
                      >
                        {getStatusIcon(lesson.slug)}
                        <span>
                          {lesson.slug} {lesson.title}
                        </span>
                      </Link>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </nav>
    </aside>
  );
}
