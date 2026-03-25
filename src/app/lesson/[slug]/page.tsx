import { notFound } from "next/navigation";
import { getLesson as getLessonContent, getQuiz } from "@/lib/mdx";
import {
  getLesson as getLessonMeta,
  getAdjacentLessons,
} from "@/lib/course-structure";
import { LessonNavigation } from "@/components/LessonNavigation";
import { MarkComplete } from "@/components/MarkComplete";
import { Quiz } from "@/components/Quiz";

export const dynamic = "force-dynamic";

export default async function LessonPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const meta = getLessonMeta(slug);
  if (!meta) notFound();

  const lesson = await getLessonContent(slug);
  if (!lesson) notFound();

  const quiz = meta.hasQuiz ? getQuiz(slug) : null;
  const { prev, next } = getAdjacentLessons(slug);

  return (
    <div className="mx-auto max-w-4xl px-8 py-10">
      {/* Header */}
      <div className="mb-8">
        <p className="mb-1 text-sm font-semibold uppercase tracking-wide text-blue-600">
          Module {meta.module} &middot; Lesson {meta.slug}
        </p>
        <h1 className="text-3xl font-bold text-slate-900">{meta.title}</h1>
        <p className="mt-1 text-slate-500">{meta.description}</p>
        {lesson.frontmatter.estimatedMinutes && (
          <p className="mt-1 text-sm text-slate-400">
            ~{lesson.frontmatter.estimatedMinutes} minutes
          </p>
        )}
      </div>

      {/* Concepts section */}
      <section id="concepts" className="prose mb-8">
        {lesson.content}
      </section>

      <div className="mb-10">
        <MarkComplete lessonSlug={slug} section="concepts" />
      </div>

      {/* Lab mark complete */}
      <div className="mb-10 rounded-xl border border-slate-200 bg-slate-50 p-6">
        <h2 className="mb-2 text-xl font-bold text-slate-800">
          Hands-On Lab
        </h2>
        <p className="mb-4 text-sm text-slate-500">
          Check the <code>labs/lesson-{slug}/</code> directory for starter code
          and instructions.
        </p>
        <MarkComplete lessonSlug={slug} section="lab" />
      </div>

      {/* Quiz section */}
      {quiz && quiz.length > 0 && (
        <section id="quiz" className="mb-8">
          <Quiz questions={quiz} lessonSlug={slug} />
        </section>
      )}

      {/* Navigation */}
      <LessonNavigation prev={prev} next={next} />
    </div>
  );
}
