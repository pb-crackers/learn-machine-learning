import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { compileMDX } from "next-mdx-remote/rsc";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypePrettyCode from "rehype-pretty-code";
import { mdxComponents } from "@/components/mdx-components";

const contentDir = path.join(process.cwd(), "content");

export interface LessonFrontmatter {
  module: number;
  lesson: number;
  title: string;
  description: string;
  prerequisites?: string[];
  estimatedMinutes?: number;
}

export interface QuizQuestion {
  id: string;
  type: "multiple_choice" | "true_false" | "code_output" | "short_answer";
  question: string;
  options?: string[];
  correct: string;
  explanation: string;
}

export async function getLesson(slug: string) {
  const [moduleNum] = slug.split(".");
  const filePath = path.join(contentDir, `module-${moduleNum}`, `lesson-${slug}.mdx`);

  if (!fs.existsSync(filePath)) {
    return null;
  }

  const source = fs.readFileSync(filePath, "utf-8");
  const { data, content } = matter(source);
  const frontmatter = data as LessonFrontmatter;

  const { content: mdxContent } = await compileMDX({
    source: content,
    components: mdxComponents,
    options: {
      parseFrontmatter: false,
      mdxOptions: {
        remarkPlugins: [remarkMath],
        rehypePlugins: [
          rehypeKatex,
          [
            rehypePrettyCode,
            {
              theme: "one-dark-pro",
              keepBackground: true,
            },
          ],
        ],
      },
    },
  });

  return { frontmatter, content: mdxContent };
}

export function getQuiz(slug: string): QuizQuestion[] | null {
  const [moduleNum] = slug.split(".");
  const quizPath = path.join(contentDir, `module-${moduleNum}`, `quiz-${slug}.json`);

  if (!fs.existsSync(quizPath)) {
    return null;
  }

  const raw = fs.readFileSync(quizPath, "utf-8");
  const data = JSON.parse(raw);
  return data.questions as QuizQuestion[];
}
