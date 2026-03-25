import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";

async function getOrCreateUser() {
  let user = await prisma.user.findFirst();
  if (!user) {
    user = await prisma.user.create({ data: { name: "Learner" } });
  }
  return user;
}

export async function GET(req: NextRequest) {
  const slug = req.nextUrl.searchParams.get("lessonSlug");
  if (!slug) {
    return NextResponse.json({ error: "lessonSlug required" }, { status: 400 });
  }

  try {
    const results = await prisma.quizResult.findMany({
      where: { lessonSlug: slug },
      orderBy: { completedAt: "desc" },
    });
    return NextResponse.json(results);
  } catch {
    return NextResponse.json([]);
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { lessonSlug, score, totalQuestions, answers } = body;

    if (!lessonSlug || score === undefined || !totalQuestions || !answers) {
      return NextResponse.json({ error: "Missing fields" }, { status: 400 });
    }

    const user = await getOrCreateUser();

    const result = await prisma.quizResult.create({
      data: {
        userId: user.id,
        lessonSlug,
        score,
        totalQuestions,
        answers,
      },
    });

    return NextResponse.json(result);
  } catch (e) {
    console.error("Quiz error:", e);
    return NextResponse.json({ error: "Server error" }, { status: 500 });
  }
}
