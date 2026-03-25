import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";

async function getOrCreateUser() {
  let user = await prisma.user.findFirst();
  if (!user) {
    user = await prisma.user.create({ data: { name: "Learner" } });
  }
  return user;
}

export async function GET() {
  try {
    const progress = await prisma.lessonProgress.findMany({
      orderBy: [{ moduleNumber: "asc" }, { lessonNumber: "asc" }],
    });
    return NextResponse.json(progress);
  } catch {
    return NextResponse.json([]);
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { lessonSlug, conceptsCompleted, labCompleted } = body;

    if (!lessonSlug) {
      return NextResponse.json({ error: "lessonSlug required" }, { status: 400 });
    }

    const user = await getOrCreateUser();
    const [moduleNumber, lessonNumber] = lessonSlug.split(".").map(Number);

    const existing = await prisma.lessonProgress.findUnique({
      where: { lessonSlug },
    });

    const data: Record<string, unknown> = { updatedAt: new Date() };
    if (conceptsCompleted !== undefined) data.conceptsCompleted = conceptsCompleted;
    if (labCompleted !== undefined) data.labCompleted = labCompleted;

    // Check if both are now complete
    const newConcepts = conceptsCompleted ?? existing?.conceptsCompleted ?? false;
    const newLab = labCompleted ?? existing?.labCompleted ?? false;
    if (newConcepts && newLab) {
      data.completedAt = new Date();
    }

    const progress = await prisma.lessonProgress.upsert({
      where: { lessonSlug },
      update: data,
      create: {
        userId: user.id,
        lessonSlug,
        moduleNumber,
        lessonNumber,
        conceptsCompleted: conceptsCompleted ?? false,
        labCompleted: labCompleted ?? false,
        completedAt: newConcepts && newLab ? new Date() : undefined,
      },
    });

    return NextResponse.json(progress);
  } catch (e) {
    console.error("Progress error:", e);
    return NextResponse.json({ error: "Server error" }, { status: 500 });
  }
}
