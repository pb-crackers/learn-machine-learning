import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";

export async function GET() {
  try {
    let user = await prisma.user.findFirst();
    if (!user) {
      user = await prisma.user.create({ data: { name: "Learner" } });
    }
    return NextResponse.json(user);
  } catch (e) {
    console.error("User error:", e);
    return NextResponse.json({ error: "Server error" }, { status: 500 });
  }
}
