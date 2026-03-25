export function WhyThisMatters({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-6 rounded-lg border-l-4 border-amber-500 bg-amber-50 p-4">
      <p className="mb-1 text-sm font-semibold uppercase tracking-wide text-amber-700">
        Why This Matters
      </p>
      <div className="text-amber-900">{children}</div>
    </div>
  );
}
