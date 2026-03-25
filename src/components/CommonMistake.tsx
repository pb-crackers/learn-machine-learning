export function CommonMistake({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-6 rounded-lg border-l-4 border-red-500 bg-red-50 p-4">
      <p className="mb-1 text-sm font-semibold uppercase tracking-wide text-red-700">
        Common Mistake
      </p>
      <div className="text-red-900">{children}</div>
    </div>
  );
}
