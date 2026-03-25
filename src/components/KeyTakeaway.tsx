export function KeyTakeaway({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-6 rounded-lg border-l-4 border-emerald-500 bg-emerald-50 p-4">
      <p className="mb-1 text-sm font-semibold uppercase tracking-wide text-emerald-700">
        Key Takeaway
      </p>
      <div className="text-emerald-900">{children}</div>
    </div>
  );
}
