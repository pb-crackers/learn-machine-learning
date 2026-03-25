export function MathNote({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-6 rounded-lg border-l-4 border-blue-500 bg-blue-50 p-4">
      <p className="mb-1 text-sm font-semibold uppercase tracking-wide text-blue-700">
        Math Note
      </p>
      <div className="text-blue-900">{children}</div>
    </div>
  );
}
