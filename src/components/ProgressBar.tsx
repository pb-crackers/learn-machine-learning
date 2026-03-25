export function ProgressBar({
  value,
  max = 100,
  size = "md",
  showLabel = true,
}: {
  value: number;
  max?: number;
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
}) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  const heights = { sm: "h-1.5", md: "h-2.5", lg: "h-4" };

  return (
    <div className="flex items-center gap-3">
      <div className={`flex-1 overflow-hidden rounded-full bg-slate-200 ${heights[size]}`}>
        <div
          className={`${heights[size]} rounded-full bg-blue-600 transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {showLabel && (
        <span className="text-sm font-medium text-slate-600">{pct}%</span>
      )}
    </div>
  );
}
