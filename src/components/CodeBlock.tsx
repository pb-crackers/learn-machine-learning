"use client";

import { useState } from "react";

export function CodeBlock({
  children,
  className,
}: {
  children: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(children);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="group relative">
      <button
        onClick={handleCopy}
        className="absolute right-2 top-2 rounded bg-slate-700 px-2 py-1 text-xs text-slate-300 opacity-0 transition-opacity hover:bg-slate-600 group-hover:opacity-100"
      >
        {copied ? "Copied!" : "Copy"}
      </button>
      <pre className={className}>
        <code>{children}</code>
      </pre>
    </div>
  );
}
