import type { ReactNode } from "react";

interface SectionCardProps {
  title?: string;
  action?: ReactNode;
  className?: string;
  children: ReactNode;
}

export function SectionCard({ title, action, className, children }: SectionCardProps) {
  return (
    <section
      className={`rounded-2xl border border-slate-200 bg-white/90 p-7 md:p-8 shadow-sm ${className ?? ""}`}
    >
      {(title || action) && (
        <div className="mb-5 flex items-center justify-between">
          {title && <h3 className="text-lg font-semibold text-slate-900">{title}</h3>}
          {action}
        </div>
      )}
      {children}
    </section>
  );
}
