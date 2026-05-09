interface SectionHeaderProps {
  eyebrow?: string;
  title: string;
  subtitle?: string;
}

export function SectionHeader({ eyebrow, title, subtitle }: SectionHeaderProps) {
  return (
    <div className="space-y-1">
      {eyebrow && (
        <p className="text-xs uppercase tracking-[0.3em] text-slate-500">{eyebrow}</p>
      )}
      <h2 className="text-2xl font-semibold text-slate-900">{title}</h2>
      {subtitle && <p className="text-sm text-slate-600">{subtitle}</p>}
    </div>
  );
}
