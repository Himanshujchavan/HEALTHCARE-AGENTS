import { NavLink, useLocation } from "react-router-dom";

const links = [
  { label: "Dashboard", to: "/dashboard" },
  { label: "Upload PDF", to: "/analyzer?tab=pdf", variant: "accent" },
  { label: "Manual Entry", to: "/analyzer?tab=manual", variant: "accent" },
  { label: "History", to: "/history" },
  { label: "Profile", to: "/profile" },
];

export function Sidebar() {
  const location = useLocation();
  const searchParams = new URLSearchParams(location.search);
  const tab = searchParams.get("tab") || "pdf";

  const isLinkActive = (to: string) => {
    if (to.startsWith("/analyzer")) {
      if (to.includes("tab=manual")) {
        return location.pathname === "/analyzer" && tab === "manual";
      }
      return location.pathname === "/analyzer" && tab === "pdf";
    }
    return location.pathname === to;
  };

  return (
    <aside className="flex h-full w-64 flex-col border-r border-slate-200 bg-white/80 px-6 py-8">
      <div className="mb-10">
        <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Preventive AI</p>
        <h1 className="mt-2 text-2xl font-semibold text-slate-900">Health Agents</h1>
      </div>

      <nav className="flex flex-1 flex-col gap-2">
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            className={() => {
              const active = isLinkActive(link.to);
              const base = "rounded-xl px-4 py-3 text-sm font-medium transition";
              const activeStyle = "bg-emerald-100 text-emerald-900";
              const idleStyle = "text-slate-600 hover:bg-slate-100";
              const accentIdle = "border border-emerald-100 text-emerald-900 hover:bg-emerald-50";

              if (active) {
                return `${base} ${activeStyle}`;
              }

              if (link.variant === "accent") {
                return `${base} ${accentIdle}`;
              }

              return `${base} ${idleStyle}`;
            }}
          >
            {link.label}
          </NavLink>
        ))}
      </nav>

      <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-xs text-slate-600">
        Your care plan updates live as new reports arrive.
      </div>
    </aside>
  );
}
