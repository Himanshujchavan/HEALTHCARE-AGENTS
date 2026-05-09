export function TopBar() {
  return (
    <header className="flex items-center justify-between border-b border-slate-200 bg-white/70 px-8 py-4 backdrop-blur">
      <div>
        <p className="text-sm text-slate-500">Welcome back</p>
        <h2 className="text-xl font-semibold text-slate-900">Your preventive health cockpit</h2>
      </div>
      <div className="flex items-center gap-3">
        <button className="rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100">
          View alerts
        </button>
        <button className="rounded-full bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-emerald-700">
          New analysis
        </button>
      </div>
    </header>
  );
}
