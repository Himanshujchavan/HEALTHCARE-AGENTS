import { motion } from "framer-motion";
import { Link } from "react-router-dom";

const features = [
  "Upload lab reports in seconds",
  "Multi-agent risk intelligence",
  "Patient-friendly preventive insights",
  "Continuous monitoring and history",
];

export function LandingPage() {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#ecfdf3_0%,_transparent_45%),linear-gradient(135deg,_#fdfcfb_0%,_#f7efe5_100%)]">
      <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-6">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Preventive AI</p>
          <h1 className="text-2xl font-semibold text-slate-900">Health Agents</h1>
        </div>
        <div className="flex items-center gap-3">
          <Link
            to="/login"
            className="rounded-full border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700"
          >
            Login
          </Link>
          <Link
            to="/register"
            className="rounded-full bg-emerald-600 px-4 py-2 text-sm font-semibold text-white"
          >
            Get started
          </Link>
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl gap-12 px-6 pb-16 pt-6 lg:grid-cols-[1.1fr_0.9fr]">
        <motion.section
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="space-y-6"
        >
          <p className="text-xs uppercase tracking-[0.3em] text-emerald-600">
            Preventive care workflow
          </p>
          <h2 className="text-4xl font-semibold text-slate-900">
            Your personal AI care team for early risk detection.
          </h2>
          <p className="text-lg text-slate-600">
            Upload lab reports or enter key health metrics. Our multi-agent system analyzes
            trends, predicts risk, and crafts a patient-friendly plan in minutes.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link
              to="/register"
              className="rounded-full bg-emerald-600 px-5 py-3 text-sm font-semibold text-white"
            >
              Create account
            </Link>
            <Link
              to="/login"
              className="rounded-full border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-700"
            >
              Sign in
            </Link>
          </div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1 }}
          className="rounded-3xl border border-slate-200 bg-white/80 p-6 shadow-sm"
        >
          <h3 className="text-lg font-semibold text-slate-900">What you get</h3>
          <ul className="mt-4 space-y-3 text-sm text-slate-600">
            {features.map((feature) => (
              <li key={feature} className="flex items-start gap-3">
                <span className="mt-1 h-2 w-2 rounded-full bg-emerald-500" />
                {feature}
              </li>
            ))}
          </ul>
          <div className="mt-6 rounded-2xl bg-emerald-50 p-4 text-sm text-emerald-900">
            Live workflow updates, clear risk visuals, and action-ready reports.
          </div>
        </motion.section>
      </main>
    </div>
  );
}
