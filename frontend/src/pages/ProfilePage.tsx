import { SectionCard } from "../components/shared/SectionCard";

export function ProfilePage() {
  return (
    <SectionCard title="Profile">
      <div className="space-y-3 text-sm text-slate-600">
        <p>Manage your account settings and notification preferences.</p>
        <button className="rounded-full border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700">
          Update password
        </button>
      </div>
    </SectionCard>
  );
}
