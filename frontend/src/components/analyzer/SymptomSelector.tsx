import { symptomOptions } from "../../constants/symptoms";

interface SymptomSelectorProps {
  selected: string[];
  onToggle: (symptom: string) => void;
  customSymptom: string;
  onCustomSymptomChange: (value: string) => void;
  onAddCustomSymptom: () => void;
}

export function SymptomSelector({
  selected,
  onToggle,
  customSymptom,
  onCustomSymptomChange,
  onAddCustomSymptom,
}: SymptomSelectorProps) {
  const customSymptoms = selected.filter((symptom) => !symptomOptions.includes(symptom));

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap gap-3">
        {symptomOptions.map((symptom) => {
          const active = selected.includes(symptom);
          return (
            <button
              type="button"
              key={symptom}
              onClick={() => onToggle(symptom)}
              className={`rounded-full border px-3.5 py-2 text-sm font-semibold transition ${
                active
                  ? "border-emerald-500 bg-emerald-50 text-emerald-800"
                  : "border-slate-200 bg-white text-slate-600 hover:border-emerald-200"
              }`}
            >
              {symptom}
            </button>
          );
        })}
      </div>

      {customSymptoms.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Custom symptoms</p>
          <div className="flex flex-wrap gap-2">
            {customSymptoms.map((symptom) => (
              <button
                type="button"
                key={symptom}
                onClick={() => onToggle(symptom)}
                className="rounded-full border border-slate-200 bg-slate-900 px-4 py-2 text-sm font-semibold text-white"
              >
                {symptom} · remove
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <input
          type="text"
          value={customSymptom}
          onChange={(event) => onCustomSymptomChange(event.target.value)}
          placeholder="Add a custom symptom"
          className="flex-1 rounded-xl border border-slate-200 px-4 py-3 text-sm text-slate-700 placeholder:text-slate-400"
        />
        <button
          type="button"
          onClick={onAddCustomSymptom}
          className="rounded-xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white"
        >
          Add symptom
        </button>
      </div>
    </div>
  );
}
