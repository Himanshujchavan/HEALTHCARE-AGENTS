import { useState } from "react";
import { useForm } from "react-hook-form";
import { HealthInput } from "../../models/health";
import { SectionCard } from "../shared/SectionCard";
import { SymptomSelector } from "./SymptomSelector";

interface ManualEntryFormProps {
  onSubmit: (payload: HealthInput) => void;
  isLoading?: boolean;
}

export function ManualEntryForm({ onSubmit, isLoading }: ManualEntryFormProps) {
  const { register, handleSubmit } = useForm<HealthInput>({
    defaultValues: {
      hba1c: 0,
      glucose: 0,
      bmi: 0,
      age: 0,
      symptoms: [],
      manual_text: "",
      use_llm: false,
    },
  });
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [customSymptom, setCustomSymptom] = useState("");

  const toggleSymptom = (symptom: string) => {
    setSymptoms((prev) =>
      prev.includes(symptom) ? prev.filter((item) => item !== symptom) : [...prev, symptom],
    );
  };

  const handleAddCustom = () => {
    const trimmed = customSymptom.trim();
    if (!trimmed) {
      return;
    }
    if (!symptoms.includes(trimmed)) {
      setSymptoms((prev) => [...prev, trimmed]);
    }
    setCustomSymptom("");
  };

  return (
    <SectionCard title="Manual health entry">
      <form
        className="space-y-6"
        onSubmit={handleSubmit((values) => onSubmit({ ...values, symptoms }))}
      >
        <div className="grid gap-4 md:grid-cols-2">
          <label className="space-y-2 text-sm text-slate-600">
            HbA1c (%)
            <input
              type="number"
              step="0.1"
              className="w-full rounded-xl border border-slate-200 px-4 py-3"
              {...register("hba1c", { valueAsNumber: true })}
            />
          </label>
          <label className="space-y-2 text-sm text-slate-600">
            Glucose (mg/dL)
            <input
              type="number"
              step="0.1"
              className="w-full rounded-xl border border-slate-200 px-4 py-3"
              {...register("glucose", { valueAsNumber: true })}
            />
          </label>
          <label className="space-y-2 text-sm text-slate-600">
            BMI
            <input
              type="number"
              step="0.1"
              className="w-full rounded-xl border border-slate-200 px-4 py-3"
              {...register("bmi", { valueAsNumber: true })}
            />
          </label>
          <label className="space-y-2 text-sm text-slate-600">
            Age
            <input
              type="number"
              className="w-full rounded-xl border border-slate-200 px-4 py-3"
              {...register("age", { valueAsNumber: true })}
            />
          </label>
        </div>

        <div className="space-y-3">
          <p className="text-sm font-semibold text-slate-900">Symptoms</p>
          <SymptomSelector
            selected={symptoms}
            onToggle={toggleSymptom}
            customSymptom={customSymptom}
            onCustomSymptomChange={setCustomSymptom}
            onAddCustomSymptom={handleAddCustom}
          />
        </div>

        <label className="flex items-start gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700">
          <input
            type="checkbox"
            className="mt-1 h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
            {...register("use_llm")}
          />
          <span>
            <span className="font-semibold text-slate-900">Use LLM enhancement for nuanced summaries</span>
            <br />
            <span className="text-slate-500">Generates a more detailed alert narrative when the model is available.</span>
          </span>
        </label>

        <button
          type="submit"
          disabled={isLoading}
          className="rounded-full bg-emerald-600 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-emerald-200"
        >
          {isLoading ? "Analyzing..." : "Submit for analysis"}
        </button>
      </form>
    </SectionCard>
  );
}
