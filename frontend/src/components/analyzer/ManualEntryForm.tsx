import { useState } from "react";
import { HeartPulse, Syringe } from "lucide-react";
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
      symptoms: [],
      manual_text: "",
      use_llm: false,
    },
  });
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [customSymptom, setCustomSymptom] = useState("");
  const [disease, setDisease] = useState<"diabetes" | "cardiovascular">("diabetes");

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
        onSubmit={handleSubmit((values) => {
          const fields = disease === "diabetes"
            ? ["hba1c", "glucose", "bmi", "age"]
            : ["age", "systolic_bp", "diastolic_bp", "total_cholesterol", "ldl", "hdl", "smoker"];
          const selectedValues = Object.fromEntries(
            fields
              .filter((field) => values[field as keyof HealthInput] !== undefined && values[field as keyof HealthInput] !== null)
              .map((field) => [field, values[field as keyof HealthInput]]),
          ) as HealthInput;
          onSubmit({ ...selectedValues, symptoms, manual_text: values.manual_text, use_llm: values.use_llm });
        })}
      >
        <div className="flex gap-3">
          <button type="button" onClick={() => setDisease("diabetes")} className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ${disease === "diabetes" ? "bg-emerald-600 text-white" : "border border-slate-200 text-slate-600"}`}>
            <Syringe size={16} /> Diabetes
          </button>
          <button type="button" onClick={() => setDisease("cardiovascular")} className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ${disease === "cardiovascular" ? "bg-emerald-600 text-white" : "border border-slate-200 text-slate-600"}`}>
            <HeartPulse size={16} /> Cardiovascular
          </button>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {disease === "diabetes" ? <>
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
            <input type="number" step="0.1" className="w-full rounded-xl border border-slate-200 px-4 py-3" {...register("glucose", { valueAsNumber: true })} />
          </label>
          <label className="space-y-2 text-sm text-slate-600">
            BMI
            <input type="number" step="0.1" className="w-full rounded-xl border border-slate-200 px-4 py-3" {...register("bmi", { valueAsNumber: true })} />
          </label>
          <label className="space-y-2 text-sm text-slate-600">
            Age
            <input type="number" className="w-full rounded-xl border border-slate-200 px-4 py-3" {...register("age", { valueAsNumber: true })} />
          </label>
          </> : <>
          <label className="space-y-2 text-sm text-slate-600">Age<input type="number" className="w-full rounded-xl border border-slate-200 px-4 py-3" {...register("age", { valueAsNumber: true })} /></label>
          <label className="space-y-2 text-sm text-slate-600">Systolic BP<input type="number" className="w-full rounded-xl border border-slate-200 px-4 py-3" {...register("systolic_bp", { valueAsNumber: true })} /></label>
          <label className="space-y-2 text-sm text-slate-600">Diastolic BP<input type="number" className="w-full rounded-xl border border-slate-200 px-4 py-3" {...register("diastolic_bp", { valueAsNumber: true })} /></label>
          <label className="space-y-2 text-sm text-slate-600">Total cholesterol<input type="number" className="w-full rounded-xl border border-slate-200 px-4 py-3" {...register("total_cholesterol", { valueAsNumber: true })} /></label>
          <label className="space-y-2 text-sm text-slate-600">LDL<input type="number" className="w-full rounded-xl border border-slate-200 px-4 py-3" {...register("ldl", { valueAsNumber: true })} /></label>
          <label className="space-y-2 text-sm text-slate-600">HDL<input type="number" className="w-full rounded-xl border border-slate-200 px-4 py-3" {...register("hdl", { valueAsNumber: true })} /></label>
          <label className="flex items-center gap-2 text-sm text-slate-600"><input type="checkbox" {...register("smoker")} /> Smoker</label>
          </>}
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
