import { useState } from "react";
import { SectionCard } from "../shared/SectionCard";
import { SymptomSelector } from "./SymptomSelector";

interface PdfUploadPanelProps {
  onSubmit: (
    file: File,
    useLLM: boolean,
    symptoms: string[],
    manualText: string,
  ) => void;
  isLoading?: boolean;
}

export function PdfUploadPanel({ onSubmit, isLoading }: PdfUploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [useLLM, setUseLLM] = useState(false);
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [customSymptom, setCustomSymptom] = useState("");
  const [manualText, setManualText] = useState("");

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
    <SectionCard
      title="Upload a lab report"
      className="border-dashed border-emerald-200 bg-white/70"
    >
      <div className="flex flex-col gap-6">
        <label className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-emerald-200 bg-emerald-50/30 px-6 py-12 text-center">
          <input
            type="file"
            accept="application/pdf"
            className="hidden"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
          <p className="text-sm font-semibold text-emerald-900">
            Drag your PDF here or click to upload
          </p>
          <p className="mt-2 text-xs text-emerald-700">
            Supported: PDF lab reports up to 10MB
          </p>
        </label>

        {file && (
          <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
            {file.name}
          </div>
        )}

        <div className="space-y-4">
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
    checked={useLLM}
    onChange={(event) => setUseLLM(event.target.checked)}
    className="mt-1 h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
  />

  <span>
    <span className="font-semibold text-slate-900">
      Use LLM enhancement for nuanced summaries
    </span>

    <br />

    <span className="text-slate-500">
      Generates a more detailed alert narrative when the model is available.
    </span>
  </span>
</label>     
        <button
          type="button"
          disabled={!file || isLoading}
          onClick={() => file && onSubmit(file, useLLM, symptoms, manualText)}
          className="rounded-full bg-emerald-600 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-emerald-200"
        >
          {isLoading ? "Analyzing..." : "Analyze report"}
        </button>
      </div>
    </SectionCard>
  );
}
