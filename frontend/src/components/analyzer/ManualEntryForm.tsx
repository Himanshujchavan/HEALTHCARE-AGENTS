import { useEffect, useMemo, useState } from "react";
import { HeartPulse, Syringe } from "lucide-react";
import { HealthInput, InputSchema, ParameterSpec } from "../../models/health";
import { SectionCard } from "../shared/SectionCard";
import { ParameterField } from "../shared/ParameterField";
import { SymptomSelector } from "./SymptomSelector";

interface ManualEntryFormProps {
  onSubmit: (payload: HealthInput) => void;
  isLoading?: boolean;
  /** The parameters the API accepts. The form renders exactly these. */
  schema?: InputSchema;
  schemaError?: string | null;
  schemaLoading?: boolean;
}

const CONDITION_LABELS: Record<string, string> = {
  diabetes: "Diabetes",
  cardiovascular: "Cardiovascular",
};

const CONDITION_ICONS: Record<string, typeof Syringe> = {
  diabetes: Syringe,
  cardiovascular: HeartPulse,
};

const BOOLEAN_LABELS = {
  unset: "Unknown — not provided",
  yes: "Current smoker",
  no: "Non-smoker",
};

/**
 * The manual entry form is driven entirely by `/input-schema`.
 *
 * The schema is produced from the disease plugins, so the fields rendered here
 * are exactly the fields the agents require, with the bounds they accept. A
 * new parameter therefore appears in the form as soon as the API accepts it,
 * without a frontend change.
 */
export function ManualEntryForm({
  onSubmit,
  isLoading,
  schema,
  schemaError,
  schemaLoading,
}: ManualEntryFormProps) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [customSymptom, setCustomSymptom] = useState("");
  const [manualText, setManualText] = useState("");
  const [useLLM, setUseLLM] = useState(false);
  const [conditions, setConditions] = useState<string[] | null>(null);

  const diseases = useMemo(
    () => Object.keys(schema?.diseases ?? {}),
    [schema?.diseases],
  );

  // The first condition is selected as soon as the schema arrives, but only
  // once: re-selecting it later would undo the user's choice.
  useEffect(() => {
    if (conditions === null && diseases.length > 0) {
      setConditions([diseases[0]]);
    }
  }, [conditions, diseases]);

  const selectedConditions = conditions ?? [];

  const toggleCondition = (condition: string) => {
    setConditions((prev) => {
      const current = prev ?? [];
      return current.includes(condition)
        ? current.filter((item) => item !== condition)
        : [...current, condition];
    });
  };

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

  const setValue = (name: string, value: string) => {
    setValues((prev) => ({ ...prev, [name]: value }));
  };

  // The fields to render: the union of what the selected conditions require
  // and what they accept, deduplicated, required fields first.
  const fields = useMemo(() => {
    if (!schema) {
      return [] as { name: string; spec: ParameterSpec; required: boolean }[];
    }
    const seen = new Set<string>();
    const required: string[] = [];
    const optional: string[] = [];

    selectedConditions.forEach((condition) => {
      const disease = schema.diseases[condition];
      if (!disease) {
        return;
      }
      disease.required.forEach((field) => {
        if (!seen.has(field)) {
          seen.add(field);
          required.push(field);
        }
      });
      disease.optional.forEach((field) => {
        if (!seen.has(field)) {
          seen.add(field);
          optional.push(field);
        }
      });
    });

    return [...required, ...optional]
      .filter((name) => schema.parameters[name])
      .map((name) => ({
        name,
        spec: schema.parameters[name],
        required: required.includes(name),
      }));
  }, [schema, selectedConditions]);

  // The same rule the API enforces: a submission is accepted only when at
  // least one condition has every required field.
  const unsatisfied = useMemo(() => {
    if (!schema) {
      return [] as { condition: string; gaps: string[] }[];
    }
    return selectedConditions
      .map((condition) => {
        const disease = schema.diseases[condition];
        if (!disease) {
          return null;
        }
        const gaps = disease.required.filter((field) => {
          const raw = values[field];
          return raw === undefined || raw === "";
        });
        return gaps.length > 0 ? { condition, gaps } : null;
      })
      .filter((entry): entry is { condition: string; gaps: string[] } => entry !== null);
  }, [schema, selectedConditions, values]);

  const canSubmit = unsatisfied.length < selectedConditions.length && selectedConditions.length > 0;

  const handleSubmit = () => {
    const payload: Partial<HealthInput> = {};

    fields.forEach(({ name, spec }) => {
      const raw = (values[name] ?? "").trim();
      if (raw === "") {
        return;
      }
      if (spec.type === "boolean") {
        if (raw === "true" || raw === "false") {
          (payload as Record<string, unknown>)[name] = raw === "true";
        }
        return;
      }
      const parsed = Number(raw);
      if (!Number.isNaN(parsed)) {
        (payload as Record<string, unknown>)[name] = parsed;
      }
    });

    onSubmit({
      ...(payload as HealthInput),
      symptoms,
      manual_text: manualText,
      use_llm: useLLM,
    });
  };

  return (
    <SectionCard title="Manual health entry">
      {schemaLoading && (
        <p className="mb-4 text-sm text-slate-600">Loading the parameters this API accepts…</p>
      )}
      {schemaError && (
        <p className="mb-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {schemaError}
        </p>
      )}

      {schema && (
        <form
          className="space-y-6"
          onSubmit={(event) => {
            event.preventDefault();
            handleSubmit();
          }}
        >
          <div className="flex flex-wrap gap-3">
            {diseases.map((condition) => {
              const Icon = CONDITION_ICONS[condition] ?? Syringe;
              const active = selectedConditions.includes(condition);
              return (
                <button
                  key={condition}
                  type="button"
                  onClick={() => toggleCondition(condition)}
                  className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ${
                    active
                      ? "bg-emerald-600 text-white"
                      : "border border-slate-200 text-slate-600"
                  }`}
                >
                  <Icon size={16} /> {CONDITION_LABELS[condition] ?? condition}
                </button>
              );
            })}
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            {fields.map(({ name, spec, required }) => (
              <ParameterField
                key={name}
                name={name}
                spec={spec}
                required={required}
                value={values[name] ?? ""}
                onChange={(value) => setValue(name, value)}
                booleanLabels={name === "smoker" ? BOOLEAN_LABELS : undefined}
                hint={
                  name === "smoker"
                    ? "Left as unknown, the score does not count smoking either way — it is not treated as a non-smoker."
                    : undefined
                }
              />
            ))}
          </div>

          {unsatisfied.length > 0 && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
              <p className="font-semibold">Some conditions are missing required values</p>
              <ul className="mt-1 list-disc pl-5">
                {unsatisfied.map(({ condition, gaps }) => (
                  <li key={condition}>
                    {CONDITION_LABELS[condition] ?? condition}: {gaps.join(", ")}
                  </li>
                ))}
              </ul>
            </div>
          )}

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

          <label className="block space-y-2 text-sm text-slate-600">
            Clinical context (optional)
            <textarea
              value={manualText}
              onChange={(event) => setManualText(event.target.value)}
              maxLength={1000}
              rows={3}
              placeholder="Family history, medication, recent complaints…"
              className="w-full rounded-xl border border-slate-200 px-4 py-3 text-sm text-slate-700"
            />
          </label>

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
            type="submit"
            disabled={isLoading || !canSubmit}
            className="rounded-full bg-emerald-600 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-emerald-200"
          >
            {isLoading ? "Analyzing..." : "Submit for analysis"}
          </button>
        </form>
      )}
    </SectionCard>
  );
}
