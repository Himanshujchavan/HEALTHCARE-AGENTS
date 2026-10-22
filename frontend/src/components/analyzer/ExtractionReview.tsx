import { useMemo, useState } from "react";
import { ExtractionPreview, InputSchema, ParameterSpec } from "../../models/health";
import { SectionCard } from "../shared/SectionCard";
import { ParameterField } from "../shared/ParameterField";
import { StatusBadge } from "../shared/StatusBadge";

interface ExtractionReviewProps {
  preview: ExtractionPreview;
  schema?: InputSchema;
  /** Called with the confirmed values, ready to be sent to the confirm endpoint. */
  onConfirm: (values: Record<string, number | boolean>) => void;
  onCancel: () => void;
  isLoading?: boolean;
}

const BOOLEAN_LABELS = {
  unset: "Unknown — not provided",
  yes: "Current smoker",
  no: "Non-smoker",
};

function displayName(name: string, schema?: InputSchema): string {
  return schema?.parameters[name]?.label || name.replace(/_/g, " ");
}

/**
 * The correction step of the report flow.
 *
 * The extraction is shown exactly as it was read from the report — value,
 * clinical status, unit and any note — and every value can be changed before
 * the assessment runs. Values the report did not contain are offered too, so
 * a user can supply what is missing rather than watching the assessment
 * refuse to run.
 */
export function ExtractionReview({
  preview,
  schema,
  onConfirm,
  onCancel,
  isLoading,
}: ExtractionReviewProps) {
  const extractedNames = Object.keys(preview.extracted_parameters ?? {});
  const missingByDisease = preview.missing_fields ?? {};

  const missingNames = useMemo(() => {
    const names: string[] = [];
    Object.values(missingByDisease).forEach((fields) => {
      fields.forEach((field) => {
        if (!names.includes(field)) {
          names.push(field);
        }
      });
    });
    return names;
  }, [missingByDisease]);

  const optionalNames = useMemo(() => {
    if (!schema) {
      return [] as string[];
    }
    const names: string[] = [];
    (preview.available_diseases ?? []).forEach((disease) => {
      const spec = schema.diseases[disease];
      spec?.optional?.forEach((field) => {
        if (!names.includes(field) && !extractedNames.includes(field)) {
          names.push(field);
        }
      });
    });
    return names;
  }, [schema, preview.available_diseases, extractedNames]);

  const [values, setValues] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {};
    Object.entries(preview.extracted_parameters ?? {}).forEach(([name, entry]) => {
      initial[name] = entry?.value === null || entry?.value === undefined ? "" : String(entry.value);
    });
    return initial;
  });

  const setValue = (name: string, value: string) => {
    setValues((prev) => ({ ...prev, [name]: value }));
  };

  // The same rule the API enforces: the confirmation is only accepted when
  // at least one condition has every required value.
  const canConfirm = useMemo(() => {
    if (!schema) {
      return true;
    }
    const diseases = Object.keys(schema.diseases);
    return diseases.some((disease) => {
      const required = schema.diseases[disease]?.required ?? [];
      return required.every((field) => {
        const raw = values[field];
        return raw !== undefined && raw !== "";
      });
    });
  }, [schema, values]);

  const handleConfirm = () => {
    const payload: Record<string, number | boolean> = {};
    const names = [...extractedNames, ...missingNames, ...optionalNames];

    names.forEach((name) => {
      const raw = (values[name] ?? "").trim();
      if (raw === "") {
        return;
      }
      const type = schema?.parameters[name]?.type;
      if (type === "boolean") {
        if (raw === "true" || raw === "false") {
          payload[name] = raw === "true";
        }
        return;
      }
      const parsed = Number(raw);
      if (!Number.isNaN(parsed)) {
        payload[name] = parsed;
      }
    });

    onConfirm(payload);
  };

  const renderField = (name: string) => {
    const spec =
      schema?.parameters[name] ??
      // The schema may not have loaded (or may have failed). Fall back to a
      // plain measurement so the value can still be corrected rather than
      // silently dropped.
      ({
        min: null,
        max: null,
        unit: "",
        label: displayName(name, schema),
        required_for: [],
        type: name === "smoker" ? "boolean" : "number",
      } as ParameterSpec);
    const extracted = preview.extracted_parameters?.[name];
    return (
      <div key={name} className="space-y-2">
        <ParameterField
          name={name}
          spec={spec}
          value={values[name] ?? ""}
          onChange={(value) => setValue(name, value)}
          booleanLabels={name === "smoker" ? BOOLEAN_LABELS : undefined}
          hint={name === "smoker" ? "Left as unknown, the score does not count smoking either way." : undefined}
        />
        {extracted && (
          <p className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
            <StatusBadge status={extracted.status} />
            <span>
              {extracted.status === "Unknown" ? "No clinical status" : extracted.status}
              {extracted.unit ? ` · ${extracted.unit}` : ""}
            </span>
            {extracted.note && <span className="italic">{extracted.note}</span>}
          </p>
        )}
      </div>
    );
  };

  return (
    <SectionCard title="Confirm the extracted values">
      <div className="space-y-6">
        <div className="rounded-xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-900">
          <p className="font-semibold">{preview.filename}</p>
          <p className="mt-1">{preview.message}</p>
          <p className="mt-2 flex flex-wrap items-center gap-2 text-xs">
            <span>Assessment can run for:</span>
            {preview.available_diseases?.length ? (
              preview.available_diseases.map((disease) => (
                <span
                  key={disease}
                  className="rounded-full border border-sky-200 bg-white px-2 py-0.5 font-semibold"
                >
                  {disease}
                </span>
              ))
            ) : (
              <span className="font-semibold">no condition yet</span>
            )}
          </p>
        </div>

        <div>
          <p className="text-sm font-semibold text-slate-900">
            Extracted from the report
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Correct any value that was read incorrectly before the assessment runs.
          </p>
          <div className="mt-3 grid gap-4 md:grid-cols-2">
            {extractedNames.length > 0 ? (
              extractedNames.map(renderField)
            ) : (
              <p className="text-sm text-slate-600">No values were extracted.</p>
            )}
          </div>
        </div>

        {missingNames.length > 0 && (
          <div>
            <p className="text-sm font-semibold text-slate-900">Still missing</p>
            <p className="mt-1 text-xs text-slate-500">
              These values are required for a condition to be assessed. Add the ones
              the report did not contain.
            </p>
            <ul className="mt-2 space-y-1 text-xs text-slate-600">
              {Object.entries(missingByDisease).map(([disease, fields]) => (
                <li key={disease}>
                  {disease}: {fields.join(", ")}
                </li>
              ))}
            </ul>
            <div className="mt-3 grid gap-4 md:grid-cols-2">
              {missingNames.map(renderField)}
            </div>
          </div>
        )}

        {optionalNames.length > 0 && (
          <details className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
            <summary className="cursor-pointer text-sm font-semibold text-slate-900">
              Add optional values
            </summary>
            <p className="mt-2 text-xs text-slate-500">
              Optional values refine the assessment. Leaving one out is recorded as
              &quot;not provided&quot;, not as a negative value.
            </p>
            <div className="mt-3 grid gap-4 md:grid-cols-2">
              {optionalNames.map(renderField)}
            </div>
          </details>
        )}

        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            disabled={isLoading || !canConfirm}
            onClick={handleConfirm}
            className="rounded-full bg-emerald-600 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-emerald-200"
          >
            {isLoading ? "Assessing..." : "Confirm and assess"}
          </button>
          <button
            type="button"
            onClick={onCancel}
            className="rounded-full border border-slate-200 px-6 py-3 text-sm font-semibold text-slate-600"
          >
            Start over
          </button>
        </div>

        {!canConfirm && (
          <p className="text-sm text-amber-800">
            Fill in the missing values of at least one condition before the
            assessment can run.
          </p>
        )}
      </div>
    </SectionCard>
  );
}
