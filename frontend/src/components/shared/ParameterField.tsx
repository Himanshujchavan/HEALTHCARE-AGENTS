import { ParameterSpec } from "../../models/health";

interface ParameterFieldProps {
  name: string;
  spec: ParameterSpec;
  /** The current value, as the raw string the form holds. */
  value: string;
  onChange: (value: string) => void;
  required?: boolean;
  /** Extra guidance shown under the control. */
  hint?: string;
  /** Labels for the three states of a boolean field. */
  booleanLabels?: { unset: string; yes: string; no: string };
}

const DEFAULT_BOOLEAN_LABELS = {
  unset: "Not provided",
  yes: "Yes",
  no: "No",
};

/**
 * One schema-driven input.
 *
 * The field's type comes from `/input-schema`, so the form cannot drift from
 * what the API accepts: a boolean stays a three-state choice (an absent value
 * is not the same as a negative one), a bounded integer becomes a code
 * selector, and a measurement becomes a bounded number input.
 */
export function ParameterField({
  name,
  spec,
  value,
  onChange,
  required = false,
  hint,
  booleanLabels = DEFAULT_BOOLEAN_LABELS,
}: ParameterFieldProps) {
  const label = spec.label || name;
  const unit = spec.unit ? ` (${spec.unit})` : "";
  const inputClass =
    "w-full rounded-xl border border-slate-200 px-4 py-3 text-sm text-slate-800";

  if (spec.type === "boolean") {
    return (
      <label className="space-y-2 text-sm text-slate-600">
        {label}
        {unit}
        {required && <span className="text-amber-700"> (required)</span>}
        <select
          data-testid={`field-${name}`}
          className={inputClass}
          value={value}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">{booleanLabels.unset}</option>
          <option value="true">{booleanLabels.yes}</option>
          <option value="false">{booleanLabels.no}</option>
        </select>
        {hint && <span className="block text-xs text-slate-500">{hint}</span>}
      </label>
    );
  }

  const min = spec.min ?? undefined;
  const max = spec.max ?? undefined;
  // A small bounded integer is an ordinal code, not a measurement: a select
  // keeps the user from typing a code the API would reject.
  const isCode = spec.type === "integer" && min !== undefined && max !== undefined && max - min <= 10;

  if (isCode) {
    const codes: number[] = [];
    for (let code = min as number; code <= (max as number); code += 1) {
      codes.push(code);
    }
    return (
      <label className="space-y-2 text-sm text-slate-600">
        {label}
        {unit}
        {required && <span className="text-amber-700"> (required)</span>}
        <select
          data-testid={`field-${name}`}
          className={inputClass}
          value={value}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">Not provided</option>
          {codes.map((code) => (
            <option key={code} value={code}>
              {code}
            </option>
          ))}
        </select>
        {hint && <span className="block text-xs text-slate-500">{hint}</span>}
      </label>
    );
  }

  return (
    <label className="space-y-2 text-sm text-slate-600">
      {label}
      {unit}
      {required && <span className="text-amber-700"> (required)</span>}
      <input
        data-testid={`field-${name}`}
        type="number"
        inputMode="decimal"
        step={spec.type === "integer" ? 1 : "any"}
        min={min}
        max={max}
        className={inputClass}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
      {hint && <span className="block text-xs text-slate-500">{hint}</span>}
    </label>
  );
}
