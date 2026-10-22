import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ResultsOverview } from "../components/results/ResultsOverview";
import { StatusBadge, statusLabel } from "../components/shared/StatusBadge";
import { DiseaseAssessment, UnifiedAssessment } from "../models/health";

function assessment(overrides: Partial<DiseaseAssessment>): DiseaseAssessment {
  return {
    status: "completed",
    risk_level: "Low",
    score: 0.1,
    factors: [],
    ...overrides,
  };
}

function unified(assessments: Record<string, DiseaseAssessment>): UnifiedAssessment {
  return {
    status: "completed",
    assessments: assessments as UnifiedAssessment["assessments"],
    differential: { status: "unavailable", results: null },
    trend: { status: "insufficient_history" },
    alerts: [],
    explanation: "An explanation.",
    agent_status: {},
    model_disclaimer: "These are model predictions, not confirmed medical diagnoses.",
  };
}

function card(title: string): HTMLElement {
  const heading = screen.getByRole("heading", { name: title });
  return heading.closest("section") as HTMLElement;
}

describe("the four assessment states", () => {
  it("shows a completed assessment with its score", () => {
    render(
      <ResultsOverview
        analysis={{
          unified: unified({
            diabetes: assessment({ status: "completed", risk_level: "Low", score: 0.12 }),
          }),
        }}
      />,
    );

    const diabetes = card("Diabetes risk assessment");
    expect(within(diabetes).getByText("Model score: 0.12")).toBeInTheDocument();
    expect(within(diabetes).queryByText(/could not run/)).not.toBeInTheDocument();
  });

  it("shows an assessment completed with errors as a reduced score", () => {
    render(
      <ResultsOverview
        analysis={{
          unified: unified({
            diabetes: assessment({
              status: "completed_with_errors",
              risk_level: "Moderate",
              score: 0.44,
            }),
          }),
        }}
      />,
    );

    const diabetes = card("Diabetes risk assessment");
    // The score is still shown — it was produced.
    expect(within(diabetes).getByText("Model score: 0.44")).toBeInTheDocument();
    // But it is reported as resting on less than the full formula.
    expect(
      within(diabetes).getByText(/some of its components could not run/i),
    ).toBeInTheDocument();
  });

  it("shows a failed assessment as a failure, not as a missing score", () => {
    render(
      <ResultsOverview
        analysis={{
          unified: unified({
            diabetes: assessment({
              status: "failed",
              risk_level: "Unavailable",
              score: null,
              reason: "The risk agent raised an error.",
            }),
          }),
        }}
      />,
    );

    const diabetes = card("Diabetes risk assessment");
    expect(within(diabetes).getAllByText("Failed").length).toBeGreaterThan(0);
    expect(within(diabetes).getByText(/the assessment for this condition failed/i)).toBeInTheDocument();
    expect(within(diabetes).getByText("The risk agent raised an error.")).toBeInTheDocument();
    expect(within(diabetes).queryByText(/model score:/i)).not.toBeInTheDocument();
  });

  it("shows an unavailable assessment as one that could not be assessed", () => {
    render(
      <ResultsOverview
        analysis={{
          unified: unified({
            diabetes: assessment({
              status: "unavailable",
              risk_level: "Unavailable",
              score: null,
            }),
          }),
        }}
      />,
    );

    const diabetes = card("Diabetes risk assessment");
    expect(within(diabetes).getAllByText("Unavailable").length).toBeGreaterThan(0);
    expect(within(diabetes).getByText(/not enough data to assess/i)).toBeInTheDocument();
    expect(within(diabetes).queryByText(/model score:/i)).not.toBeInTheDocument();
  });

  it("shows a condition that was never applicable as not applicable", () => {
    render(
      <ResultsOverview
        analysis={{
          unified: unified({
            cardiovascular: assessment({
              status: "not_applicable",
              risk_level: "Unavailable",
              score: null,
            }),
          }),
        }}
      />,
    );

    const cardiovascular = card("Cardiovascular risk assessment");
    expect(within(cardiovascular).getAllByText("Not applicable").length).toBeGreaterThan(0);
    expect(
      within(cardiovascular).getByText(/no required values were submitted/i),
    ).toBeInTheDocument();
  });
});

describe("the status vocabulary", () => {
  it("labels every state the backend can report", () => {
    expect(statusLabel("completed")).toBe("Completed");
    expect(statusLabel("completed_with_errors")).toBe("Completed with errors");
    expect(statusLabel("failed")).toBe("Failed");
    expect(statusLabel("unavailable")).toBe("Unavailable");
    expect(statusLabel("not_applicable")).toBe("Not applicable");
    expect(statusLabel("skipped")).toBe("Skipped");
    expect(statusLabel("pending")).toBe("Pending");
    expect(statusLabel("partial")).toBe("Partial");
    expect(statusLabel(undefined)).toBe("Unknown");
  });

  it("gives the degraded states their own colour", () => {
    render(
      <>
        <StatusBadge status="completed" />
        <StatusBadge status="completed_with_errors" />
        <StatusBadge status="failed" />
        <StatusBadge status="unavailable" />
      </>,
    );

    const badges = screen.getAllByTestId("status-badge");
    const classes = badges.map((badge) => badge.className);
    expect(new Set(classes).size).toBe(4);
  });
});
