import { ReactElement } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import * as healthApi from "../api/health";
import { AnalyzerPage } from "../pages/AnalyzerPage";
import { HistoryPage } from "../pages/HistoryPage";
import {
  confirmResponse,
  extractionPreview,
  historyResponse,
  historySummary,
  inputSchema,
} from "./fixtures";

vi.mock("../api/health", () => ({
  getInputSchema: vi.fn(),
  previewHealthReport: vi.fn(),
  confirmExtraction: vi.fn(),
  submitHealthData: vi.fn(),
  getAssessmentHistory: vi.fn(),
  getAssessmentHistorySummary: vi.fn(),
  reanalyzeRecord: vi.fn(),
  deleteHealthRecord: vi.fn(),
  getHealthRecord: vi.fn(),
  getHealthRecordAnalysis: vi.fn(),
  getLatestRecord: vi.fn(),
  getHealthRecords: vi.fn(),
  uploadHealthReport: vi.fn(),
}));

const mockedApi = healthApi as unknown as {
  [key: string]: ReturnType<typeof vi.fn>;
};

function renderWithProviders(ui: ReactElement, initialEntries: string[] = ["/analyzer?tab=pdf"]) {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={initialEntries}>
        <Routes>
          <Route path="/analyzer" element={ui} />
          <Route path="/history" element={ui} />
          <Route path="/results/:recordId" element={<div>Results page</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

/** The card whose heading matches, so assertions can be scoped to it. */
function cardByTitle(title: string): HTMLElement {
  const heading = screen.getByRole("heading", { name: title });
  const section = heading.closest("section");
  if (!section) {
    throw new Error(`No card found for "${title}"`);
  }
  return section as HTMLElement;
}

function uploadReport() {
  const file = new File(["%PDF-1.4 fake report"], "lab-report.pdf", {
    type: "application/pdf",
  });
  const input = document.querySelector('input[type="file"]') as HTMLInputElement;
  fireEvent.change(input, { target: { files: [file] } });
  return file;
}

describe("the report journey", () => {
  beforeEach(() => {
    mockedApi.getInputSchema.mockResolvedValue(inputSchema);
    mockedApi.previewHealthReport.mockResolvedValue(extractionPreview);
    mockedApi.confirmExtraction.mockResolvedValue(confirmResponse);
    mockedApi.getAssessmentHistory.mockResolvedValue(historyResponse);
    mockedApi.getAssessmentHistorySummary.mockResolvedValue(historySummary);
    mockedApi.reanalyzeRecord.mockResolvedValue({
      status: "success",
      message: "Record re-analyzed",
      record_id: 42,
    });
  });

  it("reads the report, shows the extracted values and confirms them", async () => {
    renderWithProviders(<AnalyzerPage />);

    // The manual form is not on screen, but the schema is still fetched.
    await waitFor(() => expect(mockedApi.getInputSchema).toHaveBeenCalled());

    uploadReport();
    fireEvent.click(screen.getByRole("button", { name: /read the report/i }));

    // The preview endpoint was called with the file, not the one-shot upload.
    await waitFor(() => expect(mockedApi.previewHealthReport).toHaveBeenCalled());
    expect(mockedApi.previewHealthReport).toHaveBeenCalledWith(expect.any(File), false);

    // The extracted values are shown, with their clinical status.
    const review = await screen.findByRole("heading", { name: /confirm the extracted values/i });
    expect(review).toBeInTheDocument();
    expect(screen.getByTestId("field-hba1c")).toHaveValue(6.8);
    expect(screen.getByTestId("field-glucose")).toHaveValue(148);
    expect(within(cardByTitle("Confirm the extracted values")).getAllByText("High")).toHaveLength(2);

    // The conditions that can be assessed are named.
    expect(within(cardByTitle("Confirm the extracted values")).getByText("diabetes")).toBeInTheDocument();

    // The values still missing for another condition are named, and offered.
    const reviewCard = cardByTitle("Confirm the extracted values");
    expect(reviewCard).toHaveTextContent("systolic_bp, total_cholesterol");
    expect(screen.getByTestId("field-systolic_bp")).toBeInTheDocument();

    // A wrong value is corrected before the assessment runs.
    fireEvent.change(screen.getByTestId("field-glucose"), { target: { value: "120" } });

    fireEvent.click(screen.getByRole("button", { name: /confirm and assess/i }));

    await waitFor(() => expect(mockedApi.confirmExtraction).toHaveBeenCalled());
    expect(mockedApi.confirmExtraction).toHaveBeenCalledWith(
      expect.objectContaining({
        extraction_id: 42,
        hba1c: 6.8,
        glucose: 120,
        bmi: 29,
        age: 45,
      }),
    );

    // The assessment is shown, and the condition that was not applicable is
    // reported as such rather than as a failure.
    const diabetes = await screen.findByRole("heading", { name: /diabetes risk assessment/i });
    expect(diabetes).toBeInTheDocument();
    const diabetesCard = cardByTitle("Diabetes risk assessment");
    expect(within(diabetesCard).getByText("Model score: 0.42")).toBeInTheDocument();
    expect(within(diabetesCard).getAllByText("Moderate").length).toBeGreaterThan(0);

    const cardiovascular = cardByTitle("Cardiovascular risk assessment");
    expect(within(cardiovascular).getAllByText("Not applicable").length).toBeGreaterThan(0);
    expect(
      within(cardiovascular).getByText(/no required values were submitted/i),
    ).toBeInTheDocument();

    // The agents that ran are reported with their own status.
    const agentPanel = cardByTitle("Agent status");
    expect(within(agentPanel).getByText("Report Analyzer")).toBeInTheDocument();
    expect(within(agentPanel).getByText("Risk Predictor")).toBeInTheDocument();
    expect(within(agentPanel).getByText("Skipped")).toBeInTheDocument();
  });

  it("starts a second report from its own values", async () => {
    renderWithProviders(<AnalyzerPage />);
    await waitFor(() => expect(mockedApi.getInputSchema).toHaveBeenCalled());

    uploadReport();
    fireEvent.click(screen.getByRole("button", { name: /read the report/i }));
    await screen.findByRole("heading", { name: /confirm the extracted values/i });
    expect(screen.getByTestId("field-hba1c")).toHaveValue(6.8);

    // A second report, with a different value for the same parameter.
    mockedApi.previewHealthReport.mockResolvedValue({
      ...extractionPreview,
      extraction_id: 43,
      extracted_parameters: {
        ...extractionPreview.extracted_parameters,
        hba1c: { value: 5.2, status: "Normal", unit: "%", note: null, source: "extracted" },
      },
    });

    uploadReport();
    fireEvent.click(screen.getByRole("button", { name: /read the report/i }));
    await waitFor(() => expect(screen.getByTestId("field-hba1c")).toHaveValue(5.2));
  });

  it("sends the manual entry the schema describes, and nothing it does not", async () => {
    mockedApi.submitHealthData.mockResolvedValue(confirmResponse);
    renderWithProviders(<AnalyzerPage />, ["/analyzer?tab=manual"]);

    await screen.findByRole("heading", { name: /manual health entry/i });
    // The fields appear once the schema has been read.
    await waitFor(() => expect(screen.getByTestId("field-hba1c")).toBeInTheDocument());

    // The fields come from the schema. Only the selected condition's fields
    // are shown, so the cardiovascular ones appear when it is selected.
    expect(screen.getByTestId("field-hba1c")).toBeInTheDocument();
    expect(screen.queryByTestId("field-systolic_bp")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /cardiovascular/i }));
    expect(await screen.findByTestId("field-systolic_bp")).toBeInTheDocument();
    expect(screen.getByTestId("field-total_cholesterol")).toBeInTheDocument();
    // A category code is a choice, not a free-text measurement.
    expect(screen.getByTestId("field-chest_pain_type")).toHaveValue("");

    // The smoking status is a three-state choice, not a checkbox: leaving it
    // alone must not send "does not smoke".
    expect(screen.getByTestId("field-smoker")).toHaveValue("");

    fireEvent.change(screen.getByTestId("field-hba1c"), { target: { value: "6.8" } });
    fireEvent.change(screen.getByTestId("field-glucose"), { target: { value: "148" } });
    fireEvent.change(screen.getByTestId("field-bmi"), { target: { value: "29" } });
    fireEvent.change(screen.getByTestId("field-age"), { target: { value: "45" } });

    fireEvent.click(screen.getByRole("button", { name: /submit for analysis/i }));

    await waitFor(() => expect(mockedApi.submitHealthData).toHaveBeenCalled());
    const payload = mockedApi.submitHealthData.mock.calls[0][0];
    expect(payload).toMatchObject({ hba1c: 6.8, glucose: 148, bmi: 29, age: 45 });
    expect(payload).not.toHaveProperty("smoker");
  });
});

describe("the history", () => {
  beforeEach(() => {
    mockedApi.getInputSchema.mockResolvedValue(inputSchema);
    mockedApi.getAssessmentHistory.mockResolvedValue(historyResponse);
    mockedApi.getAssessmentHistorySummary.mockResolvedValue(historySummary);
    mockedApi.reanalyzeRecord.mockResolvedValue({
      status: "success",
      message: "Record re-analyzed",
      record_id: 42,
    });
  });

  it("lists the assessments and charts their series", async () => {
    renderWithProviders(<HistoryPage />, ["/history"]);

    await screen.findByRole("heading", { name: /^history$/i });
    // The list fills in once the history has been read.
    await screen.findByText(/Report #42/);

    const list = cardByTitle("History");
    expect(within(list).getByText(/Report #42/)).toBeInTheDocument();
    expect(within(list).getByText("Moderate")).toBeInTheDocument();
    // The condition that was not applicable is marked, not hidden.
    expect(within(list).getByText("Not applicable")).toBeInTheDocument();

    // The summary drives the charts.
    expect(await screen.findByTestId("trend-parameter-select")).toBeInTheDocument();
    const parameterSelect = screen.getByTestId("trend-parameter-select") as HTMLSelectElement;
    const options = Array.from(parameterSelect.options).map((option) => option.value);
    expect(options).toEqual(expect.arrayContaining(["hba1c", "glucose"]));
  });

  it("re-analyses a record and moves to its result", async () => {
    renderWithProviders(<HistoryPage />, ["/history"]);

    await screen.findByRole("heading", { name: /^history$/i });
    const reanalyzeButton = await screen.findByRole("button", { name: /re-analyze/i });
    fireEvent.click(reanalyzeButton);

    await waitFor(() => expect(mockedApi.reanalyzeRecord).toHaveBeenCalledWith(42));
    await screen.findByText("Results page");
  });
});
