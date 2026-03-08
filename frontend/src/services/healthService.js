import API from "./api";

// Submit health data
export const submitHealthData = async (data) => {
  const res = await API.post("/health/health-data", data);
  return res.data;
};

// Get health analysis for a specific record ID
export const getHealthAnalysis = async (id) => {
  const res = await API.get(`/health/health-data/${id}/analysis`);
  return res.data;
};

// Get latest health record for the logged-in user
export const getLatestRecord = async () => {
  const res = await API.get("/health/latest");
  return res.data;
};
