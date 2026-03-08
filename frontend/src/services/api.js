import axios from "axios";

// Create an axios instance with the base URL of backend API
const API = axios.create({
  baseURL: "http://localhost:8000/api/v1",
});
API.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
export default API;
