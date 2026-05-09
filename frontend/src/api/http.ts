import axios from "axios";
import { API_BASE_URL } from "../config/env";
import { tokenStorage } from "./tokenStorage";

export const http = axios.create({
  baseURL: API_BASE_URL,
});

http.interceptors.request.use((config) => {
  const token = tokenStorage.get();
  if (token) {
    config.headers = config.headers ?? {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
