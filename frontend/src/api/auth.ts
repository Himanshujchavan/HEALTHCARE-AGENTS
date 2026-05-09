import { http } from "./http";
import { tokenStorage } from "./tokenStorage";
import { TokenResponse, UserCreate, UserLogin, UserResponse } from "../models/auth";

export async function registerUser(payload: UserCreate): Promise<UserResponse> {
  const response = await http.post<UserResponse>("/api/v1/auth/register", payload);
  return response.data;
}

export async function loginUser(payload: UserLogin): Promise<TokenResponse> {
  const form = new URLSearchParams();
  form.set("username", payload.username);
  form.set("password", payload.password);

  const response = await http.post<TokenResponse>("/api/v1/auth/login", form, {
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
  });

  if (response.data?.access_token) {
    tokenStorage.set(response.data.access_token);
  }

  return response.data;
}

export function logoutUser() {
  tokenStorage.clear();
}
