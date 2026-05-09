const ACCESS_TOKEN_KEY = "healthcare_access_token";

export const tokenStorage = {
  get(): string | null {
    if (typeof window === "undefined") {
      return null;
    }
    return window.localStorage.getItem(ACCESS_TOKEN_KEY);
  },
  set(token: string) {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(ACCESS_TOKEN_KEY, token);
  },
  clear() {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.removeItem(ACCESS_TOKEN_KEY);
  },
};
