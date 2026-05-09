type ImportMetaEnv = {
  VITE_API_BASE_URL?: string;
};

const viteEnv = (typeof import.meta !== "undefined"
  ? (import.meta as { env?: ImportMetaEnv }).env
  : undefined) as ImportMetaEnv | undefined;

export const API_BASE_URL = viteEnv?.VITE_API_BASE_URL || "http://localhost:8000";
