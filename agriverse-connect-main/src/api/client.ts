import axios from "axios";
import type { InternalAxiosRequestConfig } from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem("agriverse_token");

    if (token) {
      // Axios v1+ safe way
      config.headers.set("Authorization", `Bearer ${token}`);
    }

    return config;
  },
  (error) => Promise.reject(error)
);

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error?.response?.status;

    if (status === 401) {
      localStorage.removeItem("agriverse_token");
      localStorage.removeItem("agriverse_user");

      const path = window.location.pathname;

      const isAuthPage =
        path.includes("/login") ||
        path.includes("/register") ||
        path.includes("/verify-email") ||
        path.includes("/resend-verification") ||
        path.includes("/check-email");

      if (!isAuthPage) {
        window.location.href = "/login";
      }
    }

    return Promise.reject(error);
  }
);

export default apiClient;
export { API_URL };
