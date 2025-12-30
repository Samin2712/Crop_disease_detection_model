import apiClient, { API_URL } from "./client";

export interface UserInfo {
  id: number;
  username: string;
  email: string;
  roles: string[];
  identificationNumber?: string | null;
}

// ✅ NEW: build absolute URL for file paths returned by backend (e.g. "/api/files/xyz.jpg")
export const buildFileUrl = (path?: string | null) => {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_URL}${path}`;
};

export const utilApi = {
  getUserInfo: async (username: string): Promise<UserInfo> => {
    const response = await apiClient.get<UserInfo>(`/api/util/user-info/${username}`);
    return response.data;
  },

  grantAdminAccess: async (username: string): Promise<string> => {
    const response = await apiClient.post<string>(`/api/admin/grant/${username}`);
    return response.data;
  },
};

export const testApi = {
  getPublicMessage: async (): Promise<string> => {
    const response = await apiClient.get<string>("/api/test/all");
    return response.data;
  },

  getProtectedMessage: async (): Promise<string> => {
    const response = await apiClient.get<string>("/api/test/user");
    return response.data;
  },
};

export default utilApi;
