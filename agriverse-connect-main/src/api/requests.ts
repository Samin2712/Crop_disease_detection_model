import apiClient from "@/api/client";

export type RequestStatus = "OPEN" | "IN_PROGRESS" | "ARCHIVED";

// ✅ NEW: matches backend UserInfo DTO (keep it lightweight + optional)
export type RequestUserInfo = {
  username: string;
  email: string;
  roles?: string[];
  identificationNumber?: string | null;
};

export type UserRequest = {
  id: number;

  // existing fields (keep)
  createdByUsername: string;
  assignedOfficerUsername?: string | null;

  category: string;
  description: string;
  imageUrl?: string | null;
  state?: string | null;
  district?: string | null;

  status: RequestStatus;
  createdAt: string;
  takenAt?: string | null;
  archivedAt?: string | null;

  // ✅ NEW: for chat UI to show details
  createdBy?: RequestUserInfo | null;
  assignedOfficer?: RequestUserInfo | null;

  // keep if backend sends it (your backend already had this in response)
  assignedOfficerIdentificationNumber?: string | null;
};

export type UserRequestMessage = {
  id: number;
  requestId: number;
  senderUsername: string;
  senderRole?: string | null;
  message: string;
  createdAt: string;
};

export const requestsApi = {
  // multipart create
  create: async (data: {
    category: string;
    description: string;
    state?: string;
    district?: string;
    image?: File | null;
  }): Promise<UserRequest> => {
    const fd = new FormData();
    fd.append("category", data.category);
    fd.append("description", data.description);
    if (data.state) fd.append("state", data.state);
    if (data.district) fd.append("district", data.district);
    if (data.image) fd.append("image", data.image);

    const res = await apiClient.post<UserRequest>("/api/requests", fd, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },

  myRequests: async (page = 0, size = 20) => {
    const res = await apiClient.get("/api/requests/mine", { params: { page, size } });
    return res.data; // Spring Page
  },

  officerQueue: async (page = 0, size = 20) => {
    const res = await apiClient.get("/api/requests/officer/queue", { params: { page, size } });
    return res.data;
  },

  take: async (id: number): Promise<UserRequest> => {
    const res = await apiClient.post<UserRequest>(`/api/requests/${id}/take`);
    return res.data;
  },

  archive: async (id: number): Promise<UserRequest> => {
    const res = await apiClient.post<UserRequest>(`/api/requests/${id}/archive`);
    return res.data;
  },

  myArchived: async (page = 0, size = 20) => {
    const res = await apiClient.get("/api/requests/mine/archived", { params: { page, size } });
    return res.data;
  },

  messages: async (id: number, page = 0, size = 50) => {
    const res = await apiClient.get(`/api/requests/${id}/messages`, {
      params: { page, size },
    });
    return res.data; // Spring Page
  },

  sendMessage: async (id: number, message: string): Promise<UserRequestMessage> => {
    const res = await apiClient.post<UserRequestMessage>(`/api/requests/${id}/messages`, { message });
    return res.data;
  },

  officerAssigned: async (page = 0, size = 20) => {
    const res = await apiClient.get("/api/requests/officer/assigned", { params: { page, size } });
    return res.data;
  },

  officerArchived: async (page = 0, size = 20) => {
    const res = await apiClient.get("/api/requests/officer/archived", { params: { page, size } });
    return res.data;
  },

  getById: async (id: number): Promise<UserRequest> => {
    const res = await apiClient.get<UserRequest>(`/api/requests/${id}`);
    return res.data;
  },

  forward: async (id: number, toOfficerUsername: string): Promise<UserRequest> => {
    const res = await apiClient.post<UserRequest>(`/api/requests/${id}/forward`, { toOfficerUsername });
    return res.data;
  },

  listGovtOfficers: async (): Promise<{ username: string; identificationNumber?: string | null }[]> => {
    const res = await apiClient.get("/api/users/govt-officers");
    return res.data;
  },
};
