import apiClient from "./client";

export interface User {
  id: number;
  username: string;
  email: string;
  roles: string[];
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export type AccountType = "USER" | "GOVT_OFFICER";

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  accountType?: AccountType;              // NEW
  identificationNumber?: string;          // NEW (only for GOVT_OFFICER)
}


// Backend now returns a message object/string for register, not a JWT
export type RegisterResponse =
  | string
  | {
      message?: string;
    };

export type VerifyEmailResponse = string;

export type ResendVerificationResponse = string;

export const authApi = {
  login: async (data: LoginRequest): Promise<AuthResponse> => {
    const response = await apiClient.post<AuthResponse>("/auth/login", data);
    return response.data;
  },

  // IMPORTANT: register does NOT return token anymore
  register: async (data: RegisterRequest): Promise<RegisterResponse> => {
    const response = await apiClient.post<RegisterResponse>("/auth/register", data);
    return response.data;
  },

  verifyEmail: async (token: string): Promise<VerifyEmailResponse> => {
    const response = await apiClient.get<VerifyEmailResponse>("/auth/verify-email", {
      params: { token },
    });
    return response.data;
  },

  resendVerification: async (email: string): Promise<ResendVerificationResponse> => {
    const response = await apiClient.post<ResendVerificationResponse>("/auth/resend-verification", {
      email,
    });
    return response.data;
  },
};

export default authApi;
