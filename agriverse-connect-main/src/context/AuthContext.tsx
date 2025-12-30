import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { User, AuthResponse, LoginRequest, RegisterRequest, authApi } from '@/api/auth';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isAdmin: boolean;
  isGovtOfficer: boolean;
  isLoading: boolean;
  login: (data: LoginRequest) => Promise<void>;
  register: (data: RegisterRequest) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const storedToken = localStorage.getItem('agriverse_token');
    const storedUser = localStorage.getItem('agriverse_user');

    if (storedToken && storedUser) {
      try {
        setToken(storedToken);
        setUser(JSON.parse(storedUser));
      } catch {
        localStorage.removeItem('agriverse_token');
        localStorage.removeItem('agriverse_user');
      }
    }
    setIsLoading(false);
  }, []);

  const handleAuthSuccess = useCallback((response: AuthResponse) => {
    setToken(response.token);
    setUser(response.user);
    localStorage.setItem('agriverse_token', response.token);
    localStorage.setItem('agriverse_user', JSON.stringify(response.user));
  }, []);

  const login = useCallback(async (data: LoginRequest) => {
    const response = await authApi.login(data);
    handleAuthSuccess(response);
  }, [handleAuthSuccess]);

  // ✅ FIXED: no auto-login after register
  const register = useCallback(async (data: RegisterRequest) => {
    await authApi.register(data);
  }, []);

  const logout = useCallback(() => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('agriverse_token');
    localStorage.removeItem('agriverse_user');
  }, []);

  const isAuthenticated = !!token && !!user;
  const isAdmin = user?.roles?.includes('ROLE_ADMIN') ?? false;
  const isGovtOfficer = user?.roles?.includes('ROLE_GOVT_OFFICER') ?? false;

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated,
        isAdmin,
        isGovtOfficer,
        isLoading,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
};
