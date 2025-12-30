import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/context/AuthContext";
import { ThemeProvider } from "@/context/ThemeContext";
import Layout from "@/components/layout/Layout";
import ProtectedRoute from "@/routes/ProtectedRoute";
import AdminRoute from "@/routes/AdminRoute";

// Pages
import LandingPage from "@/pages/LandingPage";
import LoginPage from "@/pages/LoginPage";
import RegisterPage from "@/pages/RegisterPage";
import DashboardPage from "@/pages/DashboardPage";
import ProfilePage from "@/pages/ProfilePage";
import AdminPage from "@/pages/AdminPage";
import UnauthorizedPage from "@/pages/UnauthorizedPage";
import NotFoundPage from "@/pages/NotFoundPage";
import GovtOfficerDashboardPage from "@/pages/GovtOfficerDashboardPage";
import CreateRequestPage from "@/pages/requests/CreateRequestPage";
import RequestChatPage from "@/pages/requests/RequestChatPage";
import RequestsPage from "@/pages/requests/RequestsPage";
import DiseaseDetectionPage from "@/pages/ml/DiseaseDetectionPage";


// Forum Pages
import ForumTopicsPage from "@/pages/forum/ForumTopicsPage";
import ForumTopicPostsPage from "@/pages/forum/ForumTopicPostsPage";
import ForumPostPage from "@/pages/forum/ForumPostPage";
import CreateForumPostPage from "@/pages/forum/CreateForumPostPage";

// NEW Pages
import VerifyEmailPage from "@/pages/VerifyEmailPage";
import ResendVerificationPage from "@/pages/ResendVerificationPage";
import CheckEmailPage from "@/pages/CheckEmailPage";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider defaultTheme="system" storageKey="agriverse-ui-theme">
      <AuthProvider>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Layout>
              <Routes>
                {/* Public */}
                <Route path="/" element={<LandingPage />} />
                <Route path="/login" element={<LoginPage />} />
                <Route path="/register" element={<RegisterPage />} />

                {/* NEW: email verification flow */}
                <Route path="/verify-email" element={<VerifyEmailPage />} />
                <Route path="/resend-verification" element={<ResendVerificationPage />} />
                <Route path="/check-email" element={<CheckEmailPage />} />

                <Route path="/unauthorized" element={<UnauthorizedPage />} />

                {/* Protected */}
                <Route
                  path="/dashboard"
                  element={
                    <ProtectedRoute>
                      <DashboardPage />
                    </ProtectedRoute>
                  }
                />

                <Route
                  path="/profile"
                  element={
                    <ProtectedRoute>
                      <ProfilePage />
                    </ProtectedRoute>
                  }
                />

                {/* Forum (Protected) */}
                <Route
                  path="/forum"
                  element={
                    <ProtectedRoute>
                      <ForumTopicsPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/forum/topics/:topicId"
                  element={
                    <ProtectedRoute>
                      <ForumTopicPostsPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/forum/topics/:topicId/new"
                  element={
                    <ProtectedRoute>
                      <CreateForumPostPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/forum/posts/:postId"
                  element={
                    <ProtectedRoute>
                      <ForumPostPage />
                    </ProtectedRoute>
                  }
                />

                {/* Admin */}
                <Route
                  path="/admin"
                  element={
                    <AdminRoute>
                      <AdminPage />
                    </AdminRoute>
                  }
                />
                <Route
                path="/gov/dashboard"
                element={
                  <ProtectedRoute requireGovtOfficer>
                    <GovtOfficerDashboardPage />
                  </ProtectedRoute>
                }
              />

              <Route
                path="/requests/new"
                element={
                  <ProtectedRoute>
                    <CreateRequestPage />
                  </ProtectedRoute>
                }
              />

              <Route
                path="/requests/:id"
                element={
                  <ProtectedRoute>
                    <RequestChatPage />
                  </ProtectedRoute>
                }
              />

              <Route
                path="/requests"
                element={
                  <ProtectedRoute>
                    <RequestsPage />
                  </ProtectedRoute>
                }
              />
              <Route path="/ml/disease" element={<DiseaseDetectionPage />} />


                {/* Catch-all */}
                <Route path="*" element={<NotFoundPage />} />
              </Routes>
            </Layout>
          </BrowserRouter>
        </TooltipProvider>
      </AuthProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
