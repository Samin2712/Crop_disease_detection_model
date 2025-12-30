import { useEffect, useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { utilApi, UserInfo } from "@/api/util";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { User, Mail, Shield, Calendar, BadgeCheck } from "lucide-react";
import { AxiosError } from "axios";

const ProfilePage = () => {
  const { user, isGovtOfficer } = useAuth();
  const [profileData, setProfileData] = useState<UserInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchProfile = async () => {
      if (!user?.username) return;

      try {
        const data = await utilApi.getUserInfo(user.username);
        setProfileData(data);
      } catch (err) {
        const axiosError = err as AxiosError<{ message?: string }>;
        setError(axiosError.response?.data?.message || "Failed to load profile");
      } finally {
        setIsLoading(false);
      }
    };

    fetchProfile();
  }, [user?.username]);

  const idNo =
    (profileData as any)?.identificationNumber ||
    (user as any)?.identificationNumber ||
    null;

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-2xl mx-auto">
        <div className="mb-8 animate-fade-in">
          <h1 className="text-3xl font-bold text-foreground mb-2">Profile</h1>
          <p className="text-muted-foreground">View your account information</p>
        </div>

        <Card className="border-border/50 animate-fade-in" style={{ animationDelay: "100ms" }}>
          <CardHeader>
            <div className="flex items-center gap-4">
              {isLoading ? (
                <Skeleton className="h-16 w-16 rounded-full" />
              ) : (
                <div className="h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center">
                  <User className="h-8 w-8 text-primary" />
                </div>
              )}
              <div>
                {isLoading ? (
                  <>
                    <Skeleton className="h-6 w-32 mb-2" />
                    <Skeleton className="h-4 w-48" />
                  </>
                ) : (
                  <>
                    <CardTitle className="text-xl">{profileData?.username || user?.username}</CardTitle>
                    <CardDescription>AgriVerse Member</CardDescription>
                  </>
                )}
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-6">
            {error ? (
              <div className="p-4 rounded-lg bg-destructive/10 text-destructive text-sm">{error}</div>
            ) : (
              <>
                {/* Email */}
                <div className="flex items-center gap-4 p-4 rounded-lg bg-muted/50">
                  <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                    <Mail className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Email Address</p>
                    {isLoading ? (
                      <Skeleton className="h-5 w-48 mt-1" />
                    ) : (
                      <p className="font-medium text-foreground">{profileData?.email || user?.email}</p>
                    )}
                  </div>
                </div>

                {/* Username */}
                <div className="flex items-center gap-4 p-4 rounded-lg bg-muted/50">
                  <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                    <User className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Username</p>
                    {isLoading ? (
                      <Skeleton className="h-5 w-32 mt-1" />
                    ) : (
                      <p className="font-medium text-foreground">{profileData?.username || user?.username}</p>
                    )}
                  </div>
                </div>

                {/* Roles */}
                <div className="flex items-center gap-4 p-4 rounded-lg bg-muted/50">
                  <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                    <Shield className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Roles</p>
                    {isLoading ? (
                      <Skeleton className="h-5 w-24 mt-1" />
                    ) : (
                      <div className="flex flex-wrap gap-2 mt-1">
                        {(profileData?.roles || user?.roles || []).map((role) => (
                          <span
                            key={role}
                            className="px-2 py-1 rounded-md bg-primary/10 text-primary text-sm font-medium"
                          >
                            {role.replace("ROLE_", "")}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* ✅ Identification Number (ONLY for Agri Officer) */}
                {isGovtOfficer && (
                  <div className="flex items-center gap-4 p-4 rounded-lg bg-muted/50">
                    <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                      <BadgeCheck className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Identification Number</p>
                      {isLoading ? (
                        <Skeleton className="h-5 w-40 mt-1" />
                      ) : (
                        <p className="font-medium text-foreground">{idNo || "-"}</p>
                      )}
                    </div>
                  </div>
                )}

                {/* User ID */}
                <div className="flex items-center gap-4 p-4 rounded-lg bg-muted/50">
                  <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                    <Calendar className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">User ID</p>
                    {isLoading ? (
                      <Skeleton className="h-5 w-16 mt-1" />
                    ) : (
                      <p className="font-medium text-foreground">#{profileData?.id || user?.id}</p>
                    )}
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ProfilePage;
