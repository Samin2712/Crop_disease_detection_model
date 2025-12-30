import { useEffect, useState } from "react";
import apiClient from "@/api/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/context/AuthContext";

type UserRequest = {
  id: number;
  createdByUsername: string;
  assignedOfficerUsername?: string | null;
  assignedOfficerIdentificationNumber?: string | null;
  category: string;
  description: string;
  imageUrl?: string | null;
  state?: string | null;
  district?: string | null;
  createdAt: string;
};

export default function GovtOfficerDashboardPage() {
  const { isGovtOfficer } = useAuth();
  const [items, setItems] = useState<UserRequest[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isGovtOfficer) return;

    (async () => {
      try {
        // adjust path if your backend path is different
        const res = await apiClient.get("/user-requests/officer/queue", {
          params: { page: 0, size: 20 },
        });
        // Spring Page -> content
        setItems(res.data.content ?? []);
      } finally {
        setLoading(false);
      }
    })();
  }, [isGovtOfficer]);

  if (!isGovtOfficer) return null;

  return (
    <div className="container mx-auto px-4 py-6 space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Govt Officer Dashboard</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div>Loading...</div>
          ) : items.length === 0 ? (
            <div className="text-muted-foreground">No assigned requests yet.</div>
          ) : (
            <div className="space-y-3">
              {items.map((r) => (
                <Card key={r.id} className="border-border/50">
                  <CardHeader>
                    <CardTitle className="text-base">
                      {r.category} • #{r.id}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm space-y-1">
                    <div><b>From:</b> {r.createdByUsername}</div>
                    <div><b>Location:</b> {r.district || "-"}, {r.state || "-"}</div>
                    <div className="text-muted-foreground">{r.description}</div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
