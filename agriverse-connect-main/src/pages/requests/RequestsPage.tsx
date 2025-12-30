import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { requestsApi, UserRequest } from "@/api/requests";
import { useAuth } from "@/context/AuthContext";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

type SpringPage<T> = { content: T[] };

function StatusBadge({ status }: { status: UserRequest["status"] }) {
  const label =
    status === "OPEN" ? "Open" :
    status === "IN_PROGRESS" ? "In progress" :
    "Archived";

  const variant =
    status === "OPEN" ? "secondary" :
    status === "IN_PROGRESS" ? "default" :
    "outline";

  return <Badge variant={variant as any}>{label}</Badge>;
}

function fmt(iso?: string | null) {
  if (!iso) return "";
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}

export default function RequestsPage() {
  const { isGovtOfficer } = useAuth();
  const navigate = useNavigate();

  // user data
  const [userActive, setUserActive] = useState<UserRequest[]>([]);
  const [userArchived, setUserArchived] = useState<UserRequest[]>([]);

  // officer data
  const [queue, setQueue] = useState<UserRequest[]>([]);
  const [assigned, setAssigned] = useState<UserRequest[]>([]);
  const [officerArchived, setOfficerArchived] = useState<UserRequest[]>([]);

  const [loading, setLoading] = useState(true);

  const loadUser = async () => {
    const mine: SpringPage<UserRequest> = await requestsApi.myRequests(0, 50);
    const mineContent = mine.content ?? [];
    setUserActive(mineContent.filter(r => r.status !== "ARCHIVED"));

    // prefer backend archived endpoint if present
    try {
      const arch: SpringPage<UserRequest> = await (requestsApi as any).myArchived(0, 50);
      setUserArchived(arch.content ?? []);
    } catch {
      setUserArchived(mineContent.filter(r => r.status === "ARCHIVED"));
    }
  };

  const loadOfficer = async () => {
    const q: SpringPage<UserRequest> = await requestsApi.officerQueue(0, 50);
    setQueue(q.content ?? []);

    const a: SpringPage<UserRequest> = await (requestsApi as any).officerAssigned(0, 50);
    setAssigned(a.content ?? []);

    const ar: SpringPage<UserRequest> = await (requestsApi as any).officerArchived(0, 50);
    setOfficerArchived(ar.content ?? []);
  };

  const refresh = async () => {
    setLoading(true);
    try {
      if (isGovtOfficer) await loadOfficer();
      else await loadUser();
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, [isGovtOfficer]);

  const openChat = (id: number) => navigate(`/requests/${id}`);

  const takeAndOpen = async (id: number) => {
    await requestsApi.take(id);
    openChat(id);
  };

  return (
    <div className="container mx-auto px-4 py-6 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Requests</h1>
          <p className="text-sm text-muted-foreground">
            {isGovtOfficer
              ? "Take new requests, chat with users, and archive solved cases."
              : "Create requests, chat with the officer, and view archived solutions."}
          </p>
        </div>

        <div className="flex gap-2">
          <Button variant="outline" onClick={refresh} disabled={loading}>
            Refresh
          </Button>

          {!isGovtOfficer && (
            <Button onClick={() => navigate("/requests/new")}>
              Create request
            </Button>
          )}
        </div>
      </div>

      <Card className="border-border/50">
        <CardHeader>
          <CardTitle>{isGovtOfficer ? "Officer view" : "My requests"}</CardTitle>
        </CardHeader>

        <CardContent>
          {isGovtOfficer ? (
            <Tabs defaultValue="queue">
              <TabsList>
                <TabsTrigger value="queue">Queue</TabsTrigger>
                <TabsTrigger value="assigned">Assigned</TabsTrigger>
                <TabsTrigger value="archived">Archived</TabsTrigger>
              </TabsList>

              <TabsContent value="queue" className="mt-4">
                {loading ? (
                  <div className="text-sm text-muted-foreground">Loading...</div>
                ) : queue.length === 0 ? (
                  <div className="text-sm text-muted-foreground">No unassigned requests.</div>
                ) : (
                  <div className="space-y-3">
                    {queue.map(r => (
                      <div key={r.id} className="rounded-lg border border-border/50 p-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <div className="font-medium">#{r.id} · {r.category}</div>
                            <StatusBadge status={r.status} />
                          </div>
                          <div className="text-sm text-muted-foreground line-clamp-2">{r.description}</div>
                          <div className="text-xs text-muted-foreground">From: {r.createdByUsername} · {fmt(r.createdAt)}</div>
                        </div>

                        <div className="flex gap-2">
                          <Button onClick={() => takeAndOpen(r.id)}>Take</Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </TabsContent>

              <TabsContent value="assigned" className="mt-4">
                {loading ? (
                  <div className="text-sm text-muted-foreground">Loading...</div>
                ) : assigned.length === 0 ? (
                  <div className="text-sm text-muted-foreground">No assigned requests.</div>
                ) : (
                  <div className="space-y-3">
                    {assigned.map(r => (
                      <div key={r.id} className="rounded-lg border border-border/50 p-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <div className="font-medium">#{r.id} · {r.category}</div>
                            <StatusBadge status={r.status} />
                          </div>
                          <div className="text-sm text-muted-foreground line-clamp-2">{r.description}</div>
                          <div className="text-xs text-muted-foreground">From: {r.createdByUsername}</div>
                        </div>

                        <Button variant="outline" onClick={() => openChat(r.id)}>
                          Open chat
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </TabsContent>

              <TabsContent value="archived" className="mt-4">
                {loading ? (
                  <div className="text-sm text-muted-foreground">Loading...</div>
                ) : officerArchived.length === 0 ? (
                  <div className="text-sm text-muted-foreground">No archived requests.</div>
                ) : (
                  <div className="space-y-3">
                    {officerArchived.map(r => (
                      <div key={r.id} className="rounded-lg border border-border/50 p-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <div className="font-medium">#{r.id} · {r.category}</div>
                            <StatusBadge status={r.status} />
                          </div>
                          <div className="text-sm text-muted-foreground line-clamp-2">{r.description}</div>
                          <div className="text-xs text-muted-foreground">Archived: {fmt(r.archivedAt)}</div>
                        </div>

                        <Button variant="outline" onClick={() => openChat(r.id)}>
                          View
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </TabsContent>
            </Tabs>
          ) : (
            <Tabs defaultValue="active">
              <TabsList>
                <TabsTrigger value="active">Active</TabsTrigger>
                <TabsTrigger value="archived">Archived</TabsTrigger>
              </TabsList>

              <TabsContent value="active" className="mt-4">
                {loading ? (
                  <div className="text-sm text-muted-foreground">Loading...</div>
                ) : userActive.length === 0 ? (
                  <div className="text-sm text-muted-foreground">No active requests.</div>
                ) : (
                  <div className="space-y-3">
                    {userActive.map(r => (
                      <div key={r.id} className="rounded-lg border border-border/50 p-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <div className="font-medium">#{r.id} · {r.category}</div>
                            <StatusBadge status={r.status} />
                          </div>
                          <div className="text-sm text-muted-foreground line-clamp-2">{r.description}</div>
                          <div className="text-xs text-muted-foreground">
                            Created: {fmt(r.createdAt)}
                            {r.assignedOfficerUsername ? ` · Officer: ${r.assignedOfficerUsername}` : " · Waiting for officer"}
                          </div>
                        </div>

                        <Button variant="outline" onClick={() => openChat(r.id)}>
                          Open chat
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </TabsContent>

              <TabsContent value="archived" className="mt-4">
                {loading ? (
                  <div className="text-sm text-muted-foreground">Loading...</div>
                ) : userArchived.length === 0 ? (
                  <div className="text-sm text-muted-foreground">No archived requests.</div>
                ) : (
                  <div className="space-y-3">
                    {userArchived.map(r => (
                      <div key={r.id} className="rounded-lg border border-border/50 p-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <div className="font-medium">#{r.id} · {r.category}</div>
                            <StatusBadge status={r.status} />
                          </div>
                          <div className="text-sm text-muted-foreground line-clamp-2">{r.description}</div>
                          <div className="text-xs text-muted-foreground">
                            Archived: {fmt(r.archivedAt)}
                            {r.assignedOfficerUsername ? ` · Solved by: ${r.assignedOfficerUsername}` : ""}
                          </div>
                        </div>

                        <Button variant="outline" onClick={() => openChat(r.id)}>
                          View chat
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </TabsContent>
            </Tabs>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
