import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { requestsApi, UserRequest, UserRequestMessage } from "@/api/requests";
import { useAuth } from "@/context/AuthContext";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { buildFileUrl } from "@/api/util";

// shadcn dialog + select
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type OfficerOption = {
  username: string;
  identificationNumber?: string | null;
};

function formatTime(iso?: string) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

export default function RequestChatPage() {
  const { id } = useParams();
  const requestId = useMemo(() => Number(id), [id]);

  const { user, isGovtOfficer } = useAuth();

  const [req, setReq] = useState<UserRequest | null>(null);
  const [msgs, setMsgs] = useState<UserRequestMessage[]>([]);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(true);


    // Description collapse/expand
  const [descExpanded, setDescExpanded] = useState(false);

  const descText = req?.description ?? "—";
  const descIsLong = descText.length > 320; // tweak threshold if you want
  const descPreview = descIsLong ? `${descText.slice(0, 320)}…` : descText;


  // Forward modal state
  const [forwardOpen, setForwardOpen] = useState(false);
  const [officers, setOfficers] = useState<OfficerOption[]>([]);
  const [selectedOfficer, setSelectedOfficer] = useState<string>("");
  const [forwarding, setForwarding] = useState(false);

  const bottomRef = useRef<HTMLDivElement | null>(null);

  const load = async () => {
    const [reqRes, msgRes] = await Promise.all([
      requestsApi.getById(requestId),
      requestsApi.messages(requestId, 0, 200),
    ]);
    setReq(reqRes);
    setMsgs(msgRes.content ?? []);
  };

  useEffect(() => {
    if (!requestId) return;

    (async () => {
      try {
        await load();
      } finally {
        setLoading(false);
      }
    })();

    const t = window.setInterval(() => load().catch(() => {}), 4000);
    return () => window.clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestId]);

  useEffect(() => {
    // scroll to bottom when messages change
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs.length]);

  const status = req?.status;
  const isArchived = status === "ARCHIVED";
  const officerMustTake = isGovtOfficer && status === "OPEN";

  const send = async () => {
    const m = text.trim();
    if (!m) return;
    setText("");
    await requestsApi.sendMessage(requestId, m);
    await load();
  };

  const archive = async () => {
    const updated = await requestsApi.archive(requestId);
    setReq(updated);
  };

  const take = async () => {
    const updated = await requestsApi.take(requestId);
    setReq(updated);
  };

  const openForward = async () => {
    setForwardOpen(true);
    setSelectedOfficer("");

    if (officers.length === 0) {
      const list: OfficerOption[] = await requestsApi.listGovtOfficers();
      const filtered = (list ?? []).filter((o) => o.username !== user?.username);
      setOfficers(filtered);
    }
  };

  const forward = async () => {
    if (!selectedOfficer) return;
    setForwarding(true);
    try {
      const updated = await requestsApi.forward(requestId, selectedOfficer);
      setReq(updated);
      setForwardOpen(false);
      await load();
    } finally {
      setForwarding(false);
    }
  };

  const imageSrc = buildFileUrl(req?.imageUrl);

  // Header labels
  const creatorName = req?.createdBy?.username ?? req?.createdByUsername ?? "Unknown";
  const creatorEmail = req?.createdBy?.email ?? "";
  const assignedName = req?.assignedOfficer?.username ?? req?.assignedOfficerUsername ?? "";
  const createdAt = formatTime((req as any)?.createdAt);
  const takenAt = formatTime((req as any)?.takenAt);
  const archivedAt = formatTime((req as any)?.archivedAt);

  return (
  <div className="container mx-auto px-4 py-6 h-[calc(100vh-80px)]">
    {/* Inbox layout */}
    <div className="h-full grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-4 overflow-hidden">
        {/* LEFT: Sidebar */}
        <Card className="border-border/50 h-full overflow-y-auto lg:sticky lg:top-6">
          <CardHeader className="space-y-1">
            <CardTitle className="flex items-center justify-between gap-2">
              <span>Request #{requestId}</span>
              <span
                className={`text-xs rounded-full px-2 py-1 border ${
                  status === "OPEN"
                    ? "bg-muted/30"
                    : status === "IN_PROGRESS"
                    ? "bg-primary/10"
                    : "bg-muted/50"
                }`}
              >
                {status ?? "—"}
              </span>
            </CardTitle>
            <div className="text-sm text-muted-foreground">
              {req?.category ?? "—"}
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            {/* Actions */}
            {isGovtOfficer && (
              <div className="flex flex-wrap gap-2">
                {status === "OPEN" && (
                  <Button variant="outline" onClick={take}>
                    Take
                  </Button>
                )}

                {status === "IN_PROGRESS" && (
                  <>
                    <Button variant="outline" onClick={openForward}>
                      Forward
                    </Button>
                    <Button variant="outline" onClick={archive}>
                      Archive
                    </Button>
                  </>
                )}

                {status === "ARCHIVED" && (
                  <Button variant="outline" disabled>
                    Archived
                  </Button>
                )}
              </div>
            )}

            {/* Creator */}
            <div className="rounded-xl border border-border/50 p-3">
              <div className="text-xs text-muted-foreground">Created by</div>
              <div className="font-medium">{creatorName}</div>
              {creatorEmail && (
                <div className="text-sm text-muted-foreground">{creatorEmail}</div>
              )}
            </div>

            {/* Assigned */}
            <div className="rounded-xl border border-border/50 p-3">
              <div className="text-xs text-muted-foreground">Assigned officer</div>
              <div className="font-medium">{assignedName || "Not assigned"}</div>
              {req?.assignedOfficer?.identificationNumber && (
                <div className="text-sm text-muted-foreground">
                  ID: {req.assignedOfficer.identificationNumber}
                </div>
              )}
            </div>

            {/* Meta */}
            <div className="rounded-xl border border-border/50 p-3 space-y-1">
              <div className="text-xs text-muted-foreground">Timeline</div>
              <div className="text-sm">
                <span className="text-muted-foreground">Created: </span>
                {createdAt || "—"}
              </div>
              <div className="text-sm">
                <span className="text-muted-foreground">Taken: </span>
                {takenAt || "—"}
              </div>
              <div className="text-sm">
                <span className="text-muted-foreground">Archived: </span>
                {archivedAt || "—"}
              </div>
            </div>

            {/* Description (collapsed by default; expandable) */}
            <div className="rounded-xl border border-border/50 p-3">
              <div className="flex items-center justify-between gap-2 mb-1">
                <div className="text-xs text-muted-foreground">Description</div>

                {descIsLong && (
                  <button
                    type="button"
                    onClick={() => setDescExpanded((v) => !v)}
                    className="text-xs font-medium text-primary hover:underline"
                  >
                    {descExpanded ? "Show less" : "Show more"}
                  </button>
                )}
              </div>

              <div
                role={descIsLong ? "button" : undefined}
                tabIndex={descIsLong ? 0 : -1}
                onClick={() => {
                  if (descIsLong) setDescExpanded((v) => !v);
                }}
                onKeyDown={(e) => {
                  if (!descIsLong) return;
                  if (e.key === "Enter" || e.key === " ") setDescExpanded((v) => !v);
                }}
                className={[
                  "text-sm whitespace-pre-wrap",
                  descIsLong ? "cursor-pointer select-text" : "",
                ].join(" ")}
              >
                {descExpanded ? descText : descPreview}
              </div>

              {(req?.state || req?.district) && (
                <div className="text-xs text-muted-foreground mt-2">
                  {req?.state ?? ""}
                  {req?.state && req?.district ? ", " : ""}
                  {req?.district ?? ""}
                </div>
              )}
            </div>


            {/* Image */}
            {imageSrc && (
              <div className="rounded-xl border border-border/50 p-3 space-y-2">
                <div className="text-xs text-muted-foreground">Attachment</div>
                <a href={imageSrc} target="_blank" rel="noreferrer">
                  <img
                    src={imageSrc}
                    alt="Request attachment"
                    className="w-full max-h-[260px] object-contain rounded-lg border border-border/50 bg-muted/20"
                  />
                </a>
                <div className="text-xs text-muted-foreground">
                  Click to open
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* RIGHT: Chat */}
        <Card className="border-border/50 flex flex-col h-full min-w-0 overflow-hidden">
          <CardHeader className="border-b border-border/50">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="font-medium truncate">{creatorName}</div>
                <div className="text-xs text-muted-foreground truncate">
                  {req?.category ?? "—"}
                </div>
              </div>

              <div className="text-xs text-muted-foreground">
                {isArchived
                  ? "Chat archived"
                  : officerMustTake
                  ? "Take the request to reply"
                  : "Live"}
              </div>
            </div>
          </CardHeader>

          {/* Messages area */}
            <CardContent className="flex-1 p-0 overflow-hidden">
              <div className="h-full overflow-y-auto px-4 py-4 space-y-3">

              {loading ? (
                <div className="text-sm text-muted-foreground">Loading...</div>
              ) : msgs.length === 0 ? (
                <div className="text-sm text-muted-foreground">
                  No messages yet.
                </div>
              ) : (
                msgs.map((m) => {
                  const mine = m.senderUsername === user?.username;

                  return (
                    <div
                      key={m.id}
                      className={`flex ${mine ? "justify-end" : "justify-start"}`}
                    >
                      <div className="max-w-[85%] sm:max-w-[70%] space-y-1">
                        <div
                          className={`rounded-2xl px-4 py-2 text-sm border border-border/50 ${
                            mine
                              ? "bg-primary text-primary-foreground"
                              : "bg-muted/30"
                          }`}
                        >
                          <div className="whitespace-pre-wrap">{m.message}</div>
                        </div>

                        <div
                          className={`text-[11px] text-muted-foreground ${
                            mine ? "text-right" : "text-left"
                          }`}
                        >
                          {m.senderUsername}
                          {m.createdAt ? ` • ${formatTime(m.createdAt)}` : ""}
                        </div>
                      </div>
                    </div>
                  );
                })
              )}

              <div ref={bottomRef} />
            </div>
          </CardContent>

          {/* Composer */}
          <div className="border-t border-border/50 p-3">
            <div className="flex gap-2 items-end">
              <Textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder={
                  isArchived
                    ? "Chat archived"
                    : officerMustTake
                    ? "Take the request to reply"
                    : "Write a message..."
                }
                disabled={isArchived || officerMustTake}
                className="min-h-[44px] max-h-[160px]"
              />
              <Button
                onClick={send}
                disabled={!text.trim() || isArchived || officerMustTake}
                className="h-[44px]"
              >
                Send
              </Button>
            </div>

            {!isArchived && !officerMustTake && (
              <div className="text-[11px] text-muted-foreground mt-2">
                Tip: Press Enter for a new line; click Send to submit.
              </div>
            )}
          </div>
        </Card>

        {/* Forward Modal */}
        <Dialog open={forwardOpen} onOpenChange={setForwardOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Forward request</DialogTitle>
            </DialogHeader>

            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">
                Select the govt officer you want to forward this request to.
              </div>

              <Select value={selectedOfficer} onValueChange={setSelectedOfficer}>
                <SelectTrigger>
                  <SelectValue placeholder="Select officer..." />
                </SelectTrigger>
                <SelectContent>
                  {officers.map((o) => (
                    <SelectItem key={o.username} value={o.username}>
                      {o.username}
                      {o.identificationNumber ? ` (ID: ${o.identificationNumber})` : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {officers.length === 0 && (
                <div className="text-sm text-muted-foreground">
                  No other officers found.
                </div>
              )}
            </div>

            <DialogFooter className="gap-2">
              <Button variant="outline" onClick={() => setForwardOpen(false)}>
                Cancel
              </Button>
              <Button onClick={forward} disabled={!selectedOfficer || forwarding}>
                {forwarding ? "Forwarding..." : "Forward"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
