import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { forumApi, Topic } from "@/api/forum";
import { useAuth } from "@/context/AuthContext";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

export default function ForumTopicsPage() {
  const { isAdmin } = useAuth();

  const [topics, setTopics] = useState<Topic[]>([]);
  const [query, setQuery] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // dialog state
  const [open, setOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [creating, setCreating] = useState(false);

  const load = async () => {
    const list = await forumApi.listTopics();
    setTopics(list);
  };

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        setErr(null);
        await load();
      } catch (e: any) {
        setErr(e?.message || "Failed to load topics");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return topics;
    return topics.filter(
      (t) =>
        t.name.toLowerCase().includes(q) ||
        (t.description || "").toLowerCase().includes(q)
    );
  }, [topics, query]);

  const onCreateTopic = async () => {
    try {
      setCreating(true);
      setErr(null);

      const name = newName.trim();
      const description = newDesc.trim();

      if (!name) throw new Error("Forum name is required");

      await forumApi.createTopic({
        name,
        description: description || undefined,
      });

      // reset + close
      setNewName("");
      setNewDesc("");
      setOpen(false);

      await load();
    } catch (e: any) {
      setErr(e?.message || "Failed to create forum");
    } finally {
      setCreating(false);
    }
  };

  const onDialogOpenChange = (v: boolean) => {
    setOpen(v);
    if (!v) {
      // reset inputs when closing
      setNewName("");
      setNewDesc("");
    }
  };

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div>
          <h1 className="text-2xl font-semibold">Forum</h1>
          <p className="text-sm text-muted-foreground">
            Choose a category (cows, rice, etc.)
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="w-full sm:w-80">
            <Input
              placeholder="Search forums..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>

          {isAdmin && (
            <Dialog open={open} onOpenChange={onDialogOpenChange}>
              <DialogTrigger asChild>
                <Button>Create Forum</Button>
              </DialogTrigger>

              <DialogContent className="sm:max-w-lg">
                <DialogHeader>
                  <DialogTitle>Create a new forum</DialogTitle>
                  <DialogDescription>
                    Only admins can create new forum categories.
                  </DialogDescription>
                </DialogHeader>

                <div className="space-y-3">
                  <div className="space-y-1">
                    <label className="text-sm font-medium">Forum name</label>
                    <Input
                      placeholder="e.g. wheat"
                      value={newName}
                      onChange={(e) => setNewName(e.target.value)}
                    />
                  </div>

                  <div className="space-y-1">
                    <label className="text-sm font-medium">Description (optional)</label>
                    <Textarea
                      placeholder="Short description…"
                      value={newDesc}
                      onChange={(e) => setNewDesc(e.target.value)}
                      className="min-h-[110px]"
                    />
                  </div>
                </div>

                <DialogFooter className="gap-2 sm:gap-0">
                  <Button
                    variant="outline"
                    onClick={() => setOpen(false)}
                    disabled={creating}
                  >
                    Cancel
                  </Button>
                  <Button onClick={onCreateTopic} disabled={creating}>
                    {creating ? "Creating..." : "Create"}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          )}
        </div>
      </div>

      {loading && <div className="text-sm text-muted-foreground">Loading topics...</div>}
      {err && <div className="text-sm text-destructive">{err}</div>}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {filtered.map((t) => (
          <Card key={t.id} className="rounded-2xl">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="capitalize">{t.name}</span>
                <span className="text-xs text-muted-foreground">
                  {t.postCount} posts
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-sm text-muted-foreground line-clamp-2">
                {t.description || "—"}
              </p>
              <Button asChild className="w-full">
                <Link to={`/forum/topics/${t.id}`}>Open</Link>
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
