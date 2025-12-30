import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { forumApi, Post, Page } from "@/api/forum";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function ForumTopicPostsPage() {
  const { topicId } = useParams();
  const tid = Number(topicId);

  const PAGE_SIZE = 10;

  const [topicName, setTopicName] = useState("");
  const [pageData, setPageData] = useState<Page<Post> | null>(null);
  const [posts, setPosts] = useState<Post[]>([]);
  const [pageIndex, setPageIndex] = useState(0); // 0-based
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  // server-side search
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");

  // debounce search text
  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search), 350);
    return () => clearTimeout(t);
  }, [search]);

  const load = async (newPageIndex: number, q: string) => {
    const page = await forumApi.listPostsByTopic(tid, newPageIndex, PAGE_SIZE, q);
    setPageData(page);
    setPosts(page.content);

    if (page.content[0]?.topicName) {
      setTopicName(page.content[0].topicName);
    } else {
      const topics = await forumApi.listTopics();
      setTopicName(topics.find((x) => x.id === tid)?.name || "");
    }
  };

  // when search changes, reset to page 0
  useEffect(() => {
    setPageIndex(0);
  }, [debouncedSearch]);

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        setErr(null);
        await load(pageIndex, debouncedSearch);
      } catch (e: any) {
        setErr(e?.message || "Failed to load posts");
      } finally {
        setLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tid, pageIndex, debouncedSearch]);

  const totalPages = pageData?.totalPages ?? 0;
  const canPrev = pageIndex > 0;
  const canNext = totalPages > 0 && pageIndex < totalPages - 1;

  // Jump UI
  const [jumpValue, setJumpValue] = useState("");
  useEffect(() => setJumpValue(""), [pageIndex, debouncedSearch]); // reset jump input

  const onJump = () => {
    const n = Number(jumpValue);
    if (!Number.isFinite(n)) return;
    if (n < 1 || n > totalPages) return;
    setPageIndex(n - 1);
  };

  const showingText = useMemo(() => {
    const total = pageData?.totalElements ?? 0;
    if (total === 0) return "No posts found.";
    const start = pageIndex * PAGE_SIZE + 1;
    const end = start + posts.length - 1;
    return `Showing ${start}-${end} of ${total} posts`;
  }, [pageData, pageIndex, posts.length]);

  return (
    <div className="p-6 space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold capitalize">{topicName || "Topic"}</h1>
          <p className="text-sm text-muted-foreground">
            {showingText}
          </p>
        </div>

        <div className="flex gap-2">
          <Button asChild>
            <Link to={`/forum/topics/${tid}/new`}>Create Post</Link>
          </Button>

          <Button asChild variant="outline">
            <Link to="/forum">Back</Link>
          </Button>
        </div>
      </div>

      {/* Search (SERVER-SIDE) */}
      <div className="w-full sm:w-96">
        <Input
          placeholder="Search posts (title + content, any order)"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {/* Pagination controls */}
      <Card className="rounded-2xl">
        <CardContent className="pt-6 flex flex-wrap items-center gap-3 justify-between">
          <div className="text-sm text-muted-foreground">
            Page{" "}
            <span className="font-medium text-foreground">
              {totalPages === 0 ? 0 : pageIndex + 1}
            </span>{" "}
            of{" "}
            <span className="font-medium text-foreground">{totalPages}</span>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              disabled={!canPrev || loading}
              onClick={() => setPageIndex((p) => Math.max(0, p - 1))}
            >
              Previous
            </Button>

            <Button
              variant="outline"
              disabled={!canNext || loading}
              onClick={() => setPageIndex((p) => p + 1)}
            >
              Next
            </Button>

            <div className="flex items-center gap-2">
              <Input
                className="w-28"
                placeholder="Page #"
                value={jumpValue}
                onChange={(e) => setJumpValue(e.target.value)}
              />
              <Button
                variant="secondary"
                disabled={loading || totalPages === 0}
                onClick={onJump}
              >
                Jump
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {loading && <div className="text-sm text-muted-foreground">Loading posts...</div>}
      {err && <div className="text-sm text-destructive">{err}</div>}

      {/* Posts list */}
      <div className="space-y-3">
        {posts.map((p) => (
          <Card key={p.id} className="rounded-2xl">
            <CardHeader>
              <CardTitle className="flex items-center justify-between gap-3">
                <Link className="hover:underline" to={`/forum/posts/${p.id}`}>
                  {p.title}
                </Link>
                <span className="text-xs text-muted-foreground">{p.commentCount} comments</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <p className="text-sm text-muted-foreground line-clamp-2">{p.content}</p>
              <div className="text-xs text-muted-foreground">
                By <span className="font-medium">{p.authorUsername}</span> •{" "}
                {new Date(p.createdAt).toLocaleString()}
              </div>
              <Button asChild variant="secondary">
                <Link to={`/forum/posts/${p.id}`}>Open</Link>
              </Button>
            </CardContent>
          </Card>
        ))}

        {!loading && posts.length === 0 && (
          <div className="text-sm text-muted-foreground">No posts match your search.</div>
        )}
      </div>
    </div>
  );
}
