import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { forumApi, Comment, Post } from "@/api/forum";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

export default function ForumPostPage() {
  const { postId } = useParams();
  const pid = Number(postId);

  const [post, setPost] = useState<Post | null>(null);
  const [comments, setComments] = useState<Comment[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);

  const load = async () => {
    const p = await forumApi.getPost(pid);
    const c = await forumApi.listComments(pid, 0, 200);
    setPost(p);
    setComments(c.content);
  };

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        setErr(null);
        await load();
      } catch (e: any) {
        setErr(e?.message || "Failed to load post");
      } finally {
        setLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pid]);

  const onComment = async () => {
    try {
      setSending(true);
      setErr(null);
      const c = text.trim();
      if (!c) throw new Error("Comment cannot be empty");
      await forumApi.createComment(pid, { content: c });
      setText("");
      await load();
    } catch (e: any) {
      setErr(e?.message || "Failed to comment");
    } finally {
      setSending(false);
    }
  };

  if (loading) return <div className="p-6 text-sm text-muted-foreground">Loading...</div>;

  return (
    <div className="p-6 space-y-4">
      {err && <div className="text-sm text-destructive">{err}</div>}

      {post && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="text-2xl font-semibold">{post.title}</h1>
              <p className="text-sm text-muted-foreground">
                In <span className="font-medium capitalize">{post.topicName}</span> • by{" "}
                <span className="font-medium">{post.authorUsername}</span> •{" "}
                {new Date(post.createdAt).toLocaleString()}
              </p>
            </div>
            <Button asChild variant="outline">
              <Link to={`/forum/topics/${post.topicId}`}>Back</Link>
            </Button>
          </div>

          <Card className="rounded-2xl">
            <CardHeader>
              <CardTitle>Post</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="whitespace-pre-wrap leading-relaxed">{post.content}</p>
            </CardContent>
          </Card>

          <Card className="rounded-2xl">
            <CardHeader>
              <CardTitle>Reply</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Write a comment..."
                className="min-h-[100px]"
              />
              <Button onClick={onComment} disabled={sending}>
                {sending ? "Sending..." : "Comment"}
              </Button>
            </CardContent>
          </Card>

          <div className="space-y-3">
            {comments.map((c) => (
              <Card key={c.id} className="rounded-2xl">
                <CardContent className="pt-6 space-y-2">
                  <div className="text-xs text-muted-foreground">
                    <span className="font-medium">{c.authorUsername}</span> •{" "}
                    {new Date(c.createdAt).toLocaleString()}
                  </div>
                  <p className="whitespace-pre-wrap text-sm">{c.content}</p>
                </CardContent>
              </Card>
            ))}
            {comments.length === 0 && (
              <div className="text-sm text-muted-foreground">No comments yet.</div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
