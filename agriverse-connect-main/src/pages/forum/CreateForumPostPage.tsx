import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { forumApi } from "@/api/forum";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

export default function CreateForumPostPage() {
  const { topicId } = useParams();
  const tid = Number(topicId);
  const navigate = useNavigate();

  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const onSubmit = async () => {
    try {
      setSaving(true);
      setErr(null);

      const t = title.trim();
      const c = content.trim();
      if (!t || !c) throw new Error("Title and content are required");

      const created = await forumApi.createPost({
        topicId: tid,
        title: t,
        content: c,
      });

      // go to created post page
      navigate(`/forum/posts/${created.id}`);
    } catch (e: any) {
      setErr(e?.message || "Failed to create post");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="p-6 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Create Post</h1>
          <p className="text-sm text-muted-foreground">Write your post and publish it.</p>
        </div>

        <Button asChild variant="outline">
          <Link to={`/forum/topics/${tid}`}>Back</Link>
        </Button>
      </div>

      {err && <div className="text-sm text-destructive">{err}</div>}

      <Card className="rounded-2xl">
        <CardHeader>
          <CardTitle>New Post</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <Input
            placeholder="Title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />

          <Textarea
            placeholder="Write your post..."
            value={content}
            onChange={(e) => setContent(e.target.value)}
            className="min-h-[160px]"
          />

          <div className="flex gap-2">
            <Button onClick={onSubmit} disabled={saving}>
              {saving ? "Posting..." : "Publish"}
            </Button>

            <Button variant="secondary" onClick={() => navigate(-1)}>
              Cancel
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
