import { useEffect, useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { forumApi, Post } from "@/api/forum";
import { Link } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { User, Shield, MessageSquare, Briefcase } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

const DashboardPage = () => {
  const { user, isAdmin, isGovtOfficer } = useAuth();
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const page = await forumApi.getMyRecentPosts(0, 5);
        setPosts(page.content);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const roleLabel = isAdmin ? "Admin" : isGovtOfficer ? "GOVT Officer" : "User";
  const RoleIcon = isAdmin ? Shield : isGovtOfficer ? Briefcase : User;

  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold mb-1">Welcome back, {user?.username}</h1>
        <p className="text-muted-foreground">Here’s what you’ve been working on recently.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {/* Account */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="h-5 w-5 text-primary" />
              Account
            </CardTitle>
          </CardHeader>

          <CardContent className="space-y-2">
            <div className="font-medium">{user?.username}</div>
            <div className="text-sm text-muted-foreground">{user?.email}</div>

            <span
              className={`inline-flex items-center gap-1 text-xs px-2 py-1 rounded ${
                isAdmin || isGovtOfficer ? "bg-primary/10 text-primary" : "bg-muted"
              }`}
            >
              <RoleIcon className="h-3 w-3" /> {roleLabel}
            </span>
          </CardContent>
        </Card>

        {/* Recent Posts */}
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MessageSquare className="h-5 w-5 text-primary" />
              Your Recent Posts
            </CardTitle>
            <CardDescription>Last 5 posts you created</CardDescription>
          </CardHeader>

          <CardContent className="space-y-3">
            {loading && (
              <>
                <Skeleton className="h-6 w-full" />
                <Skeleton className="h-6 w-3/4" />
              </>
            )}

            {!loading && posts.length === 0 && (
              <div className="text-sm text-muted-foreground">You haven’t created any forum posts yet.</div>
            )}

            {posts.map((p) => (
              <div
                key={p.id}
                className="flex items-center justify-between border-b border-border/50 pb-2"
              >
                <div>
                  <div className="font-medium">{p.title}</div>
                  <div className="text-xs text-muted-foreground capitalize">
                    {p.topicName} • {new Date(p.createdAt).toLocaleDateString()}
                  </div>
                </div>

                <Button asChild size="sm" variant="outline">
                  <Link to={`/forum/posts/${p.id}`}>Open</Link>
                </Button>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default DashboardPage;
