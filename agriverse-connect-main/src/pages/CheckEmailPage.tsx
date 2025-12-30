import { Link, useSearchParams } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Mail, ArrowRight } from "lucide-react";

const CheckEmailPage = () => {
  const [params] = useSearchParams();
  const email = params.get("email") || "";

  return (
    <div className="min-h-[calc(100vh-8rem)] flex items-center justify-center py-12 px-4">
      <Card className="w-full max-w-md border-border/50 animate-fade-in">
        <CardHeader className="text-center">
          <div className="mx-auto h-12 w-12 rounded-lg bg-primary flex items-center justify-center mb-4">
            <Mail className="h-7 w-7 text-primary-foreground" />
          </div>
          <CardTitle className="text-2xl font-bold">Check your email</CardTitle>
          <CardDescription>
            We sent a verification link{email ? ` to ${email}` : ""}. Verify your email before logging in.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-3">
          <Button asChild className="w-full">
            <Link to="/login">
              Go to Login <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>

          <Button asChild variant="outline" className="w-full">
            <Link to={`/resend-verification${email ? `?email=${encodeURIComponent(email)}` : ""}`}>
              Resend verification email
            </Link>
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default CheckEmailPage;
