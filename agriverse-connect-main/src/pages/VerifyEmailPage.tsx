import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import authApi from "@/api/auth";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { CheckCircle, Loader2, XCircle } from "lucide-react";
import { AxiosError } from "axios";

const VerifyEmailPage = () => {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const { toast } = useToast();

  const token = useMemo(() => params.get("token") || "", [params]);

  const [loading, setLoading] = useState(false);
  const [ok, setOk] = useState<boolean | null>(null);
  const [message, setMessage] = useState<string>("");

  useEffect(() => {
    if (!token) {
      setOk(false);
      setMessage("Missing verification token.");
      return;
    }

    (async () => {
      try {
        setLoading(true);
        const res = await authApi.verifyEmail(token);
        setOk(true);
        setMessage(typeof res === "string" ? res : "Email verified successfully.");

        toast({
          title: "Email verified!",
          description: "You can log in now.",
        });

        setTimeout(() => navigate("/login"), 800);
      } catch (err) {
        const axiosError = err as AxiosError<any>;
        const msg =
          axiosError.response?.data?.message ||
          (typeof axiosError.response?.data === "string" ? axiosError.response?.data : null) ||
          "Verification failed. Please try again.";

        setOk(false);
        setMessage(msg);
      } finally {
        setLoading(false);
      }
    })();
  }, [token, navigate, toast]);

  return (
    <div className="min-h-[calc(100vh-8rem)] flex items-center justify-center py-12 px-4">
      <Card className="w-full max-w-md border-border/50 animate-fade-in">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl font-bold">Verify Email</CardTitle>
          <CardDescription>Confirming your verification link…</CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          {loading && (
            <div className="flex items-center justify-center gap-2 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Verifying…</span>
            </div>
          )}

          {!loading && ok === true && (
            <div className="flex items-start gap-2">
              <CheckCircle className="h-5 w-5 mt-0.5 text-primary" />
              <div>
                <div className="font-medium">Success</div>
                <div className="text-sm text-muted-foreground">{message}</div>
              </div>
            </div>
          )}

          {!loading && ok === false && (
            <div className="flex items-start gap-2">
              <XCircle className="h-5 w-5 mt-0.5 text-destructive" />
              <div>
                <div className="font-medium">Failed</div>
                <div className="text-sm text-muted-foreground">{message}</div>
              </div>
            </div>
          )}

          <div className="flex flex-col sm:flex-row gap-3">
            <Button asChild className="w-full">
              <Link to="/login">Go to Login</Link>
            </Button>
            <Button asChild variant="outline" className="w-full">
              <Link to="/resend-verification">Resend Email</Link>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default VerifyEmailPage;
