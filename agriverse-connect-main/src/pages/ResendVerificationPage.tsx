import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import authApi from "@/api/auth";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Mail } from "lucide-react";
import { AxiosError } from "axios";

const ResendVerificationPage = () => {
  const { toast } = useToast();
  const [params] = useSearchParams();

  const initialEmail = useMemo(() => params.get("email") || "", [params]);

  const [email, setEmail] = useState(initialEmail);
  const [loading, setLoading] = useState(false);

  const onResend = async () => {
    try {
      setLoading(true);
      const res = await authApi.resendVerification(email.trim());

      toast({
        title: "Sent!",
        description: typeof res === "string" ? res : "Verification email resent.",
      });
    } catch (err) {
      const axiosError = err as AxiosError<any>;
      const msg =
        axiosError.response?.data?.message ||
        (typeof axiosError.response?.data === "string" ? axiosError.response?.data : null) ||
        "Failed to resend verification email.";

      toast({
        title: "Error",
        description: msg,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-[calc(100vh-8rem)] flex items-center justify-center py-12 px-4">
      <Card className="w-full max-w-md border-border/50 animate-fade-in">
        <CardHeader className="text-center">
          <div className="mx-auto h-12 w-12 rounded-lg bg-primary flex items-center justify-center mb-4">
            <Mail className="h-7 w-7 text-primary-foreground" />
          </div>
          <CardTitle className="text-2xl font-bold">Resend Verification</CardTitle>
          <CardDescription>Enter your email to get a new verification link.</CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          <Input
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            type="email"
            placeholder="you@example.com"
          />

          <Button className="w-full" disabled={loading || !email.trim()} onClick={onResend}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Resend Email"}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default ResendVerificationPage;
