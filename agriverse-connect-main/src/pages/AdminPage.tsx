import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { utilApi } from '@/api/util';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { useToast } from '@/hooks/use-toast';
import { Shield, User, Loader2, CheckCircle, AlertTriangle } from 'lucide-react';
import { AxiosError } from 'axios';

const grantAdminSchema = z.object({
  username: z.string()
    .trim()
    .min(1, 'Username is required')
    .max(30, 'Username must be less than 30 characters'),
});

type GrantAdminFormData = z.infer<typeof grantAdminSchema>;

const AdminPage = () => {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [lastResult, setLastResult] = useState<{ success: boolean; message: string } | null>(null);

  const form = useForm<GrantAdminFormData>({
    resolver: zodResolver(grantAdminSchema),
    defaultValues: {
      username: '',
    },
  });

  const onSubmit = async (data: GrantAdminFormData) => {
    setIsLoading(true);
    setLastResult(null);
    
    try {
      const response = await utilApi.grantAdminAccess(data.username);
      setLastResult({ success: true, message: response || `Admin access granted to ${data.username}` });
      toast({
        title: 'Success!',
        description: `Admin access has been granted to ${data.username}.`,
      });
      form.reset();
    } catch (error) {
      const axiosError = error as AxiosError<{ message?: string }>;
      let message = 'Failed to grant admin access. Please try again.';
      
      if (axiosError.response?.status === 404) {
        message = `User "${data.username}" not found.`;
      } else if (axiosError.response?.status === 403) {
        message = 'You do not have permission to perform this action.';
      } else if (axiosError.response?.data?.message) {
        message = axiosError.response.data.message;
      }
      
      setLastResult({ success: false, message });
      toast({
        title: 'Error',
        description: message,
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-2xl mx-auto">
        <div className="mb-8 animate-fade-in">
          <div className="flex items-center gap-3 mb-2">
            <div className="h-10 w-10 rounded-lg bg-primary flex items-center justify-center">
              <Shield className="h-6 w-6 text-primary-foreground" />
            </div>
            <h1 className="text-3xl font-bold text-foreground">Admin Panel</h1>
          </div>
          <p className="text-muted-foreground">
            Manage user permissions and access controls.
          </p>
        </div>

        <Card className="border-border/50 animate-fade-in" style={{ animationDelay: '100ms' }}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="h-5 w-5 text-primary" />
              Grant Admin Access
            </CardTitle>
            <CardDescription>
              Promote a user to administrator role. This will give them full access to the admin panel.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                <FormField
                  control={form.control}
                  name="username"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Username</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                          <Input
                            placeholder="Enter username to promote"
                            className="pl-10"
                            {...field}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <Button type="submit" disabled={isLoading}>
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Granting Access...
                    </>
                  ) : (
                    <>
                      <Shield className="mr-2 h-4 w-4" />
                      Grant Admin Access
                    </>
                  )}
                </Button>
              </form>
            </Form>

            {lastResult && (
              <div
                className={`mt-6 p-4 rounded-lg flex items-start gap-3 ${
                  lastResult.success
                    ? 'bg-success/10 border border-success/20'
                    : 'bg-destructive/10 border border-destructive/20'
                }`}
              >
                {lastResult.success ? (
                  <CheckCircle className="h-5 w-5 text-success flex-shrink-0 mt-0.5" />
                ) : (
                  <AlertTriangle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
                )}
                <div>
                  <p
                    className={`font-medium ${
                      lastResult.success ? 'text-success' : 'text-destructive'
                    }`}
                  >
                    {lastResult.success ? 'Success' : 'Error'}
                  </p>
                  <p className="text-sm text-foreground mt-1">{lastResult.message}</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Admin Info Card */}
        <Card className="border-border/50 mt-6 animate-fade-in" style={{ animationDelay: '200ms' }}>
          <CardHeader>
            <CardTitle className="text-lg">Admin Responsibilities</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                Manage user roles and permissions
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                Access administrative endpoints
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                Monitor system health and logs
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                Handle user escalations and support
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default AdminPage;
