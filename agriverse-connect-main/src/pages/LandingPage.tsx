import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useAuth } from '@/context/AuthContext';
import { testApi } from '@/api/util';
import { useEffect, useState } from 'react';
import { Leaf, Sprout, Sun, Cloud, CheckCircle, XCircle, Loader2, ArrowRight, BarChart3, Shield, Zap } from 'lucide-react';

const LandingPage = () => {
  const { isAuthenticated } = useAuth();
  const [backendStatus, setBackendStatus] = useState<'loading' | 'connected' | 'disconnected'>('loading');
  const [statusMessage, setStatusMessage] = useState('');

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const message = await testApi.getPublicMessage();
        setBackendStatus('connected');
        setStatusMessage(message);
      } catch {
        setBackendStatus('disconnected');
        setStatusMessage('Unable to connect to backend');
      }
    };
    checkBackend();
  }, []);

  const features = [
    {
      icon: Sprout,
      title: 'Crop Management',
      description: 'Track and manage your crops with precision agriculture tools and real-time analytics.',
    },
    {
      icon: BarChart3,
      title: 'Smart Analytics',
      description: 'Get insights from your farm data with AI-powered predictions and recommendations.',
    },
    {
      icon: Shield,
      title: 'Secure Platform',
      description: 'Enterprise-grade security to protect your agricultural data and operations.',
    },
    {
      icon: Zap,
      title: 'Fast & Reliable',
      description: 'Lightning-fast performance with 99.9% uptime for critical farming operations.',
    },
  ];

  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <section className="relative py-20 lg:py-32 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-background to-accent/5" />
        <div className="container mx-auto px-4 relative">
          <div className="max-w-3xl mx-auto text-center animate-fade-in">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-6">
              <Leaf className="h-4 w-4" />
              Cultivating Tomorrow's Agriculture
            </div>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-foreground mb-6 tracking-tight">
              The Future of
              <span className="text-primary"> Agricultural</span>
              <br />
              Management
            </h1>
            <p className="text-lg md:text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              AgriVerse empowers farmers and agricultural businesses with cutting-edge tools to optimize operations, increase yields, and build sustainable futures.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              {isAuthenticated ? (
                <Button size="lg" asChild>
                  <Link to="/dashboard">
                    Go to Dashboard
                    <ArrowRight className="ml-2 h-5 w-5" />
                  </Link>
                </Button>
              ) : (
                <>
                  <Button size="lg" asChild>
                    <Link to="/register">
                      Get Started Free
                      <ArrowRight className="ml-2 h-5 w-5" />
                    </Link>
                  </Button>
                  <Button size="lg" variant="outline" asChild>
                    <Link to="/login">Sign In</Link>
                  </Button>
                </>
              )}
            </div>
          </div>

          {/* Decorative Elements */}
          <div className="absolute top-20 left-10 text-primary/20 animate-pulse hidden lg:block">
            <Sun className="h-16 w-16" />
          </div>
          <div className="absolute bottom-20 right-10 text-accent/30 animate-pulse hidden lg:block">
            <Cloud className="h-20 w-20" />
          </div>
        </div>
      </section>


      {/* Features Section */}
      <section className="py-20 bg-card/30">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-4">
              Everything You Need to Succeed
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Our platform provides comprehensive tools designed specifically for modern agriculture.
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <Card
                key={feature.title}
                className="border-border/50 hover:border-primary/30 transition-colors group"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <CardContent className="pt-6">
                  <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                    <feature.icon className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="font-semibold text-foreground mb-2">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <Card className="bg-primary text-primary-foreground border-0">
            <CardContent className="py-12 text-center">
              <h2 className="text-3xl font-bold mb-4">Ready to Transform Your Farm?</h2>
              <p className="text-primary-foreground/80 mb-8 max-w-xl mx-auto">
                Join thousands of farmers who are already using AgriVerse to optimize their operations.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                {isAuthenticated ? (
                  <Button size="lg" variant="secondary" asChild>
                    <Link to="/dashboard">Open Dashboard</Link>
                  </Button>
                ) : (
                  <>
                    <Button size="lg" variant="secondary" asChild>
                      <Link to="/register">Create Free Account</Link>
                    </Button>
                    <Button
                      size="lg"
                      variant="outline"
                      className="bg-transparent border-primary-foreground/30 text-primary-foreground hover:bg-primary-foreground/10"
                      asChild
                    >
                      <Link to="/login">Sign In</Link>
                    </Button>
                  </>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;
