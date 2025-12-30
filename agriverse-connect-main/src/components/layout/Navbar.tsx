import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { useTheme } from '@/context/ThemeContext';
import { Button } from '@/components/ui/button';
import { Sun, Moon, Leaf, Menu, X, LogOut, Shield, Briefcase } from 'lucide-react';
import { useState } from 'react';

const Navbar = () => {
  const { isAuthenticated, isAdmin, isGovtOfficer, user, logout } = useAuth(); // ✅ add isGovtOfficer
const { theme, setTheme } = useTheme();
  const navigate = useNavigate();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/');
    setMobileMenuOpen(false);
  };

  const NavLinks = () => (
    <>
      {isAuthenticated ? (
        <>
          <Link
            to="/dashboard"
            className="text-foreground/80 hover:text-primary transition-colors"
            onClick={() => setMobileMenuOpen(false)}
          >
            Dashboard
          </Link>

          <Link
            to="/profile"
            className="text-foreground/80 hover:text-primary transition-colors"
            onClick={() => setMobileMenuOpen(false)}
          >
            Profile
          </Link>

          <Link
            to="/forum"
            className="text-foreground/80 hover:text-primary transition-colors"
            onClick={() => setMobileMenuOpen(false)}
          >
            Forum
          </Link>




          {isAuthenticated && (
            <Link
              to="/requests"
              className="text-foreground/80 hover:text-primary transition-colors"
              onClick={() => setMobileMenuOpen(false)}
            >
              Requests
            </Link>
          )}

            {isAuthenticated && !isGovtOfficer && (
              <Link
                to="/ml/disease"
                className="text-foreground/80 hover:text-primary transition-colors"
                onClick={() => setMobileMenuOpen(false)}
              >
                Disease Detection
              </Link>
            )}


          {isAdmin && (
            <Link
              to="/admin"
              className="text-foreground/80 hover:text-primary transition-colors flex items-center gap-1"
              onClick={() => setMobileMenuOpen(false)}
            >
              <Shield className="h-4 w-4" />
              Admin
            </Link>
          )}
        </>
      ) : null}
    </>
  );

  const AuthButtons = () => (
    <>
      {isAuthenticated ? (
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground hidden sm:inline">
            {user?.username}
          </span>
          <Button variant="outline" size="sm" onClick={handleLogout}>
            <LogOut className="h-4 w-4 mr-1" />
            Logout
          </Button>
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" asChild>
            <Link to="/login">Login</Link>
          </Button>
          <Button size="sm" asChild>
            <Link to="/register">Get Started</Link>
          </Button>
        </div>
      )}
    </>
  );

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-border/50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 font-bold text-xl">
            <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
              <Leaf className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="text-foreground">AgriVerse</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-6">
            <NavLinks />
          </div>

          {/* Right side */}
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => {
                // if system, flip based on current system preference
                if (theme === "system") {
                  const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
                  setTheme(isDark ? "light" : "dark");
                  return;
                }
                setTheme(theme === "light" ? "dark" : "light");
              }}
              className="h-9 w-9"
            >
              {(() => {
                const effectiveTheme =
                  theme === "system"
                    ? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
                    : theme;
                return effectiveTheme === "light" ? (
                  <Moon className="h-5 w-5" />
                ) : (
                  <Sun className="h-5 w-5" />
                );
              })()}
            </Button>

            <div className="hidden md:flex">
              <AuthButtons />
            </div>

            {/* Mobile menu button */}
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden py-4 border-t border-border/50 animate-fade-in">
            <div className="flex flex-col gap-4">
              <NavLinks />
              <div className="pt-4 border-t border-border/50">
                <AuthButtons />
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
