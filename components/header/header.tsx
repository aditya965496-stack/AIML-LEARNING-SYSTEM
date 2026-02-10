import { Link, useLocation } from "react-router";
import { Brain, BookOpen, Code, Volume2, Image } from "lucide-react";
import { ColorSchemeToggle } from "~/components/ui/color-scheme-toggle/color-scheme-toggle";
import styles from "./header.module.css";

export function Header() {
  const location = useLocation();

  const navItems = [
    { path: "/", label: "Home", icon: Brain },
    { path: "/text-explanation", label: "Text", icon: BookOpen },
    { path: "/code-generation", label: "Code", icon: Code },
    { path: "/audio-learning", label: "Audio", icon: Volume2 },
    { path: "/image-visualization", label: "Visual", icon: Image },
  ];

  return (
    <header className={styles.header}>
      <div className={styles.container}>
        <Link to="/" className={styles.brand}>
          <Brain className={styles.logo} />
          <span className={styles.brandText}>GyanGuru</span>
        </Link>

        <nav className={styles.nav}>
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={styles.navLink}
                data-active={location.pathname === item.path}
              >
                <Icon className={styles.navIcon} />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className={styles.actions}>
          <ColorSchemeToggle />
        </div>
      </div>
    </header>
  );
}
