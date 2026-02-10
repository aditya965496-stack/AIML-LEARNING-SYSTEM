import { Link } from "react-router";
import { BookOpen, Code, Volume2, Image, Sparkles, Users, GraduationCap } from "lucide-react";
import { Header } from "~/components/header/header";
import { Badge } from "~/components/ui/badge/badge";
import styles from "./home.module.css";

export default function Home() {
  const features = [
    {
      icon: BookOpen,
      title: "Text Explanations",
      description:
        "Generate comprehensive, structured explanations of ML concepts with customizable depth levels. Perfect for detailed understanding and reference.",
      badge: "Detailed",
      path: "/text-explanation",
    },
    {
      icon: Code,
      title: "Code Generation",
      description:
        "Create production-ready Python implementations with automatic dependency detection, detailed comments, and execution instructions for Google Colab.",
      badge: "Practical",
      path: "/code-generation",
    },
    {
      icon: Volume2,
      title: "Audio Learning",
      description:
        "Convert explanations into natural-sounding audio lessons for on-the-go learning. Ideal for commutes and auditory learners.",
      badge: "Flexible",
      path: "/audio-learning",
    },
    {
      icon: Image,
      title: "Visual Diagrams",
      description:
        "Generate educational diagrams, flowcharts, and illustrations to visualize complex ML architectures and algorithms.",
      badge: "Intuitive",
      path: "/image-visualization",
    },
  ];

  const scenarios = [
    {
      icon: GraduationCap,
      title: "Students",
      description:
        "Master complex ML concepts through multi-modal learning. Get detailed explanations, working code examples, and visual aids all in one place.",
    },
    {
      icon: Users,
      title: "Professionals",
      description:
        "Quickly explore new algorithms and techniques. Generate production-ready code and create presentation materials with AI assistance.",
    },
    {
      icon: Sparkles,
      title: "Educators",
      description:
        "Create comprehensive course materials including explanations, code examples, audio lessons, and professional diagrams for your students.",
    },
  ];

  return (
    <div className={styles.container}>
      <Header />

      <section className={styles.hero}>
        <h1 className={styles.heroTitle}>AI-Powered Learning Assistant for Machine Learning</h1>
        <p className={styles.heroSubtitle}>
          Transform how you learn ML concepts through interactive, multi-modal AI-generated content
        </p>
        <p className={styles.heroDescription}>
          Powered by Google's Gemini 2.0 Flash AI, GyanGuru provides instant, personalized learning experiences that
          adapt to your preferred learning style—whether text, code, audio, or visual.
        </p>
      </section>

      <section className={styles.features}>
        <div className={styles.featuresGrid}>
          {features.map((feature) => {
            const Icon = feature.icon;
            return (
              <Link key={feature.path} to={feature.path} className={styles.featureCard}>
                <Icon className={styles.featureIcon} />
                <h3 className={styles.featureTitle}>{feature.title}</h3>
                <p className={styles.featureDescription}>{feature.description}</p>
                <Badge className={styles.featureBadge} variant="secondary">
                  {feature.badge}
                </Badge>
              </Link>
            );
          })}
        </div>
      </section>

      <section className={styles.scenarios}>
        <h2 className={styles.scenariosTitle}>Who Benefits from GyanGuru?</h2>
        <div className={styles.scenariosGrid}>
          {scenarios.map((scenario) => {
            const Icon = scenario.icon;
            return (
              <div key={scenario.title} className={styles.scenarioCard}>
                <Icon className={styles.featureIcon} />
                <h3 className={styles.scenarioTitle}>{scenario.title}</h3>
                <p className={styles.scenarioText}>{scenario.description}</p>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
