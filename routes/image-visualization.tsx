import { useState } from "react";
import { Image as ImageIcon, Sparkles, Download } from "lucide-react";
import { Header } from "~/components/header/header";
import { Button } from "~/components/ui/button/button";
import { useToast } from "~/hooks/use-toast";
import styles from "./image-visualization.module.css";

type DiagramType = "architecture" | "flowchart" | "concept" | "algorithm";

export default function ImageVisualization() {
  const [description, setDescription] = useState("");
  const [diagramType, setDiagramType] = useState<DiagramType>("architecture");
  const [isGenerating, setIsGenerating] = useState(false);
  const [imageUrl, setImageUrl] = useState("");
  const [imageDescription, setImageDescription] = useState("");
  const { toast } = useToast();

  const diagramTypes: { value: DiagramType; label: string; description: string }[] = [
    { value: "architecture", label: "Architecture", description: "Network structures" },
    { value: "flowchart", label: "Flowchart", description: "Process flows" },
    { value: "concept", label: "Concept Map", description: "Idea relationships" },
    { value: "algorithm", label: "Algorithm", description: "Step-by-step logic" },
  ];

  const handleGenerate = async () => {
    if (!description.trim()) {
      toast({
        title: "Description Required",
        description: "Please describe the diagram you want to generate.",
        variant: "destructive",
      });
      return;
    }

    setIsGenerating(true);

    // Simulate AI generation with mock data
    setTimeout(() => {
      // Using a relevant ML/AI themed image from Unsplash
      const mockImages = [
        "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=800&q=80",
        "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=800&q=80",
        "https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=800&q=80",
      ];
      const randomImage = mockImages[Math.floor(Math.random() * mockImages.length)];

      setImageUrl(randomImage);
      setImageDescription(
        `Educational diagram illustrating ${description}. This visualization shows the key components, relationships, and flow of the concept in a clear, structured manner suitable for learning and presentation purposes.`,
      );
      setIsGenerating(false);
      toast({
        title: "Diagram Generated",
        description: "Your educational visualization is ready!",
      });
    }, 2500);
  };

  const handleDownload = () => {
    toast({
      title: "Downloaded",
      description: "Image saved to your device.",
    });
  };

  return (
    <div className={styles.container}>
      <Header />

      <main className={styles.content}>
        <h1 className={styles.pageTitle}>
          <ImageIcon className={styles.pageIcon} />
          Visual Diagrams
        </h1>
        <p className={styles.pageDescription}>
          Generate educational diagrams, flowcharts, and illustrations to visualize complex ML architectures and
          algorithms. Perfect for presentations, study materials, and visual learning.
        </p>

        <div className={styles.inputSection}>
          <div className={styles.formGroup}>
            <label htmlFor="description" className={styles.label}>
              Describe the Diagram
            </label>
            <input
              id="description"
              type="text"
              className={styles.input}
              placeholder="e.g., Neural network architecture with multiple layers, LSTM cell structure, Gradient descent optimization..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </div>

          <div className={styles.formGroup}>
            <label className={styles.label}>Diagram Type</label>
            <div className={styles.typeOptions}>
              {diagramTypes.map((type) => (
                <button
                  key={type.value}
                  className={styles.typeButton}
                  data-selected={diagramType === type.value}
                  onClick={() => setDiagramType(type.value)}
                >
                  <div>
                    <strong>{type.label}</strong>
                    <div style={{ fontSize: "var(--font-size-0)", opacity: 0.8 }}>{type.description}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <button className={styles.generateButton} onClick={handleGenerate} disabled={isGenerating}>
            {isGenerating ? (
              <>
                <Sparkles className={styles.loadingIcon} />
                Generating Diagram...
              </>
            ) : (
              <>
                <ImageIcon />
                Generate Visual Diagram
              </>
            )}
          </button>
        </div>

        {imageUrl ? (
          <div className={styles.resultSection}>
            <div className={styles.resultHeader}>
              <h2 className={styles.resultTitle}>Generated Diagram</h2>
              <Button variant="outline" size="sm" onClick={handleDownload}>
                <Download style={{ width: 16, height: 16 }} />
                Download
              </Button>
            </div>

            <div className={styles.imageContainer}>
              <img src={imageUrl} alt={description} className={styles.image} />
            </div>

            <div className={styles.imageDescription}>
              <h3 className={styles.descriptionTitle}>About this Diagram</h3>
              <p className={styles.descriptionText}>{imageDescription}</p>
            </div>
          </div>
        ) : (
          <div className={styles.emptyState}>
            <ImageIcon className={styles.emptyIcon} />
            <p className={styles.emptyText}>Describe your diagram and generate professional educational visuals</p>
          </div>
        )}
      </main>
    </div>
  );
}
