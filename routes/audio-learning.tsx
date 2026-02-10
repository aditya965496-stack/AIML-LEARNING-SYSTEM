import { useState } from "react";
import { Volume2, Sparkles, Download, Play, Pause } from "lucide-react";
import { Header } from "~/components/header/header";
import { Button } from "~/components/ui/button/button";
import { useToast } from "~/hooks/use-toast";
import styles from "./audio-learning.module.css";

type VoiceType = "professional" | "conversational" | "energetic";

export default function AudioLearning() {
  const [content, setContent] = useState("");
  const [voice, setVoice] = useState<VoiceType>("conversational");
  const [isGenerating, setIsGenerating] = useState(false);
  const [hasAudio, setHasAudio] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [transcript, setTranscript] = useState("");
  const { toast } = useToast();

  const voiceOptions: { value: VoiceType; label: string; description: string }[] = [
    { value: "professional", label: "Professional", description: "Clear & formal" },
    { value: "conversational", label: "Conversational", description: "Friendly & engaging" },
    { value: "energetic", label: "Energetic", description: "Dynamic & enthusiastic" },
  ];

  const handleGenerate = async () => {
    if (!content.trim()) {
      toast({
        title: "Content Required",
        description: "Please enter the text you want to convert to audio.",
        variant: "destructive",
      });
      return;
    }

    setIsGenerating(true);

    // Simulate AI generation
    setTimeout(() => {
      setTranscript(content);
      setHasAudio(true);
      setIsGenerating(false);
      toast({
        title: "Audio Generated",
        description: "Your audio lesson is ready to play!",
      });
    }, 2000);
  };

  const togglePlayback = () => {
    setIsPlaying(!isPlaying);
    toast({
      title: isPlaying ? "Paused" : "Playing",
      description: isPlaying ? "Audio playback paused" : "Audio playback started",
    });
  };

  const handleDownload = () => {
    toast({
      title: "Downloaded",
      description: "Audio file saved to your device.",
    });
  };

  return (
    <div className={styles.container}>
      <Header />

      <main className={styles.content}>
        <h1 className={styles.pageTitle}>
          <Volume2 className={styles.pageIcon} />
          Audio Learning
        </h1>
        <p className={styles.pageDescription}>
          Convert ML explanations into natural-sounding audio lessons. Perfect for on-the-go learning, commutes, and
          auditory learners who prefer listening to reading.
        </p>

        <div className={styles.inputSection}>
          <div className={styles.formGroup}>
            <label htmlFor="content" className={styles.label}>
              Text Content to Convert
            </label>
            <textarea
              id="content"
              className={styles.textarea}
              placeholder="Paste or type the ML concept explanation you want to convert to audio..."
              value={content}
              onChange={(e) => setContent(e.target.value)}
              rows={6}
            />
          </div>

          <div className={styles.formGroup}>
            <label className={styles.label}>Voice Style</label>
            <div className={styles.voiceOptions}>
              {voiceOptions.map((option) => (
                <button
                  key={option.value}
                  className={styles.voiceButton}
                  data-selected={voice === option.value}
                  onClick={() => setVoice(option.value)}
                >
                  <div>
                    <strong>{option.label}</strong>
                    <div style={{ fontSize: "var(--font-size-0)", opacity: 0.8 }}>{option.description}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <button className={styles.generateButton} onClick={handleGenerate} disabled={isGenerating}>
            {isGenerating ? (
              <>
                <Sparkles className={styles.loadingIcon} />
                Generating Audio...
              </>
            ) : (
              <>
                <Volume2 />
                Generate Audio Lesson
              </>
            )}
          </button>
        </div>

        {hasAudio ? (
          <div className={styles.resultSection}>
            <div className={styles.resultHeader}>
              <h2 className={styles.resultTitle}>Audio Lesson</h2>
              <Button variant="outline" size="sm" onClick={handleDownload}>
                <Download style={{ width: 16, height: 16 }} />
                Download MP3
              </Button>
            </div>

            <div className={styles.audioPlayer}>
              <div className={styles.audioControls}>
                <button className={styles.playButton} onClick={togglePlayback}>
                  {isPlaying ? <Pause size={24} /> : <Play size={24} />}
                </button>

                <div className={styles.waveform}>
                  {Array.from({ length: 40 }).map((_, i) => (
                    <div
                      key={i}
                      className={styles.waveBar}
                      style={{
                        height: `${Math.random() * 40 + 10}px`,
                        opacity: isPlaying ? 0.8 : 0.3,
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>

            <div className={styles.transcript}>
              <strong style={{ display: "block", marginBottom: "var(--space-3)", color: "var(--color-neutral-12)" }}>
                Transcript:
              </strong>
              {transcript}
            </div>
          </div>
        ) : (
          <div className={styles.emptyState}>
            <Volume2 className={styles.emptyIcon} />
            <p className={styles.emptyText}>Enter your content and generate an audio lesson for on-the-go learning</p>
          </div>
        )}
      </main>
    </div>
  );
}
