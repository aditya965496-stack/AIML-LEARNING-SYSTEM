import { useState } from "react";
import { BookOpen, Sparkles, Download } from "lucide-react";
import { Header } from "~/components/header/header";
import { Button } from "~/components/ui/button/button";
import { useToast } from "~/hooks/use-toast";
import styles from "./text-explanation.module.css";

type DepthLevel = "brief" | "moderate" | "comprehensive";

export default function TextExplanation() {
  const [topic, setTopic] = useState("");
  const [depth, setDepth] = useState<DepthLevel>("moderate");
  const [isGenerating, setIsGenerating] = useState(false);
  const [explanation, setExplanation] = useState("");
  const { toast } = useToast();

  const depthOptions: { value: DepthLevel; label: string; description: string }[] = [
    { value: "brief", label: "Brief", description: "Quick overview" },
    { value: "moderate", label: "Moderate", description: "Balanced detail" },
    { value: "comprehensive", label: "Comprehensive", description: "In-depth coverage" },
  ];

  const handleGenerate = async () => {
    if (!topic.trim()) {
      toast({
        title: "Topic Required",
        description: "Please enter a Machine Learning topic to explain.",
        variant: "destructive",
      });
      return;
    }

    setIsGenerating(true);

    // Simulate AI generation with mock data
    setTimeout(() => {
      const mockExplanation = generateMockExplanation(topic, depth);
      setExplanation(mockExplanation);
      setIsGenerating(false);
      toast({
        title: "Explanation Generated",
        description: "Your AI-powered explanation is ready!",
      });
    }, 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([explanation], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${topic.replace(/\s+/g, "-").toLowerCase()}-explanation.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Downloaded",
      description: "Explanation saved to your device.",
    });
  };

  return (
    <div className={styles.container}>
      <Header />

      <main className={styles.content}>
        <h1 className={styles.pageTitle}>
          <BookOpen className={styles.pageIcon} />
          Text Explanations
        </h1>
        <p className={styles.pageDescription}>
          Generate comprehensive, AI-powered explanations of Machine Learning concepts. Choose your depth level and get
          instant, structured content tailored to your learning needs.
        </p>

        <div className={styles.inputSection}>
          <div className={styles.formGroup}>
            <label htmlFor="topic" className={styles.label}>
              ML Topic or Concept
            </label>
            <input
              id="topic"
              type="text"
              className={styles.input}
              placeholder="e.g., Backpropagation in Neural Networks, LSTM Architecture, Gradient Descent..."
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
            />
          </div>

          <div className={styles.formGroup}>
            <label className={styles.label}>Explanation Depth</label>
            <div className={styles.depthOptions}>
              {depthOptions.map((option) => (
                <button
                  key={option.value}
                  className={styles.depthButton}
                  data-selected={depth === option.value}
                  onClick={() => setDepth(option.value)}
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
                Generating...
              </>
            ) : (
              <>
                <Sparkles />
                Generate Explanation
              </>
            )}
          </button>
        </div>

        {explanation ? (
          <div className={styles.resultSection}>
            <div className={styles.resultHeader}>
              <h2 className={styles.resultTitle}>{topic}</h2>
              <Button variant="outline" size="sm" onClick={handleDownload}>
                <Download style={{ width: 16, height: 16 }} />
                Download
              </Button>
            </div>
            <div className={styles.resultContent}>{explanation}</div>
          </div>
        ) : (
          <div className={styles.emptyState}>
            <BookOpen className={styles.emptyIcon} />
            <p className={styles.emptyText}>Enter a topic and generate your first AI-powered explanation</p>
          </div>
        )}
      </main>
    </div>
  );
}

function generateMockExplanation(topic: string, depth: DepthLevel): string {
  const depthContent = {
    brief: `# ${topic}

## Overview
${topic} is a fundamental concept in Machine Learning that plays a crucial role in model training and optimization. It involves the systematic adjustment of model parameters to minimize error and improve prediction accuracy.

## Key Points
• Core mechanism for learning in neural networks
• Enables gradient-based optimization
• Essential for training deep learning models
• Widely used across various ML architectures

## Applications
This technique is applied in supervised learning, deep learning, and optimization tasks across computer vision, natural language processing, and reinforcement learning domains.`,

    moderate: `# ${topic}

## Introduction
${topic} represents a cornerstone technique in Machine Learning, enabling models to learn from data through iterative parameter optimization. This approach has revolutionized how we build intelligent systems.

## Fundamental Concepts

### Mathematical Foundation
The technique relies on calculus and optimization theory, particularly the chain rule for computing gradients. These gradients indicate the direction and magnitude of parameter adjustments needed to minimize the loss function.

### Algorithm Steps
1. Forward pass: Compute predictions using current parameters
2. Loss calculation: Measure prediction error
3. Backward pass: Calculate gradients of loss with respect to parameters
4. Parameter update: Adjust weights using computed gradients
5. Iteration: Repeat until convergence

## Practical Implementation
In practice, ${topic} is implemented using frameworks like TensorFlow, PyTorch, or JAX. These libraries provide automatic differentiation, making gradient computation efficient and accurate.

### Key Considerations
• Learning rate selection affects convergence speed
• Batch size impacts gradient estimation quality
• Regularization prevents overfitting
• Initialization strategy influences training stability

## Real-World Applications
This technique powers modern AI systems including:
• Image recognition and computer vision
• Natural language processing and translation
• Recommendation systems
• Autonomous vehicles
• Medical diagnosis systems

## Advantages and Limitations
**Advantages:**
• Efficient parameter optimization
• Scalable to large datasets
• Proven effectiveness across domains

**Limitations:**
• Can get stuck in local minima
• Requires careful hyperparameter tuning
• Computationally intensive for large models`,

    comprehensive: `# ${topic}

## Executive Summary
${topic} is a fundamental algorithmic technique in Machine Learning that enables neural networks and other models to learn from data through gradient-based optimization. This comprehensive guide explores the mathematical foundations, implementation details, and practical applications of this essential ML concept.

## Historical Context
The development of ${topic} traces back to the 1960s and 1970s, with significant contributions from researchers in optimization theory and neural networks. The modern formulation became widely adopted in the 1980s and has since become the backbone of deep learning.

## Mathematical Foundations

### Calculus and Optimization Theory
The technique is grounded in multivariate calculus, specifically the chain rule for computing derivatives of composite functions. Given a loss function L(θ) where θ represents model parameters, the goal is to find:

θ* = argmin L(θ)

### Gradient Computation
Gradients are computed using the chain rule:
∂L/∂θᵢ = ∂L/∂yⱼ × ∂yⱼ/∂θᵢ

This allows efficient computation of how each parameter affects the overall loss.

## Detailed Algorithm

### Forward Propagation
1. Initialize input data x and parameters θ
2. Compute activations layer by layer: aˡ = f(Wˡaˡ⁻¹ + bˡ)
3. Generate final prediction ŷ
4. Calculate loss L(ŷ, y)

### Backward Propagation
1. Compute output layer gradient: δᴸ = ∇L ⊙ f'(zᴸ)
2. Propagate gradients backward: δˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ ⊙ f'(zˡ)
3. Calculate parameter gradients: ∂L/∂W = δaᵀ
4. Update parameters: θ ← θ - η∇L(θ)

## Implementation Strategies

### Optimization Algorithms
• **Stochastic Gradient Descent (SGD)**: Basic but effective
• **Momentum**: Accelerates convergence in relevant directions
• **Adam**: Adaptive learning rates per parameter
• **RMSprop**: Addresses diminishing learning rates
• **AdaGrad**: Adapts learning rate based on parameter history

### Regularization Techniques
• L1/L2 regularization: Prevents overfitting
• Dropout: Randomly deactivates neurons during training
• Batch normalization: Normalizes layer inputs
• Early stopping: Halts training when validation performance degrades

### Advanced Considerations
• **Gradient clipping**: Prevents exploding gradients
• **Learning rate scheduling**: Adjusts learning rate during training
• **Batch size effects**: Larger batches provide more stable gradients
• **Initialization strategies**: Xavier, He initialization for stable training

## Practical Applications

### Computer Vision
• Image classification (ResNet, VGG, EfficientNet)
• Object detection (YOLO, Faster R-CNN)
• Semantic segmentation (U-Net, DeepLab)
• Image generation (GANs, Diffusion models)

### Natural Language Processing
• Language models (GPT, BERT, T5)
• Machine translation (Transformer architectures)
• Sentiment analysis and text classification
• Question answering systems

### Other Domains
• Time series forecasting (LSTM, GRU)
• Reinforcement learning (DQN, PPO)
• Recommendation systems
• Drug discovery and protein folding

## Performance Optimization

### Computational Efficiency
• GPU acceleration using CUDA
• Mixed precision training (FP16/FP32)
• Distributed training across multiple devices
• Gradient accumulation for large batch sizes

### Memory Management
• Gradient checkpointing
• Model parallelism for large architectures
• Efficient data loading and preprocessing
• Memory-efficient optimizers

## Common Challenges and Solutions

### Vanishing/Exploding Gradients
**Problem**: Gradients become too small or too large in deep networks
**Solutions**: 
• Use ReLU or variants (Leaky ReLU, ELU)
• Implement residual connections
• Apply batch normalization
• Use gradient clipping

### Overfitting
**Problem**: Model memorizes training data
**Solutions**:
• Increase training data
• Apply regularization (L1/L2, dropout)
• Use data augmentation
• Implement early stopping

### Slow Convergence
**Problem**: Training takes too long
**Solutions**:
• Tune learning rate
• Use adaptive optimizers (Adam, RMSprop)
• Implement learning rate scheduling
• Improve initialization

## Best Practices

1. **Data Preparation**: Normalize inputs, handle missing values, augment data
2. **Architecture Design**: Start simple, add complexity gradually
3. **Hyperparameter Tuning**: Use systematic search (grid, random, Bayesian)
4. **Monitoring**: Track loss, accuracy, gradients, and validation metrics
5. **Reproducibility**: Set random seeds, version control code and data
6. **Documentation**: Record experiments, hyperparameters, and results

## Future Directions

The field continues to evolve with:
• Neural architecture search (NAS)
• Meta-learning and few-shot learning
• Efficient training methods (knowledge distillation)
• Neuromorphic computing
• Quantum machine learning

## Conclusion

${topic} remains the cornerstone of modern Machine Learning, enabling the training of increasingly sophisticated models. Understanding its principles, implementation details, and best practices is essential for anyone working in AI and ML. As the field advances, these fundamentals continue to underpin new innovations and applications.

## References and Further Reading

• Deep Learning (Goodfellow, Bengio, Courville)
• Neural Networks and Deep Learning (Nielsen)
• Pattern Recognition and Machine Learning (Bishop)
• Research papers on optimization in deep learning
• Online courses: fast.ai, deeplearning.ai, Stanford CS231n`,
  };

  return depthContent[depth];
}
