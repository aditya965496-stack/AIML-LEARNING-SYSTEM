import { useState } from "react";
import { Code, Sparkles, Download, Copy } from "lucide-react";
import { Header } from "~/components/header/header";
import { Button } from "~/components/ui/button/button";
import { useToast } from "~/hooks/use-toast";
import styles from "./code-generation.module.css";

export default function CodeGeneration() {
  const [description, setDescription] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedCode, setGeneratedCode] = useState("");
  const [dependencies, setDependencies] = useState<string[]>([]);
  const { toast } = useToast();

  const handleGenerate = async () => {
    if (!description.trim()) {
      toast({
        title: "Description Required",
        description: "Please describe the ML code you want to generate.",
        variant: "destructive",
      });
      return;
    }

    setIsGenerating(true);

    // Simulate AI generation with mock data
    setTimeout(() => {
      const { code, deps } = generateMockCode(description);
      setGeneratedCode(code);
      setDependencies(deps);
      setIsGenerating(false);
      toast({
        title: "Code Generated",
        description: "Your Python implementation is ready!",
      });
    }, 2500);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(generatedCode);
    toast({
      title: "Copied",
      description: "Code copied to clipboard.",
    });
  };

  const handleDownload = () => {
    const blob = new Blob([generatedCode], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ml_code.py";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Downloaded",
      description: "Code saved as ml_code.py",
    });
  };

  return (
    <div className={styles.container}>
      <Header />

      <main className={styles.content}>
        <h1 className={styles.pageTitle}>
          <Code className={styles.pageIcon} />
          Code Generation
        </h1>
        <p className={styles.pageDescription}>
          Generate production-ready Python code for Machine Learning algorithms. Get complete implementations with
          detailed comments, automatic dependency detection, and execution instructions.
        </p>

        <div className={styles.inputSection}>
          <div className={styles.formGroup}>
            <label htmlFor="description" className={styles.label}>
              Describe the ML Code You Need
            </label>
            <textarea
              id="description"
              className={styles.textarea}
              placeholder="e.g., Neural network with backpropagation from scratch, LSTM for time-series forecasting, K-means clustering implementation..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={4}
            />
          </div>

          <button className={styles.generateButton} onClick={handleGenerate} disabled={isGenerating}>
            {isGenerating ? (
              <>
                <Sparkles className={styles.loadingIcon} />
                Generating Code...
              </>
            ) : (
              <>
                <Code />
                Generate Python Code
              </>
            )}
          </button>
        </div>

        {generatedCode ? (
          <div className={styles.resultSection}>
            <div className={styles.resultHeader}>
              <h2 className={styles.resultTitle}>Generated Implementation</h2>
              <div style={{ display: "flex", gap: "var(--space-2)" }}>
                <Button variant="outline" size="sm" onClick={handleCopy}>
                  <Copy style={{ width: 16, height: 16 }} />
                  Copy
                </Button>
                <Button variant="outline" size="sm" onClick={handleDownload}>
                  <Download style={{ width: 16, height: 16 }} />
                  Download
                </Button>
              </div>
            </div>

            <div className={styles.codeBlock}>
              <code className={styles.code}>{generatedCode}</code>
            </div>

            {dependencies.length > 0 && (
              <div className={styles.dependencies}>
                <h3 className={styles.dependenciesTitle}>Required Dependencies</h3>
                <div className={styles.dependenciesList}>
                  {dependencies.map((dep) => (
                    <span key={dep} className={styles.dependencyBadge}>
                      {dep}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className={styles.instructions}>
              <h3 className={styles.instructionsTitle}>Execution Instructions</h3>
              <ol className={styles.instructionsList}>
                <li>Install required dependencies: pip install {dependencies.join(" ")}</li>
                <li>Copy the code to a Python file (e.g., ml_code.py)</li>
                <li>Run the script: python ml_code.py</li>
                <li>For Google Colab: Upload the file or paste the code into a new cell</li>
                <li>Modify hyperparameters and data paths as needed for your use case</li>
              </ol>
            </div>
          </div>
        ) : (
          <div className={styles.emptyState}>
            <Code className={styles.emptyIcon} />
            <p className={styles.emptyText}>Describe your ML implementation and generate production-ready code</p>
          </div>
        )}
      </main>
    </div>
  );
}

function generateMockCode(description: string): { code: string; deps: string[] } {
  const code = `"""
Machine Learning Implementation
Generated by GyanGuru AI Learning Assistant

Description: ${description}
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

class NeuralNetwork:
    """
    A simple feedforward neural network with backpropagation.
    
    This implementation demonstrates the core concepts of:
    - Forward propagation
    - Backpropagation algorithm
    - Gradient descent optimization
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize the neural network with random weights.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in hidden layer
            output_size (int): Number of output neurons
            learning_rate (float): Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))
        
        # Store activations for backpropagation
        self.hidden_activation = None
        self.output_activation = None
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1 - x)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Args:
            X (np.ndarray): Input data of shape (batch_size, input_size)
            
        Returns:
            np.ndarray: Network output of shape (batch_size, output_size)
        """
        # Hidden layer computation
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_activation = self.sigmoid(hidden_input)
        
        # Output layer computation
        output_input = np.dot(self.hidden_activation, self.weights_hidden_output) + self.bias_output
        self.output_activation = self.sigmoid(output_input)
        
        return self.output_activation
    
    def backward(self, X, y, output):
        """
        Backpropagation algorithm to compute gradients.
        
        Args:
            X (np.ndarray): Input data
            y (np.ndarray): True labels
            output (np.ndarray): Network predictions
        """
        batch_size = X.shape[0]
        
        # Calculate output layer error
        output_error = output - y
        output_delta = output_error * self.sigmoid_derivative(output)
        
        # Calculate hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_activation)
        
        # Update weights and biases using gradient descent
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_activation.T, output_delta) / batch_size
        self.bias_output -= self.learning_rate * np.sum(output_delta, axis=0, keepdims=True) / batch_size
        
        self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_delta) / batch_size
        self.bias_hidden -= self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True) / batch_size
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network using backpropagation.
        
        Args:
            X (np.ndarray): Training data
            y (np.ndarray): Training labels
            epochs (int): Number of training iterations
            verbose (bool): Whether to print training progress
            
        Returns:
            list: Training loss history
        """
        loss_history = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss (Mean Squared Error)
            loss = np.mean((output - y) ** 2)
            loss_history.append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return loss_history
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Predictions
        """
        return self.forward(X)


def generate_sample_data(n_samples=1000, n_features=10):
    """
    Generate synthetic dataset for demonstration.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of input features
        
    Returns:
        tuple: (X, y) training data and labels
    """
    X = np.random.randn(n_samples, n_features)
    # Create a simple linear relationship with noise
    weights = np.random.randn(n_features, 1)
    y = (np.dot(X, weights) + np.random.randn(n_samples, 1) * 0.1 > 0).astype(float)
    return X, y


def plot_training_history(loss_history):
    """
    Visualize the training loss over epochs.
    
    Args:
        loss_history (list): List of loss values during training
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Neural Network with Backpropagation")
    print("GyanGuru AI Learning Assistant")
    print("=" * 60)
    print()
    
    # Generate sample data
    print("Generating sample dataset...")
    X, y = generate_sample_data(n_samples=1000, n_features=10)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print()
    
    # Create and train the neural network
    print("Initializing neural network...")
    nn = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=20,
        output_size=1,
        learning_rate=0.1
    )
    
    print("Training neural network...")
    print()
    loss_history = nn.train(X_train, y_train, epochs=1000, verbose=True)
    
    # Make predictions
    print()
    print("Evaluating model performance...")
    train_predictions = nn.predict(X_train)
    test_predictions = nn.predict(X_test)
    
    # Calculate accuracy (binary classification)
    train_pred_binary = (train_predictions > 0.5).astype(float)
    test_pred_binary = (test_predictions > 0.5).astype(float)
    
    train_accuracy = accuracy_score(y_train, train_pred_binary)
    test_accuracy = accuracy_score(y_test, test_pred_binary)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print()
    
    # Visualize training progress
    print("Plotting training history...")
    plot_training_history(loss_history)
    
    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
`;

  const deps = ["numpy", "matplotlib", "scikit-learn"];

  return { code, deps };
}
