export interface Lesson {
  slug: string;
  module: number;
  lesson: number;
  title: string;
  description: string;
  hasQuiz: boolean;
}

export interface Module {
  number: number;
  title: string;
  description: string;
  lessons: Lesson[];
}

export const modules: Module[] = [
  {
    number: 0,
    title: "Python for ML",
    description: "Bridge from intermediate Python to ML-ready Python",
    lessons: [
      {
        slug: "0.1",
        module: 0,
        lesson: 1,
        title: "NumPy & Vectorized Thinking",
        description: "NDArrays, broadcasting, and why vectors beat loops",
        hasQuiz: true,
      },
      {
        slug: "0.2",
        module: 0,
        lesson: 2,
        title: "Pandas, Matplotlib & the ML Data Pipeline",
        description: "DataFrames, visualization, and the typical ML workflow",
        hasQuiz: true,
      },
    ],
  },
  {
    number: 1,
    title: "Math Refreshers",
    description: "Just enough math to understand ML — no more, no less",
    lessons: [
      {
        slug: "1.1",
        module: 1,
        lesson: 1,
        title: "Linear Algebra for ML",
        description: "Vectors, matrices, and what they mean for ML",
        hasQuiz: true,
      },
      {
        slug: "1.2",
        module: 1,
        lesson: 2,
        title: "Statistics & Probability for ML",
        description: "Distributions, Bayes, and why ML cares about variance",
        hasQuiz: true,
      },
      {
        slug: "1.3",
        module: 1,
        lesson: 3,
        title: "Calculus Intuition (Gradients & Optimization)",
        description: "Derivatives, gradients, and gradient descent",
        hasQuiz: true,
      },
    ],
  },
  {
    number: 2,
    title: "Classical Machine Learning",
    description: "The foundational algorithms that underpin all of ML",
    lessons: [
      {
        slug: "2.1",
        module: 2,
        lesson: 1,
        title: "What Is Machine Learning?",
        description: "The big picture: supervised, unsupervised, and the ML workflow",
        hasQuiz: true,
      },
      {
        slug: "2.2",
        module: 2,
        lesson: 2,
        title: "Linear Regression & the Cost Function",
        description: "Fitting lines, measuring error, and gradient descent in action",
        hasQuiz: true,
      },
      {
        slug: "2.3",
        module: 2,
        lesson: 3,
        title: "Classification: Logistic Regression & Beyond",
        description: "Sigmoid, decision boundaries, and evaluation metrics",
        hasQuiz: true,
      },
      {
        slug: "2.4",
        module: 2,
        lesson: 4,
        title: "Decision Trees, Random Forests & Ensemble Methods",
        description: "Splitting, bagging, boosting, and why ensembles win",
        hasQuiz: true,
      },
      {
        slug: "2.5",
        module: 2,
        lesson: 5,
        title: "Unsupervised Learning: Clustering & Dimensionality Reduction",
        description: "K-means, PCA, and finding structure without labels",
        hasQuiz: true,
      },
    ],
  },
  {
    number: 3,
    title: "Neural Networks & Deep Learning",
    description: "From perceptrons to PyTorch — understanding deep learning",
    lessons: [
      {
        slug: "3.1",
        module: 3,
        lesson: 1,
        title: "The Perceptron & Neural Network Fundamentals",
        description: "Neurons, activation functions, and network architecture",
        hasQuiz: true,
      },
      {
        slug: "3.2",
        module: 3,
        lesson: 2,
        title: "Backpropagation & Training Neural Networks",
        description: "Forward pass, backward pass, and how networks learn",
        hasQuiz: true,
      },
      {
        slug: "3.3",
        module: 3,
        lesson: 3,
        title: "Deep Learning with PyTorch",
        description: "Tensors, autograd, and building networks with a framework",
        hasQuiz: true,
      },
      {
        slug: "3.4",
        module: 3,
        lesson: 4,
        title: "Convolutional Neural Networks (Intro to Vision)",
        description: "Convolutions, feature maps, and image classification",
        hasQuiz: true,
      },
      {
        slug: "3.5",
        module: 3,
        lesson: 5,
        title: "Regularization, Optimization & Practical Deep Learning",
        description: "Dropout, Adam, transfer learning, and practical tips",
        hasQuiz: true,
      },
    ],
  },
  {
    number: 4,
    title: "NLP & LLMs",
    description: "From bag-of-words to transformers — understanding language AI",
    lessons: [
      {
        slug: "4.1",
        module: 4,
        lesson: 1,
        title: "Text Representation & Classical NLP",
        description: "Bag of words, TF-IDF, and word embeddings",
        hasQuiz: true,
      },
      {
        slug: "4.2",
        module: 4,
        lesson: 2,
        title: "Sequence Models: RNNs & LSTMs",
        description: "Hidden state, vanishing gradients, and gating mechanisms",
        hasQuiz: true,
      },
      {
        slug: "4.3",
        module: 4,
        lesson: 3,
        title: "The Transformer Architecture",
        description: "Self-attention, multi-head attention, and positional encoding",
        hasQuiz: true,
      },
      {
        slug: "4.4",
        module: 4,
        lesson: 4,
        title: "Large Language Models: How GPT & Claude Work",
        description: "Pretraining, tokenization, scaling laws, and RLHF",
        hasQuiz: true,
      },
      {
        slug: "4.5",
        module: 4,
        lesson: 5,
        title: "Embeddings, RAG & Building with LLMs",
        description: "Vector databases, RAG, prompt engineering, and agents",
        hasQuiz: true,
      },
    ],
  },
  {
    number: 5,
    title: "Capstone Project",
    description: "Build an end-to-end ML prediction service",
    lessons: [
      {
        slug: "5.1",
        module: 5,
        lesson: 1,
        title: "Phase 1: Data Pipeline & Exploration",
        description: "Load, explore, clean, and prepare a real-world dataset",
        hasQuiz: false,
      },
      {
        slug: "5.2",
        module: 5,
        lesson: 2,
        title: "Phase 2: Model Building & Training",
        description: "Baseline model, neural network, and experimentation",
        hasQuiz: false,
      },
      {
        slug: "5.3",
        module: 5,
        lesson: 3,
        title: "Phase 3: API, Serving & Putting It All Together",
        description: "FastAPI backend, model serving, and documentation",
        hasQuiz: false,
      },
    ],
  },
];

export function getAllLessons(): Lesson[] {
  return modules.flatMap((m) => m.lessons);
}

export function getLesson(slug: string): Lesson | undefined {
  return getAllLessons().find((l) => l.slug === slug);
}

export function getModule(number: number): Module | undefined {
  return modules.find((m) => m.number === number);
}

export function getAdjacentLessons(slug: string): {
  prev: Lesson | null;
  next: Lesson | null;
} {
  const all = getAllLessons();
  const idx = all.findIndex((l) => l.slug === slug);
  return {
    prev: idx > 0 ? all[idx - 1] : null,
    next: idx < all.length - 1 ? all[idx + 1] : null,
  };
}
