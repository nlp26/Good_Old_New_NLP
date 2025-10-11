Good_Old_New_NLP

Old NLP Solutions in a Modern Stack

This repository demonstrates classic NLP techniques reimagined using modern frameworks and libraries such as PyTorch, spaCy, and Hugging Face Transformers.
Each notebook revisits a traditional NLP task with current best practices, powerful pretrained models, and a clean, extensible structure.
------------------------------------------------
📘 1. Amazon Review Sentiment Analysis with LSTM
Notebook: amazon_torch_NLP.ipynb

🔍 Overview
Implements sentiment classification on the Amazon Polarity dataset using a Long Short-Term Memory (LSTM) network in PyTorch.

⚙️ Workflow
Data Loading: Import the dataset via datasets.load_dataset.
Preprocessing: Clean, tokenize, and label text.
Splitting: Train/validation data separation.
Vocabulary & Tokenizer: Build a torchtext vocabulary.
Custom Dataset: Define a PyTorch Dataset for batching.
Model Architecture: Create an LSTM classifier with nn.Embedding, nn.LSTM, and nn.Linear.
Training Loop: Optimize using Adam, monitor loss and metrics.
Evaluation: Compute accuracy, precision, recall, and F1-score.
Save & Predict: Export model weights and predict new samples.

🧪 Results
Tracks training/validation metrics.
Displays confusion matrix and evaluation summary.
Demonstrates reproducible performance on unseen data.

💾 How to Run
pip install datasets torch torchtext scikit-learn

Run all cells in Jupyter or Colab, then use predict_sentiment() for inference.

🚀 Extension Ideas
Test GRU, bi-LSTM, or Transformer architectures.
Add pretrained embeddings (GloVe, FastText).
Integrate attention mechanisms.
Apply hyperparameter tuning or experiment tracking (e.g., W&B).
------------------------------------------------
🧩 2. spaCy NVIDIA Product NLP Pipeline
Notebook: spacy_nvidia_track.ipynb

🔍 Overview
Explores spaCy 2025 capabilities through exhaustive linguistic analysis of NVIDIA product descriptions.

⚙️ Workflow
Tokenization, POS, and Dependency Parsing of product texts.
Named Entity Recognition (NER) using en_core_web_trf and en_core_web_md.
EntityRuler: Add domain-specific recognition rules for NVIDIA terms.
Vector Similarity: Compare semantic relationships between products.
Performance Benchmarking: Measure throughput of transformer vs. vector pipelines.

🧪 Results
Visual dependency graphs via displacy.
Entity comparison between transformer and standard models.
Timing analysis for scalability insights.

💾 How to Run
pip install spacy requests numpy
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_md

Run cells in sequence to analyze and visualize outputs.

🚀 Extension Ideas
Add similarity clustering using cosine distance.
Integrate SentenceTransformers for better embeddings.
Fine-tune custom NER on NVIDIA corpora.
Cache inferences and pipeline components for speed.
------------------------------------------------
🤖 3. Hugging Face Transformers — NVIDIA NLP Suite
Notebook: hf_nvidia_track.ipynb

🔍 Overview
Performs an end-to-end NLP evaluation of NVIDIA product data using Hugging Face Transformers pipelines.

⚙️ Tasks Covered
NER: Detects entities like product names and architectures.
Summarization: Generates concise descriptions.
Zero-shot Classification: Categorizes products (GPU, Platform, AI System).
Embedding Similarity: Computes semantic closeness among product names.

🧠 Models Used
Task	Model	Description
Named Entity Recognition	dslim/bert-base-NER	Robust general-purpose NER
Summarization	facebook/bart-large-cnn	High-quality abstractive summarization
Classification	facebook/bart-large-mnli	Zero-shot category assignment
Embeddings	sentence-transformers/all-MiniLM-L6-v2	Sentence-level semantic similarity

💾 How to Run
pip install transformers datasets torch sentencepiece accelerate

Run sequentially in Jupyter or Colab. Modify candidate_labels for custom category evaluation.

🚀 Extension Ideas
Swap MiniLM with bge-large or text-embedding-3-large.
Use faiss or qdrant for scalable similarity search.
Add question-answering for detailed product insights.
Integrate results with FastAPI or Streamlit for live demos.

🧭 Repository Vision
Good_Old_New_NLP bridges the past and future of NLP — modernizing foundational methods (like LSTMs and NER) with cutting-edge transformer ecosystems, providing a practical comparative framework for educational and production use.
