# Good_Old_New_NLP
Old NLP Solutions in modern stack

-------------------------------------------------------------------------------------------------------------------------------

https://github.com/nlp26/Good_Old_New_NLP/blob/main/amazon_torch_NLP.ipynb

"""
README — Amazon Review Sentiment Analysis with LSTM
This notebook demonstrates a complete workflow for sentiment analysis on the Amazon Polarity dataset using a Long Short-Term Memory (LSTM) network built with PyTorch.

🔍 Overview
Trains an LSTM model to classify Amazon reviews as positive or negative.
Uses the datasets library for data handling and torchtext for tokenization and batching.
Implements all major NLP preprocessing, training, and evaluation steps in a reproducible pipeline.

⚙️ What the Notebook Covers
Data Loading – Load Amazon Polarity dataset via datasets.load_dataset.
Preprocessing – Extract review texts and labels; perform text cleaning and tokenization.
Splitting – Create training and validation subsets.
Vocabulary & Tokenizer – Build a word-level vocabulary using torchtext.
Custom Dataset – Define a PyTorch Dataset for batched text inputs.
DataLoaders – Create loaders for efficient mini-batch training.
Model Architecture – Define a simple LSTM classifier using nn.Embedding, nn.LSTM, and nn.Linear.
Training Loop – Optimize model with Adam, compute loss (BCE/CE), and track metrics per epoch.
Evaluation – Compute accuracy, precision, recall, and F1-score.
Model Saving – Save the trained model’s state_dict for reuse.
Prediction – Provide a helper function to classify new text samples.

🧪 Performance & Results
Measures training and validation accuracy per epoch.
Outputs confusion matrix and per-class performance metrics.
Demonstrates generalization ability across validation data.

💾 How to Run
Ensure you have dependencies installed:
pip install datasets torch torchtext scikit-learn
Open the notebook in Jupyter or Google Colab.
Run all cells sequentially — model training and evaluation will execute automatically.
Use the provided predict_sentiment() function for inference on custom text inputs.

🚀 Extension Ideas
Try GRU, bi-LSTM, or Transformer-based architectures.
Use pretrained embeddings (e.g., GloVe, FastText).
Perform hyperparameter tuning (learning rate, hidden size, dropout).
Add attention mechanisms for explainable predictions.
Integrate experiment tracking with tools like Weights & Biases or MLflow.
"""

-------------------------------------------------------------------------------------------------------------------------------

https://github.com/nlp26/Good_Old_New_NLP/blob/main/spacy_nvidia_track.ipynb

"""
README — spaCy NVIDIA Test Notebook

This notebook demonstrates how to test spaCy comprehensively:

1. **Tokenization, POS, Dependency, and NER** tested on real NVIDIA product data.
2. **Transformer vs vector models** compared side-by-side.
3. **EntityRuler** shows domain-specific rule enhancement.
4. **Vector similarity** evaluates semantic proximity to hardware terms.
5. **Performance metrics** benchmark model latency.

### Extension ideas
- Add similarity clustering (using cosine distances)
- Integrate SentenceTransformers for richer semantic embeddings
- Train custom NER with NVIDIA-specific dataset
- Cache model inference for larger datasets

Use this as a baseline to evaluate new spaCy releases or GPU-optimized pipelines.
"""
-------------------------------------------------------------------------------------------------------------------------------



