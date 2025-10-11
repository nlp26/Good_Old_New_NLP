# Good_Old_New_NLP
Old NLP Solutions in modern stack

-------------------------------------------------------------------------------------------------------------------------------

https://github.com/nlp26/Good_Old_New_NLP/blob/main/amazon_torch_NLP.ipynb

"""
README - Amazon Review Sentiment Analysis with LSTM

This project demonstrates a simple sentiment analysis model using a Long Short-Term Memory (LSTM) network in PyTorch. The model is trained on the Amazon Polarity dataset to classify reviews as either positive or negative.

Project Steps
Import Libraries: Essential libraries for data handling, processing, and model building are imported.
Load Dataset: The Amazon Polarity dataset is loaded using the datasets library.
Preprocess Data: Review texts and their corresponding labels are extracted and prepared.
Split Data: The dataset is split into training and validation sets.
Tokenization & Vocabulary: Text data is tokenized, and a vocabulary is built from the training data.
Custom PyTorch Dataset: A custom Dataset class is created to handle data loading and processing for PyTorch DataLoaders.
DataLoader Setup: DataLoaders are set up for efficient batching of training and validation data.
Model Definition: A simple LSTM-based classification model is defined using PyTorch's nn.Module.
Training Setup: The optimizer and loss function are configured for model training.
Training Loop: The model is trained on the training data for a specified number of epochs.
Evaluation: The trained model is evaluated on the validation set, and metrics like accuracy, precision, recall, and F1-score are calculated.
Save Model: The trained model's state dictionary is saved for future use.
Load and Predict: The saved model is loaded, and a function is provided to predict the sentiment of new review texts.
How to Run
Ensure you have the necessary libraries installed (datasets, torch, torchtext, scikit-learn). You may need to run !pip install datasets torch torchtext scikit-learn.
Execute the code cells sequentially in a Python environment like Google Colab or a local Jupyter Notebook.
This notebook provides a basic example of building a text classification model using LSTMs. It can be extended further by exploring different model architectures, hyperparameter tuning, and more advanced text preprocessing techniques.
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
