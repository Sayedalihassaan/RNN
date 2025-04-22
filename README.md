## Sentiment Analysis with Recurrent Neural Networks on IMDB 

## Dataset
This repository contains the code and documentation for a sentiment analysis project using Recurrent Neural Networks (RNNs) and Bidirectional LSTMs on the IMDB movie review dataset. The project classifies movie reviews as positive or negative, demonstrating a complete natural language processing (NLP) pipeline, including data preprocessing, model training, evaluation, and inference.
Table of Contents

Project Overview
Features
Installation
Usage
Project Structure
Dataset
Methodology
Results
Future Improvements
Contributing
License
Acknowledgements

Project Overview
The goal of this project is to build and compare two deep learning models for sentiment analysis: a Simple RNN and a Bidirectional LSTM. Both models are trained on the IMDB dataset to predict whether a movie review expresses a positive or negative sentiment. The project showcases key NLP techniques, including word embedding, sequence padding, and model evaluation, with a focus on leveraging TensorFlow for GPU-accelerated training.
Features

Data Preprocessing: Loads and preprocesses the IMDB dataset, including tokenization and sequence padding.
Word Embeddings: Uses an embedding layer to convert words into dense vectors.
Model Architectures:
Simple RNN with two stacked layers.
Bidirectional LSTM with two stacked layers for improved context understanding.


Training: Implements early stopping to prevent overfitting and optimize training.
Evaluation: Visualizes training and validation accuracy/loss curves and computes test set performance.
Inference: Provides a function to classify new movie reviews as positive or negative.
Model Persistence: Saves and loads the trained model for reuse.

Installation
To run this project locally, follow these steps:

Clone the Repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies: The required libraries are listed in requirements.txt. Install them using:
pip install -r requirements.txt

The requirements.txt includes:
tensorflow==2.17.0
numpy==2.1.1
pandas==2.2.3
matplotlib==3.9.2
seaborn==0.13.2


Optional: GPU Support: Ensure you have a CUDA-compatible GPU and install TensorFlow with GPU support if needed:
pip install tensorflow[and-cuda]

Verify GPU availability in TensorFlow:
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))



Usage

Run the Notebook: Open the RNN.ipynb notebook in Jupyter or Google Colab:
jupyter notebook RNN.ipynb

Execute the cells sequentially to preprocess data, train models, evaluate performance, and perform inference.

Inference with the Trained Model: Use the provided inferance function to classify a new movie review:
from tensorflow.keras.models import load_model

model = load_model("model_saveed.keras")
text = "This movie was absolutely fantastic and thrilling!"
prediction = inferance(text, model)
print(f"Sentiment: {prediction}")

The function returns "Positive" or "Negative" based on the model's prediction.

Visualize Results: The notebook generates plots for training/validation accuracy and loss, saved in the notebook output or exportable as images.


Project Structure
your-repo-name/
├── RNN.ipynb                   # Main notebook with the project code
├── model_saveed.keras         # Saved Bidirectional LSTM model
├── requirements.txt           # Dependencies for the project
├── README.md                  # Project documentation
└── LICENSE                    # License file (e.g., MIT)

Dataset
The IMDB dataset is sourced from TensorFlow's tf.keras.datasets.imdb. It contains:

Training Set: 25,000 movie reviews.
Test Set: 25,000 movie reviews.
Labels: Binary (0 for negative, 1 for positive).
Vocabulary: Limited to the top 10,000 most frequent words.
Sequence Length: Reviews are padded/truncated to 300 tokens.

Methodology
The project follows a structured NLP pipeline:

Data Loading: Loads the IMDB dataset with a vocabulary size of 10,000.
Preprocessing:
Converts word indices to text for interpretability.
Pads sequences to a fixed length of 300 tokens.


Model Architectures:
Simple RNN: Two stacked RNN layers (64 and 32 units) with an embedding layer (128 dimensions).
Bidirectional LSTM: Two stacked bidirectional LSTM layers (64 and 32 units) for capturing bidirectional context.


Training:
Uses binary cross-entropy loss and Adam optimizer (learning rate 0.01).
Implements early stopping with a patience of 3 epochs based on validation loss.


Evaluation:
Plots training and validation accuracy/loss curves.
Computes test set accuracy and loss.


Inference:
Provides a function to preprocess and classify new text inputs.



Tools and Libraries



Library
Purpose



tensorflow
Model building, training, and inference


numpy
Numerical operations


pandas
Data manipulation (unused but included)


matplotlib
Plotting accuracy/loss curves


seaborn
Enhanced visualizations (unused but included)


Results

Simple RNN Performance:
Test Accuracy: ~50% (indicating poor performance, likely due to vanishing gradients).
Test Loss: ~0.693 (close to random guessing).


Bidirectional LSTM:
Not fully trained in the notebook, but expected to outperform Simple RNN due to better handling of long-term dependencies.


Visualizations:
Accuracy and loss curves show the Simple RNN's lack of convergence, highlighting the need for LSTM or other improvements.


Inference Example:
The provided review is classified as "Positive" by the Bidirectional LSTM model, though this may not reflect trained performance.



Note: The Simple RNN results suggest it struggles with the IMDB dataset. The Bidirectional LSTM model is defined but not trained in the notebook. Training it would likely yield better results.
Future Improvements

Train the LSTM Model: Complete training and evaluation of the Bidirectional LSTM model.
Hyperparameter Tuning: Experiment with learning rates, batch sizes, and layer sizes.
Advanced Architectures: Incorporate GRU or Transformer-based models (e.g., BERT).
Data Augmentation: Use techniques like synonym replacement to increase dataset diversity.
Error Analysis: Analyze misclassified reviews to identify patterns and improve model robustness.
Preprocessing Enhancements: Implement stemming or lemmatization to reduce vocabulary size.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please ensure your code follows the project's coding style and includes relevant tests.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgements

TensorFlow: For providing the IMDB dataset and deep learning framework.
Keras: For the high-level API used in model building.
IMDB Dataset: For the publicly available movie review data.


Feel free to star ⭐ this repository if you find it useful! For questions or feedback, open an issue or contact me at [your-email@example.com].
