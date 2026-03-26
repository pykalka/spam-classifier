📧 Spam Classifier (Machine Learning Project)

📌 Overview

This project is a machine learning-based spam classifier that detects whether a message is spam or not.

It implements a full ML pipeline including training, prediction, model persistence, explainability, and online learning.

🚀 Features
Text preprocessing and vectorization (TF-IDF)
Model training using Naive Bayes
Model persistence (saving/loading with joblib)
Spam prediction for user input
Model explainability using LIME
Online learning (incremental model updates)

🛠️ Tech Stack
Python
NumPy
Pandas
scikit-learn
LIME
Joblib
Matplotlib

📊 Model Details
Algorithm: Multinomial Naive Bayes
Vectorization: TF-IDF
Train/Test split: 80/20

Metrics:

Accuracy: 97.93%

▶️ How to Run
1. Clone repository
git clone https://github.com/pykalka/spam-classifier.git
cd spam-classifier
2. Install dependencies
pip install -r requirements.txt
3. Run project
cd .venv
python main.py

💡 Example

Input:

You won a free ticket! Call now!

Output:

Spam (93.94%)

📂 Project Structure
src/        - source code
model/      - saved model
data/       - dataset
main.py     - entry point
