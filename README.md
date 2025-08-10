# Spam Email Classifier

## Project Overview

This project implements a machine learning-based spam email classifier. It's a full-stack data science project that demonstrates the process of building, training, evaluating, and deploying a text classification model. The goal is to classify SMS messages as either "ham" (legitimate) or "spam" with high accuracy.

## Dataset

The model is trained on the **UCI SMS Spam Collection Dataset**, which contains labeled SMS messages.

## Project Structure

The repository is organized into the following directories:

* `data/`: Contains the raw dataset (`spam.csv`).
* `notebooks/`: Contains Jupyter Notebooks for exploratory data analysis (`eda.ipynb`) and model development (`spam_classifier.ipynb`).
* `src/`: Contains modular Python scripts for reusable functions (e.g., `preprocess.py` for text cleaning, `train.py` for model training).
* `models/`: Stores the trained machine learning models and vectorizers (`.pkl` files).
* `app.py`: The Python script for the Streamlit web application.

## Getting Started

### Prerequisites

You need to have Python 3.x installed. It's recommended to use a virtual environment.

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd spam-email-classifier
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Run the Jupyter Notebooks:**
    Navigate to the `notebooks/` directory and open `spam_classifier.ipynb` to see the full data science pipeline, from cleaning to model evaluation.

2.  **Run the Streamlit App:**
    After training the model and ensuring the `.pkl` files are saved in the `models/` directory, run the app from the project's root directory:
    ```bash
    streamlit run app.py
    ```
    The app will open in your web browser, allowing you to test the classifier.

## Model Details

* **Preprocessing:** Lowercasing, punctuation and stopword removal, and stemming.
* **Vectorization:** `TfidfVectorizer`.
* **Class Imbalance:** Handled using `SMOTE` during model training.
* **Model:** `LogisticRegression`, with hyperparameters tuned using `GridSearchCV`.
* **Evaluation:** The model achieves high scores on Precision, Recall, and F1-Score, with a low number of false positives (misclassified ham messages).

## Deliverables

* `spam_classifier.ipynb`: A comprehensive notebook detailing the project steps.
* `requirements.txt`: A list of all project dependencies.
* `app.py`: The code for the Streamlit web application.
* Trained models and vectorizers (`.pkl` files).
