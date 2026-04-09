# Fake News Detection using Machine Learning (TF-IDF + Logistic Regression)

## Overview

This project is a machine learning-based fake news detection system that classifies news articles as Real or Fake using Natural Language Processing (NLP) techniques. It uses TF-IDF vectorization for feature extraction and Logistic Regression for classification.

---

## Dataset

The dataset consists of two CSV files:

* `true.csv` containing real news articles
* `fake.csv` containing fake news articles

---

## Features

* Data preprocessing and cleaning
* Handling missing values
* Combining title and text content
* TF-IDF feature extraction
* Logistic Regression model training
* Model evaluation using accuracy score, confusion matrix, and classification report
* Visualization of results

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## Workflow

1. Load datasets
2. Assign labels (Real = 1, Fake = 0)
3. Merge and shuffle data
4. Preprocess text data
5. Split dataset into training and testing sets
6. Convert text to numerical features using TF-IDF
7. Train Logistic Regression model
8. Evaluate model performance
9. Visualize results

---

## Output

* Model accuracy displayed in the console
* Confusion matrix printed and saved as an image
* Classification report showing precision, recall, and F1-score
* Training vs testing accuracy comparison graph

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/your-username/fake-news-detection.git
```

Navigate to the project directory:

```bash
cd fake-news-detection
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the project:

```bash
python main.py
```

---

## Folder Structure

```
fake-news-detection/
│
├── data/
│   ├── true.csv
│   └── fake.csv
│
├── figures/
│   └── Figure3_1_ConfusionMatrix.png
│
├── main.py
└── README.md
```

---

## Future Improvements

* Implement deep learning models such as LSTM or BERT
* Build a web interface using Flask or Streamlit
* Enable real-time news classification
