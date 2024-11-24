
# Yelp Review NLP Classification Project

This repository contains a Natural Language Processing (NLP) project aimed at classifying Yelp reviews into 1-star or 5-star categories based on their textual content. The project utilizes a combination of machine learning and text processing techniques to achieve classification, demonstrating two approaches:

- Count Vectorizer with Naive Bayes
- TF-IDF Transformation with Naive Bayes (with insights on performance differences)

## Table of Contents
- Introduction
- Dataset
- Project Workflow
- Requirements
- Setup Instructions
- EDA and Visualizations
- Modeling Approaches
- Results
- Future Improvements
- Contributing
- License
- Acknowledgments

## Introduction
In this project, we aim to analyze and classify Yelp reviews using NLP techniques. By leveraging text data, the goal is to predict whether a review belongs to the 1-star or 5-star category based on its content. The project includes exploratory data analysis (EDA), feature engineering, and building classification pipelines using scikit-learn.

## Dataset
The dataset used in this project is the Yelp Review Dataset, containing customer reviews, ratings, and metadata for various businesses.

**Key Features:**
- `stars`: Star rating assigned by the user (1-5).
- `text`: Text content of the review.
- `cool`, `useful`, `funny`: Metadata for votes on the review.
- Additional metadata like `business_id`, `user_id`, and `date`.

## Project Workflow
The project is divided into the following steps:

1. **Data Loading and Preprocessing**
   - Load the data, handle missing values, and create new features such as text length.

2. **Exploratory Data Analysis (EDA)**
   - Visualize relationships between star ratings and text features using histograms, boxplots, and heatmaps.

3. **Text Feature Extraction**
   - Convert text into numerical features using:
     - CountVectorizer
     - TF-IDF Transformer

4. **Model Building**
   - Build a classification model using Multinomial Naive Bayes for text classification.

5. **Evaluation**
   - Evaluate the models using confusion matrices and classification reports to analyze precision, recall, and F1-scores.

6. **Pipeline Implementation**
   - Create a pipeline for preprocessing and classification to streamline model training and evaluation.

## Requirements
To run this project, ensure the following dependencies are installed:

- Python (3.7+)
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Install all requirements using:

```bash
pip install -r requirements.txt
```

## Setup Instructions
Clone the repository:

```bash
git clone https://github.com/your-username/yelp-nlp-classification.git
cd yelp-nlp-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset:

Place the `yelp.csv` file in the project directory.

Run the Jupyter Notebook for EDA and model building:

```bash
jupyter notebook Yelp_Review_Classification.ipynb
```

For pipeline execution, use:

```bash
python pipeline.py
```

## EDA and Visualizations
The project includes several visualizations to understand the data:

- **Histograms**: Distribution of text length by star ratings.
- **Boxplots**: Variation in text length for different ratings.
- **Countplots**: Frequency of each star rating.
- **Heatmaps**: Correlation between numerical features (e.g., cool, funny, useful, text length).

## Modeling Approaches
1. **Count Vectorizer + Multinomial Naive Bayes**
   - Converts text into token counts.
   - Trains a Naive Bayes classifier on the tokenized data.

2. **TF-IDF Transformer + Multinomial Naive Bayes**
   - Enhances Count Vectorizer features using term frequency-inverse document frequency (TF-IDF).
   - Observed to perform poorly due to over-normalization in this specific task.

## Results
- **Model 1: Count Vectorizer + Naive Bayes**
  - Accuracy: ~90%
  - Precision (1-star): 84%
  - Precision (5-star): 92%

- **Model 2: TF-IDF + Naive Bayes**
  - Accuracy: ~81%
  - Poor performance due to data imbalance and lack of discriminative features.

## Future Improvements
- Experiment with other machine learning models (e.g., Random Forest, SVM).
- Perform hyperparameter tuning for better model optimization.
- Address class imbalance using techniques like SMOTE or weighted loss functions.
- Incorporate pre-trained embeddings (e.g., Word2Vec, GloVe) or transformer models (e.g., BERT).

## Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Added new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Special thanks to:
- Kaggle for providing the Yelp Review dataset.
- Scikit-learn documentation for machine learning guidance.


