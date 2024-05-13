# YouTube Comments Classifier

## Overview:
The "YouTube Comments Classifier" project focuses on collecting comments from specified YouTube channels, cleaning and preprocessing the data, and then using machine learning and deep learning models to predict labels for these comments. The project consists of three main files: scrap.py, clean.py, and model.py.
 - *Note that the labels for the data were added manually to avoid misclassification. the comment such as "Poor Ben" does not classify as a negative or positive comment. Hence the neutral or class 0 was added*

## scrap.py
Description: The scraping script uses Selenium and BeautifulSoup to collect comments from predefined YouTube channels. It also performs data cleaning, language filtering, and exports the data to CSV files. It performs the following tasks:

### Web Driver Initialization: 
- Initializes a Chrome web driver using Selenium.

### YouTube Channels: 
- Specifies a list of YouTube channel URLs to scrape comments from.

### Scraping Process:
- Visits each channel URL.
- Extracts video links from the channel page.
- Iterates through the video links to extract comments using Selenium
- The comments, along with the corresponding user information, are stored in lists (users and comments).

### Data Cleaning:
- Creates a dictionary (dic) with users and comments.
- Converts the dictionary to a Pandas DataFrame (df).
- Saves the DataFrame to a CSV file named data.csv.

### Language Detection and Filtering:
- Removes null entries and resets the DataFrame index.
- Applies language detection to filter out non-English comments using the detect_language function (which is not provided in the code snippet).
- Saves the filtered DataFrame to a new CSV file named data.csv.

## clean.py
Description: This file cleans and preprocesses the collected comments, applying techniques such as spell checking, lemmatization, and language filtering. The cleaned data is saved to a new CSV file. It includes the following functionalities:

### Data Loading:
- Imports necessary libraries and downloads required resources.
- Reads the previously collected data from the "data.csv" CSV file into a Pandas DataFrame (df).

### Data Cleaning:
- Removes rows with empty or null comments.
- Resets the DataFrame index after removing rows.
- Applies the detect_language function to filter out non-English comments.

### Text Preprocessing:
- Defines a function preprocess for cleaning and correcting comments.
- Removes non-alphabetic characters from comments using regular expressions.
- Corrects misspelled words using the SpellChecker library.
- Tokenizes comments into words and removes stop words.
- Lemmatizes words based on their parts of speech using the WordNetLemmatizer.

### Additional Cleaning:
- Merges multiple lines within comments into a single line by replacing newline characters.
- Drops unnecessary columns and rows containing a specific phrase ("amazons amk33x").

### Data Export:
- Saves the cleaned DataFrame to a new CSV file named "cleaned.csv".

### Prints:
- Displays the cleaned DataFrame head and shape for verification.

## model.py
Description: The model script implements and evaluates machine learning (Logistic Regression and Naive Bayes) and deep learning (RNN and CNN) models for comment classification. It compares the models using accuracy scores and confusion matrices, displaying the results through various charts.

The model.py file is responsible for implementing and evaluating machine learning (ML) and deep learning (DL) models for classifying YouTube comments. It includes the following functionalities:

### Data Loading:
- Imports necessary libraries and loads the cleaned data from the "cleaned.csv" CSV file into a Pandas DataFrame (df).

### Data Preprocessing:
- Drops rows with missing values and unnecessary columns.

### Machine Learning Models (Logistic Regression and Naive Bayes):
- Splits the data into training and testing sets for ML models.
- Creates a Bag-of-Words (BoW) representation with n-grams using TF-IDF.
- Performs hyperparameter tuning using GridSearchCV for Logistic Regression and Naive Bayes.
- Trains the best models and evaluates them, displaying accuracy and confusion matrices.

### Deep Learning Models (RNN and CNN):
- Tokenizes text data for RNN using Keras Tokenizer.
- Splits the data for RNN into training and testing sets.
- Builds and trains a Simple RNN model and a CNN model for comment classification.
- Evaluates and displays accuracy and confusion matrices for both RNN and CNN.

### Model Comparison:
- Compares the accuracy of ML models (Logistic Regression and Naive Bayes) and DL models (RNN and CNN) using bar charts.
- Displays separate charts for ML models, DL models, and an overall comparison.
### Prints:
- Displays accuracy and confusion matrices for each model.

## Model Performance
- Logistic Regression: 70% accuracy
- Naive Bayes: 67% accuracy
-  RNN: 45% accuracy
-  CNN: 52% accuracy
