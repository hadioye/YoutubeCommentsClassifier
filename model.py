import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split , GridSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.util import ngrams
from nltk import word_tokenize
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM , Conv1D, GlobalMaxPooling1D
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.read_csv("cleaned.csv")

# Drop rows with missing values and unnecessary column
df.dropna(inplace=True)
df.drop("Unnamed: 0", axis=1, inplace=True)

# Splitting data into train and test sets for Logistic Regression and Naive Bayes
X = df['Comments']
y = df['labels']

X_trainML, X_testML, y_trainML, y_testML = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Bag-of-Words (BoW) representation with n-grams for Logistic Regression and Naive Bayes
ngram_range = (1, 2)
vectorizer_tfidf = TfidfVectorizer(ngram_range=ngram_range, tokenizer=word_tokenize)
X_trainML = vectorizer_tfidf.fit_transform(X_trainML)
X_testML = vectorizer_tfidf.transform(X_testML)

param_lr = {
    'max_iter' : [1000,2000,5000],
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs'],
}

param_nb = {
    'alpha': [0.1, 1, 10],
}

#logistic regression
gscv_lr = GridSearchCV(LogisticRegression() , param_grid= param_lr , cv = 2, scoring='accuracy')
gscv_lr.fit(X_trainML , y_trainML)

best_model_lr = gscv_lr.best_estimator_
best_model_lr.fit(X_trainML, y_trainML)

y_pred_lr = best_model_lr.predict(X_testML)
accuracylr = accuracy_score(y_testML, y_pred_lr)

cm_lr = confusion_matrix(y_testML, y_pred_lr) 
cm_lr = cm_lr.astype('float') / cm_lr.sum(axis=1)[:, np.newaxis]
print("------------------------------------------------------------------------------------------------------")
print("Logistic Regression Results:")
print("Accuracy:", accuracylr)
sns.heatmap(cm_lr , fmt = '.2f', annot = True , xticklabels=['Negative', 'Neutral', 'Positive'] , yticklabels=['Negative', 'Neutral', 'Positive']) 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Naive Bayes
gscv_nb = GridSearchCV(MultinomialNB() , param_grid= param_nb , cv = 2, scoring='accuracy')
gscv_nb.fit(X_trainML , y_trainML)

best_model_nb = gscv_nb.best_estimator_
best_model_nb.fit(X_trainML, y_trainML)

y_pred_nb = best_model_nb.predict(X_testML)
accuracynb = accuracy_score(y_testML, y_pred_nb)

cm_nb = confusion_matrix(y_testML, y_pred_nb) 
cm_nb = cm_nb.astype('float') / cm_nb.sum(axis=1)[:, np.newaxis]
print("------------------------------------------------------------------------------------------------------")
print("\nNaive Bayes Results:")
print("Accuracy:", accuracynb)
sns.heatmap(cm_nb, fmt='.2f' , annot = True , xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive']) 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Tokenize text data for RNN
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['Comments'])
sequences = tokenizer.texts_to_sequences(df['Comments'])

max_seq_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length)

# Split data for RNN
X_rnn = padded_sequences
y_rnn = to_categorical(df['labels'] + 1, num_classes=3)  # +1 to shift labels to 0, 1, 2

X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_rnn, y_rnn, test_size=0.2, random_state=42)

# Build RNN model
embedding_dim = 50
rnn_units = 64

model1 = Sequential()
model1.add(Embedding(max_words, embedding_dim, input_length=max_seq_length))
model1.add(SimpleRNN(rnn_units))
model1.add(Dense(3, activation='softmax'))

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model1.fit(X_train_rnn, y_train_rnn, epochs=10, batch_size=128, validation_data=(X_test_rnn, y_test_rnn), )

# Evaluate the model
losscnn, accuracyrnn = model1.evaluate(X_test_rnn, y_test_rnn)
rnn_predictions = model1.predict(X_test_rnn)
rnn_predictions = np.argmax(rnn_predictions, axis=1)
y_true_rnn = np.argmax(y_test_rnn, axis=1)

cm_rnn = confusion_matrix(y_true_rnn, rnn_predictions) 
cm_rnn = cm_rnn.astype('float') / cm_rnn.sum(axis=1)[:, np.newaxis]
print("------------------------------------------------------------------------------------------------------")
print("\n RNN results:")
print("Accuracy:", accuracyrnn)
sns.heatmap(cm_rnn , fmt= '.2f' , annot = True , xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive']) 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


model2 = Sequential()

model2.add(Embedding(max_words, output_dim=embedding_dim, input_length=max_seq_length))
model2.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model2.add(GlobalMaxPooling1D())

model2.add(Dense(rnn_units, activation='relu'))
model2.add(Dense(3, activation='softmax'))

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model2.fit(X_train_rnn, y_train_rnn, epochs=10, batch_size=32, validation_data=(X_test_rnn, y_test_rnn))

losscnn, accuracycnn = model2.evaluate(X_test_rnn, y_test_rnn)
cnn_predictions = model2.predict(X_test_rnn)
cnn_predictions = np.argmax(cnn_predictions, axis=1)

cm_cnn = confusion_matrix(y_true_rnn, cnn_predictions) 
cm_cnn = cm_cnn.astype('float') / cm_cnn.sum(axis=1)[:, np.newaxis]
print("------------------------------------------------------------------------------------------------------")
print("\nCNN results:")
print("Accuracy:", accuracycnn)
sns.heatmap(cm_cnn , fmt = '.2f', annot = True , xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive']) 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

ml_models = ['Logistic Regression', 'Naive Bayes']

plt.bar(ml_models, [accuracylr , accuracynb], color=['red', 'green'])
plt.xlabel('ML Models')
plt.ylabel('Accuracy')
plt.title('Comparison of ML Models')
plt.ylim(0.0, 1.0)
plt.show()

dl_models = ['RNN', 'CNN']

plt.bar(dl_models, [accuracyrnn , accuracycnn], color=['red', 'green'])
plt.xlabel('DL Models')
plt.ylabel('Accuracy')
plt.title('Comparison of DL Models')
plt.ylim(0.0, 1.0)
plt.show()


namemodels = ['Logistic Regression', 'Naive Bayes' , 'RNN' , 'CNN' ]

plt.bar(namemodels, [accuracylr , accuracynb , accuracyrnn , accuracycnn], color=['red', 'green' , 'blue' , 'orange'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of all Models')
plt.ylim(0.0, 1.0)
plt.show()
