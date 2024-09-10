#Step 1: Setup - import libraries

#nltk = Natural Language toolkit for processing text
import nltk
#pandas = Data manipulation and analysis 
import pandas as pd
#CountVectorizer = Converts text to matrix
from sklearn.feature_extraction.text import CountVectorizer
#train_test_split = Splits data into training and testing sets
from sklearn.model_selection import train_test_split
#MultinomialNB = Naive Bayes classifier for multinomial models
from sklearn.naive_bayes import MultinomialNB
#accuracy_score, classification report = Model performance metrics
from sklearn.metrics import accuracy_score, classification_report
#movie_reviews = Movie review dataset used 
from nltk.corpus import movie_reviews

#Step 2: Data prep

#Download dataset 
nltk.download("movie_reviews")
#Load dataset into text string of review, sentiment category
documents = [
    (" ".join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]
#Converts list into DataFrame with Columns for reviews and sentiments
df = pd.DataFrame(documents, columns = ["review", "sentiment"])
#print for debug/check
#print(df)

#Step 3: Model Training

#Use CountVectorizer to convert the text reviews into numerical feature vectors, max_features = maximum most common words, X = matrix of these features, y = sentiment labels
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]
#Split data into training and testing sets, test_size = what % of data is used for testing, random_state = ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)
#Train a Naive Bayes classifier using training data
model = MultinomialNB()
model.fit(X_train, y_train)
#Run predictions on the test set, evaluate performance using accuracy and create classification report showing precision, recall, f1-score
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

#Step 4: Prediction

#Defines a function to predict the sentiment of a given text, transforming the text into the same feature vector format used in training, and using the trained model to predict the sentiment
def predict_sentiment(text):
    text_vector = vectorizer.transform({text})
    prediction = model.predict(text_vector)
    return prediction[0]
#Test prediction function
print(predict_sentiment("I think the movie was amazing, the entire family loved it."))
print(predict_sentiment("It was a disaster of a film, I don't understand how anyone could enjoy it."))
print(predict_sentiment("The movie was alright, easily forgettable but worth the watch."))
