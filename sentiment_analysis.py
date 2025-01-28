import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import string
import re
from bs4 import BeautifulSoup

nltk.download('stopwords')

# Function to clean reviews
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub(r'\d+', '', text)
    stop_words = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI setup
st.title("Dynamic Sentiment Analysis App")
st.sidebar.header("Upload Training Dataset")
training_file = st.sidebar.file_uploader("Upload a CSV file for training (must contain 'content' and 'score')", type="csv")

if training_file:
    st.write("## Training Dataset Preview")
    training_df = pd.read_csv(training_file)
    st.write(training_df.head())

    # Check for required columns
    if 'content' in training_df.columns and 'score' in training_df.columns:
        # Clean and preprocess the data
        training_df = training_df.dropna(subset=['content'])
        training_df['cleaned_review'] = training_df['content'].apply(clean_text)
        training_df['sentiment'] = training_df['score'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else 2))

        # Vectorize the text data
        tfidf = TfidfVectorizer(max_features=5000)
        X = tfidf.fit_transform(training_df['cleaned_review'])
        y = training_df['sentiment']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train models
        model = LogisticRegression()
        model.fit(X_train, y_train)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        xgb_model.fit(X_train, y_train)

        st.sidebar.header("Select Model")
        model_choice = st.sidebar.selectbox("Choose a model for prediction:", ("Logistic Regression", "Random Forest", "XGBoost"))

        # Upload input reviews for analysis
        st.sidebar.header("Analyze New Reviews")
        input_file = st.sidebar.file_uploader("Upload a CSV file with reviews (must have 'review' column)", type="csv", key="new_reviews")

        if input_file:
            new_reviews_df = pd.read_csv(input_file, names=['review'])
            new_reviews_df['cleaned_review'] = new_reviews_df['review'].apply(clean_text)
            reviews_vectorized = tfidf.transform(new_reviews_df['cleaned_review'])

            # Predict sentiment based on the chosen model
            if model_choice == 'Logistic Regression':
                predictions = model.predict(reviews_vectorized)
            elif model_choice == 'Random Forest':
                predictions = rf_model.predict(reviews_vectorized)
            elif model_choice == 'XGBoost':
                predictions = xgb_model.predict(reviews_vectorized)

            sentiment_labels = {1: 'Positive', 0: 'Negative', 2: 'Neutral'}
            new_reviews_df['Sentiment'] = [sentiment_labels[pred] for pred in predictions]

            st.write("## Sentiment Analysis Results")
            st.write(new_reviews_df[['review', 'Sentiment']])

            # Provide option to download the results
            st.download_button(
                label="Download Analysis Results",
                data=new_reviews_df.to_csv(index=False),
                file_name="output_analysis.csv",
                mime="text/csv"
            )

            # Overall sentiment
            sentiment_counts = new_reviews_df['Sentiment'].value_counts()
            total_reviews = len(new_reviews_df)
            overall_sentiment = (
                "Positive" if sentiment_counts.get('Positive', 0) > total_reviews / 2
                else "Negative" if sentiment_counts.get('Negative', 0) > total_reviews / 2
                else "Neutral"
            )
            st.markdown(f"### **Overall Analysis: {overall_sentiment} Sentiment**")

            # Areas for improvement
            if sentiment_counts.get('Negative', 0) > 0 or sentiment_counts.get('Neutral', 0) > 0:
                st.markdown("### Areas for Improvement:")
                st.write("- Consider addressing frequent concerns raised in negative reviews.")
                st.write("- Explore neutral reviews for insights that could enhance user experience.")
    else:
        st.error("The training dataset must contain 'content' and 'score' columns.")
else:
    st.write("### Please upload a training dataset to begin.")
