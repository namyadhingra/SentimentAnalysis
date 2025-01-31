import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import string
import re

# Download NLTK stopwords
nltk.download('stopwords')

# Function to clean the reviews
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub(r'\d+', '', text)  # Remove numbers
    stop_words = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI setup
st.title("Sentiment Analysis App")
st.sidebar.header("Upload Training Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("## Dataset Preview")
        st.write(df.head())

        # Check if required columns exist
        if 'content' not in df.columns or 'score' not in df.columns:
            st.error("The dataset must contain 'content' and 'score' columns.")
        else:
            # Preprocess data
            df = df.dropna(subset=['content'])  # Drop rows with missing reviews
            df['cleaned_review'] = df['content'].apply(clean_text)

            # Map scores to sentiment labels
            df['sentiment'] = df['score'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else 2))

            # Vectorize the text data
            tfidf = TfidfVectorizer(max_features=5000)
            X = tfidf.fit_transform(df['cleaned_review'])
            y = df['sentiment']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train models
            with st.spinner("Training models..."):
                model = LogisticRegression()
                model.fit(X_train, y_train)

                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)

                xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
                xgb_model.fit(X_train, y_train)

            # Evaluate models
            st.sidebar.header("Model Evaluation")
            if st.sidebar.button("Evaluate Models"):
                st.write("### Logistic Regression Performance")
                y_pred_lr = model.predict(X_test)
                st.text(classification_report(y_test, y_pred_lr))
                st.write(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}")

                st.write("### Random Forest Performance")
                y_pred_rf = rf_model.predict(X_test)
                st.text(classification_report(y_test, y_pred_rf))
                st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")

                st.write("### XGBoost Performance")
                y_pred_xgb = xgb_model.predict(X_test)
                st.text(classification_report(y_test, y_pred_xgb))
                st.write(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.2f}")

            # Upload input reviews for analysis
            st.sidebar.header("Analyze New Reviews")
            input_file = st.sidebar.file_uploader("Upload a CSV file with reviews", type="csv", key="new_reviews")

            if input_file:
                new_reviews_df = pd.read_csv(input_file, names=['review'])
                new_reviews_df['cleaned_review'] = new_reviews_df['review'].apply(clean_text)
                reviews_vectorized = tfidf.transform(new_reviews_df['cleaned_review'])

                # Predict sentiment based on the chosen model
                st.sidebar.header("Select Model")
                model_choice = st.sidebar.selectbox("Choose a model for prediction:", ("Logistic Regression", "Random Forest", "XGBoost"))

                if model_choice == 'Logistic Regression':
                    predictions = model.predict(reviews_vectorized)
                elif model_choice == 'Random Forest':
                    predictions = rf_model.predict(reviews_vectorized)
                elif model_choice == 'XGBoost':
                    predictions = xgb_model.predict(reviews_vectorized)

                sentiment_labels = {1: 'Positive', 0: 'Negative', 2: 'Neutral'}
                new_reviews_df['Sentiment'] = [sentiment_labels[pred] for pred in predictions]

                # Calculate total positive and negative reviews
                total_positive = new_reviews_df[new_reviews_df['Sentiment'] == 'Positive'].shape[0]
                total_negative = new_reviews_df[new_reviews_df['Sentiment'] == 'Negative'].shape[0]
                total_neutral = new_reviews_df[new_reviews_df['Sentiment'] == 'Neutral'].shape[0]

                # Add summary row to the DataFrame
                summary_df = pd.DataFrame({
                    'review': ['Total Positive Reviews', 'Total Negative Reviews', 'Total Neutral Reviews'],
                    'Sentiment': [total_positive, total_negative, total_neutral]
                })

                # Combine the results with the summary
                final_df = pd.concat([new_reviews_df[['review', 'Sentiment']], summary_df], ignore_index=True)

                st.write("## Sentiment Analysis Results")
                st.write(final_df)

                # Provide option to download the results
                st.download_button(
                    label="Download Analysis Results",
                    data=final_df.to_csv(index=False),
                    file_name="output_analysis.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"An error occurred: {e}")