### Sentiment Analysis App

---

#### **Overview**
This project is a **Sentiment Analysis Web Application** built using **Streamlit**. The app allows users to upload a dataset containing reviews, preprocess the data, train multiple machine learning models, and analyze the sentiment of new reviews. The app supports three models for prediction: 
- Logistic Regression
- Random Forest
- XGBoost

---

#### **Features**
1. **Dataset Upload**:
   - Upload a CSV file containing reviews and their respective scores.
   - Displays the first few rows of the dataset for preview.

2. **Data Preprocessing**:
   - Cleans the text reviews by removing punctuation, HTML tags, numbers, and stopwords.
   - Maps review scores to sentiment labels:
     - Positive (1): Score >= 4
     - Neutral (2): Score = 3
     - Negative (0): Score <= 2

3. **Model Training**:
   - Supports Logistic Regression, Random Forest, and XGBoost models.
   - Automatically splits the dataset into training and testing sets (80%-20%).

4. **Sentiment Prediction**:
   - Upload a separate CSV file with reviews for analysis.
   - Predicts the sentiment of the reviews using the selected model.
   - Displays predictions with an option to download the results.

---

#### **Setup Instructions**
1. **Install Dependencies**:
   - Install the required Python libraries:
     ```bash
     pip install streamlit pandas scikit-learn xgboost nltk
     ```
   - Download NLTK stopwords:
     ```bash
     python -c "import nltk; nltk.download('stopwords')"
     ```

2. **Run the Application**:
   - Save the script in a `.py` file (e.g., `sentiment_analysis_app.py`).
   - Run the application using Streamlit:
     ```bash
     streamlit run sentiment_analysis_app.py
     ```

3. **Upload Dataset**:
   - The uploaded dataset should be a CSV file with at least two columns:
     - `content`: The text reviews.
     - `score`: Numerical ratings for each review.

4. **Analyze New Reviews**:
   - Upload a separate CSV file with a single column named `review` to analyze new text reviews.

---

#### **How It Works**
1. **Data Cleaning**:
   - Removes unnecessary elements like punctuation, HTML tags, and stopwords.
   - Uses the `clean_text` function to preprocess reviews.

2. **TF-IDF Vectorization**:
   - Transforms the text data into numerical features using **TfidfVectorizer**.

3. **Model Selection**:
   - Train Logistic Regression, Random Forest, and XGBoost models on the preprocessed data.
   - Allows users to choose a model via the sidebar.

4. **Prediction**:
   - Applies the selected model to the new reviews.
   - Outputs the sentiment predictions as Positive, Neutral, or Negative.

5. **Result Export**:
   - Provides an option to download the analysis results as a CSV file.

---

#### **File Format Requirements**
- **Training Dataset**:
  - Must include the columns `content` and `score`.
- **Input Reviews File**:
  - Must have a single column `review` containing text reviews.

---

#### **Dependencies**
- Python 3.7+
- Libraries:
  - Streamlit
  - Pandas
  - Scikit-learn
  - XGBoost
  - NLTK

---

#### **Future Enhancements**
- Add support for other machine learning models or deep learning techniques.
- Enable user-defined hyperparameter tuning.
- Visualize model performance metrics (e.g., confusion matrix, ROC curve).

---

#### **Usage Example**
1. Run the application:
   ```bash
   streamlit run sentiment_analysis_app.py
   ```
2. Upload a training dataset with `content` and `score` columns.
3. Select a machine learning model (Logistic Regression, Random Forest, or XGBoost).
4. Upload a file with new reviews for sentiment prediction.
5. View and download the sentiment analysis results.

---

#### **Contact**
If you have questions or encounter issues, feel free to reach out!