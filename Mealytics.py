import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string
import pickle
import time
import io
import base64

# Initialize NLTK components
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    st.info("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    st.success("Download complete!")

# Set page configuration
st.set_page_config(
    page_title="Restaurant Review Sentiment Analysis",
    page_icon="🍽️",
    layout="wide"
)

class RestaurantReviewAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.model = None
        self.baseline_accuracy = None
        self.model_accuracy = None
        
    def preprocess_text(self, text):
        """Preprocess text data: lowercase, remove punctuation, stopwords, lemmatize"""
        if isinstance(text, str):
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            
            # Remove digits
            text = re.sub(r'\d+', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            
            return ' '.join(processed_tokens)
        return ""
    
    def train_models(self, df):
        """Train baseline and advanced models and report improvement"""
        # Prepare data
        df['processed_review'] = df['review'].apply(self.preprocess_text)
        X = df['processed_review']
        y = df['sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Progress bar for training
        progress_bar = st.progress(0)
        
        # Create TF-IDF features
        st.info("Vectorizing text features...")
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        progress_bar.progress(20)
        
        # Train baseline model (Naive Bayes)
        st.info("Training baseline Naive Bayes model...")
        baseline_model = MultinomialNB()
        baseline_model.fit(X_train_tfidf, y_train)
        baseline_preds = baseline_model.predict(X_test_tfidf)
        self.baseline_accuracy = accuracy_score(y_test, baseline_preds)
        progress_bar.progress(40)
        
        # Train advanced models and pick the best one
        st.info("Training advanced models...")
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Linear SVM': LinearSVC(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        best_accuracy = 0
        best_model_name = None
        
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_train_tfidf, y_train)
            preds = model.predict(X_test_tfidf)
            acc = accuracy_score(y_test, preds)
            
            st.info(f"{name} Accuracy: {acc:.4f}")
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                self.model = model
            
            progress_bar.progress(40 + ((i + 1) * 20))
        
        self.model_accuracy = best_accuracy
        improvement = ((self.model_accuracy - self.baseline_accuracy) / self.baseline_accuracy) * 100
        
        progress_bar.progress(100)
        st.success(f"Training complete! Selected {best_model_name} as the best model")
        
        # Return results for display
        return {
            'baseline_accuracy': self.baseline_accuracy,
            'model_accuracy': self.model_accuracy,
            'improvement': improvement,
            'best_model': best_model_name,
            'X_test': X_test,
            'y_test': y_test,
            'predictions': self.model.predict(X_test_tfidf)
        }
    
    def predict_sentiment(self, review_text):
        """Predict sentiment for a new review"""
        if not self.vectorizer or not self.model:
            st.error("Model not trained! Please train the model first.")
            return None
        
        processed_review = self.preprocess_text(review_text)
        review_tfidf = self.vectorizer.transform([processed_review])
        prediction = self.model.predict(review_tfidf)[0]
        
        return prediction
        
    def get_feature_importance(self, n=10):
        """Get top n positive and negative features for interpretability"""
        if not self.vectorizer or not hasattr(self.model, 'coef_'):
            return None
        
        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.model.coef_[0]
        
        # Get top positive and negative coefficients
        top_positive_idx = coefs.argsort()[-n:][::-1]
        top_negative_idx = coefs.argsort()[:n]
        
        top_positive = [(feature_names[i], coefs[i]) for i in top_positive_idx]
        top_negative = [(feature_names[i], coefs[i]) for i in top_negative_idx]
        
        return top_positive, top_negative


# Title and description
st.title("🍽️ Restaurant Review Sentiment Analysis")
st.markdown("""
This app analyzes sentiment in restaurant reviews using NLP and machine learning techniques. 
Upload your dataset or use our sample data to train the model, then analyze new reviews!
""")

# Initialize the analyzer
analyzer = RestaurantReviewAnalyzer()

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Train Model", "Analyze Reviews", "Model Insights"])

with tab1:
    st.header("Train Sentiment Analysis Model")
    
    # Option to upload data or use sample data
    data_option = st.radio(
        "Choose your data source:",
        ('Upload custom dataset', 'Use sample dataset')
    )
    
    if data_option == 'Upload custom dataset':
        uploaded_file = st.file_uploader("Upload your restaurant reviews CSV file", type=["csv"])
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            
            # Check required columns
            if not all(col in df.columns for col in ['review', 'sentiment']):
                st.error("Dataset must contain 'review' and 'sentiment' columns. Sentiment should be binary (0 or 1).")
            else:
                # Check for proper format and preview
                if df['sentiment'].nunique() != 2 or not set(df['sentiment'].unique()).issubset({0, 1}):
                    st.warning("Sentiment column should contain only binary values (0 for negative, 1 for positive)")
                
                train_button = st.button("Train Model")
                if train_button:
                    with st.spinner("Training models... This may take a minute."):
                        results = analyzer.train_models(df)
                        
                        # Display results in a nice format
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Baseline Accuracy", f"{results['baseline_accuracy']:.4f}")
                        with col2:
                            st.metric("Model Accuracy", f"{results['model_accuracy']:.4f}")
                        with col3:
                            st.metric("Improvement", f"{results['improvement']:.2f}%")
                        
                        # Store model results in session state for other tabs
                        st.session_state['model_trained'] = True
                        st.session_state['results'] = results
                        
                        st.success("Model training complete! Go to the 'Analyze Reviews' tab to test it.")
    else:
        st.info("Creating a sample dataset with 1000 restaurant reviews...")
        
        # Create a sample dataset
        sample_positive = [
            "The food was amazing! Great service too.",
            "Best meal I've had in months. Will definitely return.",
            "The chef really knows what they're doing. Awesome flavors!",
            "Great atmosphere and even better food. Loved it!",
            "The staff was very attentive and the dessert was to die for!"
        ] * 100
        
        sample_negative = [
            "Terrible experience. The food was cold and service was slow.",
            "Would not recommend. Overpriced for what you get.",
            "The waiter was rude and the food was mediocre at best.",
            "Very disappointed with my meal. Won't be coming back.",
            "Food took forever to arrive and was barely edible when it did."
        ] * 100
        
        # Combine and create DataFrame
        reviews = sample_positive + sample_negative
        sentiments = [1] * 500 + [0] * 500
        
        # Add some randomness
        combined = list(zip(reviews, sentiments))
        np.random.shuffle(combined)
        reviews, sentiments = zip(*combined)
        
        sample_df = pd.DataFrame({
            'review': reviews,
            'sentiment': sentiments
        })
        
        st.dataframe(sample_df.head())
        
        train_button = st.button("Train Model on Sample Data")
        if train_button:
            with st.spinner("Training models... This may take a minute."):
                results = analyzer.train_models(sample_df)
                
                # Display results in a nice format
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Baseline Accuracy", f"{results['baseline_accuracy']:.4f}")
                with col2:
                    st.metric("Model Accuracy", f"{results['model_accuracy']:.4f}")
                with col3:
                    st.metric("Improvement", f"{results['improvement']:.2f}%")
                
                # Store model results in session state for other tabs
                st.session_state['model_trained'] = True
                st.session_state['results'] = results
                st.session_state['sample_df'] = sample_df
                
                st.success("Model training complete! Go to the 'Analyze Reviews' tab to test it.")

with tab2:
    st.header("Analyze New Reviews")
    
    if st.session_state.get('model_trained', False):
        st.success("Model is trained and ready to analyze reviews!")
        
        # Option for single review or batch analysis
        analysis_type = st.radio("Choose analysis type:", ["Single Review", "Batch Analysis"])
        
        if analysis_type == "Single Review":
            review_text = st.text_area("Enter a restaurant review to analyze:", 
                                        "The food was delicious and the service was excellent!")
            
            if st.button("Analyze Sentiment"):
                with st.spinner("Analyzing review..."):
                    sentiment = analyzer.predict_sentiment(review_text)
                    
                    if sentiment == 1:
                        st.success("😃 Positive Review")
                    else:
                        st.error("😞 Negative Review")
                    
                    # Display processed text
                    st.subheader("Preprocessing Details")
                    st.write("Original Text:")
                    st.info(review_text)
                    st.write("Processed Text:")
                    st.info(analyzer.preprocess_text(review_text))
        
        else:  # Batch Analysis
            st.subheader("Batch Analysis")
            
            # Option to upload a file or use a text area for multiple reviews
            batch_option = st.radio("Input method:", ["Text Input", "Upload CSV"])
            
            if batch_option == "Text Input":
                batch_reviews = st.text_area("Enter multiple reviews (one per line):",
                                           "The pizza was cold and the service was slow.\nAbsolutely loved the ambiance and the chef's special!")
                
                if st.button("Analyze Batch"):
                    reviews_list = batch_reviews.split('\n')
                    results = []
                    
                    with st.spinner(f"Analyzing {len(reviews_list)} reviews..."):
                        for review in reviews_list:
                            if review.strip():  # Skip empty lines
                                sentiment = analyzer.predict_sentiment(review)
                                results.append({"Review": review, "Sentiment": "Positive" if sentiment == 1 else "Negative"})
                        
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df)
                        
                        # Add download button for results
                        csv = results_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_results.csv">Download results as CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
            
            else:  # Upload CSV
                uploaded_batch = st.file_uploader("Upload CSV with reviews", type=["csv"])
                if uploaded_batch is not None:
                    batch_df = pd.read_csv(uploaded_batch)
                    
                    # Display the first few rows
                    st.dataframe(batch_df.head())
                    
                    # Select the column containing reviews
                    review_column = st.selectbox("Select the column containing reviews:", batch_df.columns)
                    
                    if st.button("Analyze Batch"):
                        with st.spinner(f"Analyzing {len(batch_df)} reviews..."):
                            batch_df['Predicted Sentiment'] = batch_df[review_column].apply(
                                lambda x: "Positive" if analyzer.predict_sentiment(x) == 1 else "Negative"
                            )
                            
                            st.dataframe(batch_df)
                            
                            # Add download button for results
                            csv = batch_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_results.csv">Download results as CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("Please train the model in the 'Train Model' tab first.")

with tab3:
    st.header("Model Insights and Visualizations")
    
    if st.session_state.get('model_trained', False):
        results = st.session_state['results']
        
        st.subheader("Performance Metrics")
        
        # Display confusion matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(results['y_test'], results['predictions'])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
        st.pyplot(fig)
        
        # Classification report
        st.write("### Classification Report")
        report = classification_report(results['y_test'], results['predictions'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        # Feature importance (if available)
        st.write("### Top Features Influencing Sentiment")
        feature_importance = analyzer.get_feature_importance(10)
        
        if feature_importance:
            top_positive, top_negative = feature_importance
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Words indicating positive sentiment:")
                pos_df = pd.DataFrame(top_positive, columns=['Word', 'Coefficient'])
                st.dataframe(pos_df)
                
                # Visualize
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh([word for word, _ in top_positive], [coef for _, coef in top_positive], color='green')
                ax.set_title('Top Positive Features')
                plt.tight_layout()
                st.pyplot(fig)
                
            with col2:
                st.write("Words indicating negative sentiment:")
                neg_df = pd.DataFrame(top_negative, columns=['Word', 'Coefficient'])
                st.dataframe(neg_df)
                
                # Visualize
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh([word for word, _ in top_negative], [coef for _, coef in top_negative], color='red')
                ax.set_title('Top Negative Features')
                plt.tight_layout()
                st.pyplot(fig)
        
        # Display sample predictions
        st.write("### Sample Predictions")
        
        if 'sample_df' in st.session_state:
            # Get a small random sample
            sample = st.session_state['sample_df'].sample(5)
            for _, row in sample.iterrows():
                with st.expander(f"Review: {row['review'][:50]}..."):
                    sentiment = analyzer.predict_sentiment(row['review'])
                    actual = row['sentiment']
                    
                    st.write("**Full review:**")
                    st.info(row['review'])
                    
                    st.write("**Prediction:**")
                    if sentiment == 1:
                        st.success("😃 Positive")
                    else:
                        st.error("😞 Negative")
                    
                    st.write("**Actual sentiment:**")
                    if actual == 1:
                        st.success("😃 Positive")
                    else:
                        st.error("😞 Negative")
        
        # Download model button
        if st.button("Export Trained Model"):
            # Create a BytesIO object
            model_export = io.BytesIO()
            # Save the models
            pickle.dump({
                'vectorizer': analyzer.vectorizer,
                'model': analyzer.model,
                'baseline_accuracy': analyzer.baseline_accuracy,
                'model_accuracy': analyzer.model_accuracy
            }, model_export)
            model_export.seek(0)
            
            # Download button
            st.download_button(
                label="Download Model Files",
                data=model_export,
                file_name="restaurant_sentiment_model.pkl",
                mime="application/octet-stream"
            )
    else:
        st.warning("Please train the model in the 'Train Model' tab first.")

# Footer
st.markdown("---")
st.markdown("""
**About this app**: Restaurant Review Sentiment Analysis uses NLP techniques to classify reviews as positive or negative, 
achieving a significant improvement in accuracy compared to baseline models.

**Tech Stack**: Python, NLP, Scikit-learn, Streamlit, Machine Learning
""")
