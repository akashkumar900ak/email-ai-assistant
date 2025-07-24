import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class EmailClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.pipeline = None
        self.is_trained = False
        self.lemmatizer = WordNetLemmatizer()
        
        # Load pre-trained model if exists
        self.load_model()
    
    def preprocess_text(self, text):
        """Clean and preprocess email text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def generate_training_data(self):
        """Generate sample training data for the classifier"""
        training_data = [
            # Work emails
            ("urgent meeting tomorrow board presentation", "work"),
            ("project deadline status update required", "work"),
            ("quarterly report financial results", "work"),
            ("team meeting conference room booking", "work"),
            ("client proposal contract negotiation", "work"),
            ("budget approval expense report", "work"),
            ("performance review scheduled next week", "work"),
            ("business travel authorization request", "work"),
            ("vendor invoice payment processing", "work"),
            ("strategic planning session attendance", "work"),
            ("salary increase promotion discussion", "work"),
            ("office relocation new address", "work"),
            ("training workshop registration required", "work"),
            ("company policy update announcement", "work"),
            ("product launch marketing campaign", "work"),
            ("sales target quarterly goals", "work"),
            ("hr department employee handbook", "work"),
            ("it support system maintenance", "work"),
            ("legal compliance regulatory requirements", "work"),
            ("customer feedback service improvement", "work"),
            
            # Personal emails
            ("birthday party invitation weekend", "personal"),
            ("family dinner restaurant reservation", "personal"),
            ("vacation photos trip memories", "personal"),
            ("doctor appointment health checkup", "personal"),
            ("grocery shopping list weekend", "personal"),
            ("movie tickets date night", "personal"),
            ("gym membership fitness goals", "personal"),
            ("book recommendation reading list", "personal"),
            ("recipe sharing cooking tips", "personal"),
            ("pet photos cute animals", "personal"),
            ("weather update weekend plans", "personal"),
            ("friend catching up coffee", "personal"),
            ("hobby project progress update", "personal"),
            ("home improvement renovation plans", "personal"),
            ("school reunion planning committee", "personal"),
            ("volunteer work charity event", "personal"),
            ("sports game tickets invitation", "personal"),
            ("concert tickets music festival", "personal"),
            ("gardening tips plant care", "personal"),
            ("travel planning vacation itinerary", "personal"),
            
            # Spam emails
            ("congratulations million dollar lottery winner", "spam"),
            ("urgent bank account verification required", "spam"),
            ("limited time offer discount sale", "spam"),
            ("click here claim prize money", "spam"),
            ("nigerian prince inheritance money", "spam"),
            ("viagra cialis cheap medication", "spam"),
            ("weight loss miracle solution", "spam"),
            ("make money fast home business", "spam"),
            ("credit card debt consolidation", "spam"),
            ("free trial subscription offer", "spam"),
            ("casino gambling jackpot winner", "spam"),
            ("bitcoin investment opportunity guaranteed", "spam"),
            ("dating site lonely singles", "spam"),
            ("insurance quote comparison service", "spam"),
            ("tech support computer virus", "spam"),
            ("charity donation tax deductible", "spam"),
            ("investment opportunity high returns", "spam"),
            ("prescription drugs online pharmacy", "spam"),
            ("loan approval instant cash", "spam"),
            ("sweepstakes entry form winner", "spam"),
            
            # Promotional emails
            ("flash sale discount coupon code", "promotional"),
            ("newsletter subscription latest updates", "promotional"),
            ("product announcement new features", "promotional"),
            ("seasonal sale winter collection", "promotional"),
            ("membership rewards points earned", "promotional"),
            ("webinar invitation free registration", "promotional"),
            ("survey feedback customer opinion", "promotional"),
            ("event invitation conference registration", "promotional"),
            ("catalog preview upcoming products", "promotional"),
            ("loyalty program benefits exclusive", "promotional"),
            ("app update new version available", "promotional"),
            ("social media follow connect", "promotional"),
            ("contest entry win prizes", "promotional"),
            ("subscription renewal account expires", "promotional"),
            ("referral program bonus rewards", "promotional"),
            ("demo request product trial", "promotional"),
            ("white paper download research", "promotional"),
            ("case study success stories", "promotional"),
            ("partnership opportunity collaboration", "promotional"),
            ("community forum discussion group", "promotional")
        ]
        
        return training_data
    
    def train_model(self):
        """Train the email classification model"""
        print("Training email classifier...")
        
        # Generate training data
        training_data = self.generate_training_data()
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data, columns=['text', 'category'])
        
        # Preprocess text
        df['text_clean'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text_clean'], df['category'], 
            test_size=0.2, random_state=42, stratify=df['category']
        )
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        # Train model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Classification Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return accuracy
    
    def classify_email(self, text):
        """Classify a single email"""
        if not self.is_trained and self.pipeline is None:
            # Train model if not already trained
            self.train_model()
        
        if isinstance(text, str) and text.strip():
            # Preprocess text
            cleaned_text = self.preprocess_text(text)
            
            if cleaned_text:
                # Make prediction
                prediction = self.pipeline.predict([cleaned_text])[0]
                confidence = self.pipeline.predict_proba([cleaned_text]).max()
                
                # If confidence is too low, classify as 'personal' by default
                if confidence < 0.3:
                    return 'personal'
                
                return prediction
        
        return 'personal'  # Default classification
    
    def classify_batch(self, texts):
        """Classify multiple emails at once"""
        if not self.is_trained and self.pipeline is None:
            self.train_model()
        
        predictions = []
        for text in texts:
            prediction = self.classify_email(text)
            predictions.append(prediction)
        
        return predictions
    
    def get_classification_confidence(self, text):
        """Get classification confidence scores"""
        if not self.is_trained and self.pipeline is None:
            self.train_model()
        
        cleaned_text = self.preprocess_text(text)
        if cleaned_text:
            probabilities = self.pipeline.predict_proba([cleaned_text])[0]
            classes = self.pipeline.classes_
            
            confidence_dict = dict(zip(classes, probabilities))
            return confidence_dict
        
        return {}
    
    def save_model(self):
        """Save the trained model to disk"""
        if self.pipeline is not None:
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.pipeline, 'models/email_classifier.pkl')
            print("Model saved successfully!")
    
    def load_model(self):
        """Load a pre-trained model from disk"""
        model_path = 'models/email_classifier.pkl'
        if os.path.exists(model_path):
            try:
                self.pipeline = joblib.load(model_path)
                self.is_trained = True
                print("Pre-trained model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.is_trained = False
        else:
            print("No pre-trained model found. Will train on first use.")
    
    def retrain_with_feedback(self, text, correct_label):
        """Retrain model with user feedback (incremental learning simulation)"""
        # This is a simplified version - in production, you'd implement proper incremental learning
        print(f"Received feedback: '{text[:50]}...' should be classified as '{correct_label}'")
        
        # For demonstration, we'll just add this to our training data and retrain
        # In production, you'd want a more sophisticated approach
        
        # Add feedback to training data and retrain
        self.train_model()
    
    def get_feature_importance(self, category=None, top_n=10):
        """Get most important features for classification"""
        if not self.is_trained:
            return {}
        
        # Get feature names
        feature_names = self.pipeline.named_steps['vectorizer'].get_feature_names_out()
        
        # Get coefficients
        coefficients = self.pipeline.named_steps['classifier'].coef_
        
        if category:
            # Get class index
            classes = self.pipeline.classes_
            if category in classes:
                class_idx = list(classes).index(category)
                coef = coefficients[class_idx]
                
                # Get top features
                top_indices = np.argsort(np.abs(coef))[-top_n:][::-1]
                top_features = [(feature_names[i], coef[i]) for i in top_indices]
                
                return top_features
        
        return {}

# Example usage
if __name__ == "__main__":
    classifier = EmailClassifier()
    
    # Test classification
    test_emails = [
        "urgent meeting tomorrow with the board",
        "happy birthday party invitation for weekend",
        "congratulations you won million dollars click here",
        "sale offer 50% discount on all items"
    ]
    
    for email in test_emails:
        category = classifier.classify_email(email)
        confidence = classifier.get_classification_confidence(email)
        print(f"Email: {email}")
        print(f"Category: {category}")
        print(f"Confidence: {confidence}")
        print("-" * 50)