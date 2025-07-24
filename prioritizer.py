import re
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os

class EmailPrioritizer:
    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.is_trained = False
        
        # Keywords for priority detection
        self.high_priority_keywords = [
            'urgent', 'asap', 'immediate', 'emergency', 'critical', 'deadline',
            'important', 'priority', 'rush', 'quickly', 'time sensitive',
            'action required', 'respond immediately', 'needs attention',
            'final notice', 'last chance', 'expires today', 'expires tomorrow'
        ]
        
        self.medium_priority_keywords = [
            'meeting', 'review', 'update', 'follow up', 'reminder', 'please',
            'request', 'feedback', 'discussion', 'scheduled', 'appointment',
            'proposal', 'report', 'presentation', 'conference', 'workshop'
        ]
        
        self.low_priority_keywords = [
            'newsletter', 'notification', 'announcement', 'fyi', 'info',
            'social', 'marketing', 'promotional', 'sale', 'offer',
            'subscription', 'unsubscribe', 'automated', 'no reply'
        ]
        
        # VIP senders (can be customized)
        self.vip_domains = [
            'ceo@', 'president@', 'director@', 'manager@', 'boss@',
            'hr@', 'legal@', 'finance@', 'admin@', 'support@'
        ]
        
        # Load pre-trained model if exists
        self.load_model()
    
    def extract_features(self, email):
        """Extract features from email for prioritization"""
        features = {}
        
        subject = email.get('subject', '').lower()
        body = email.get('body', '').lower()
        sender = email.get('sender', '').lower()
        text = f"{subject} {body}"
        
        # Keyword-based features
        features['high_priority_keywords'] = sum(1 for keyword in self.high_priority_keywords if keyword in text)
        features['medium_priority_keywords'] = sum(1 for keyword in self.medium_priority_keywords if keyword in text)
        features['low_priority_keywords'] = sum(1 for keyword in self.low_priority_keywords if keyword in text)
        
        # Sender-based features
        features['is_vip_sender'] = 1 if any(vip in sender for vip in self.vip_domains) else 0
        features['is_internal'] = 1 if '@company.com' in sender or '@organization.com' in sender else 0
        
        # Subject line features
        features['has_urgent_subject'] = 1 if any(word in subject for word in ['urgent', 'asap', 'important']) else 0
        features['subject_length'] = len(subject.split())
        features['has_question_mark'] = 1 if '?' in subject else 0
        features['has_exclamation'] = 1 if '!' in subject else 0
        features['all_caps_words'] = len([word for word in subject.split() if word.isupper() and len(word) > 2])
        
        # Content features
        features['body_length'] = len(body.split())
        features['has_deadline'] = 1 if any(word in text for word in ['deadline', 'due date', 'expires', 'by tomorrow', 'by today']) else 0
        features['has_meeting_request'] = 1 if any(word in text for word in ['meeting', 'call', 'conference', 'appointment']) else 0
        features['has_action_items'] = 1 if any(word in text for word in ['action required', 'please', 'need to', 'must', 'should']) else 0
        
        # Time-based features
        if 'date' in email and email['date']:
            try:
                if isinstance(email['date'], str):
                    email_date = datetime.fromisoformat(email['date'].replace('Z', '+00:00'))
                else:
                    email_date = email['date']
                
                now = datetime.now()
                hours_old = (now - email_date.replace(tzinfo=None)).total_seconds() / 3600
                features['hours_since_received'] = hours_old
                features['is_recent'] = 1 if hours_old < 2 else 0
            except:
                features['hours_since_received'] = 0
                features['is_recent'] = 0
        else:
            features['hours_since_received'] = 0
            features['is_recent'] = 0
        
        # Reply pattern features
        features['is_reply'] = 1 if subject.startswith('re:') or subject.startswith('reply:') else 0
        features['is_forward'] = 1 if subject.startswith('fwd:') or subject.startswith('forward:') else 0
        
        return features
    
    def rule_based_priority(self, email):
        """Simple rule-based prioritization"""
        subject = email.get('subject', '').lower()
        body = email.get('body', '').lower()
        sender = email.get('sender', '').lower()
        text = f"{subject} {body}"
        
        priority_score = 0
        
        # High priority indicators
        if any(keyword in text for keyword in self.high_priority_keywords):
            priority_score += 3
        
        if any(vip in sender for vip in self.vip_domains):
            priority_score += 2
        
        if any(word in subject for word in ['urgent', 'asap', 'important', 'critical']):
            priority_score += 2
        
        if '!' in subject or subject.isupper():
            priority_score += 1
        
        # Medium priority indicators
        if any(keyword in text for keyword in self.medium_priority_keywords):
            priority_score += 1
        
        if any(word in text for word in ['meeting', 'deadline', 'review', 'update']):
            priority_score += 1
        
        # Low priority indicators (reduce score)
        if any(keyword in text for keyword in self.low_priority_keywords):
            priority_score -= 1
        
        if 'unsubscribe' in text or 'newsletter' in text:
            priority_score -= 2
        
        # Convert score to priority level
        if priority_score >= 4:
            return 'high'
        elif priority_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def generate_training_data(self):
        """Generate training data for the ML model"""
        training_data = []
        
        # High priority examples
        high_priority_emails = [
            {
                'subject': 'URGENT: Server down - immediate action required',
                'body': 'The main server is down and all services are affected. We need immediate action to resolve this critical issue.',
                'sender': 'admin@company.com',
                'date': datetime.now(),
                'priority': 'high'
            },
            {
                'subject': 'Important: Board meeting moved to tomorrow',
                'body': 'The board meeting has been rescheduled to tomorrow at 9 AM. Please confirm your attendance ASAP.',
                'sender': 'ceo@company.com',
                'date': datetime.now(),
                'priority': 'high'
            },
            {
                'subject': 'Critical: Security breach detected',
                'body': 'We have detected a potential security breach. All users must change their passwords immediately.',
                'sender': 'security@company.com',
                'date': datetime.now(),
                'priority': 'high'
            },
            {
                'subject': 'Deadline reminder: Project due today',
                'body': 'This is a final reminder that the project deadline is today. Please submit your work immediately.',
                'sender': 'manager@company.com',
                'date': datetime.now(),
                'priority': 'high'
            }
        ]
        
        # Medium priority examples
        medium_priority_emails = [
            {
                'subject': 'Meeting request: Weekly team sync',
                'body': 'Hi, can we schedule our weekly team sync for Thursday? Please let me know your availability.',
                'sender': 'colleague@company.com',
                'date': datetime.now(),
                'priority': 'medium'
            },
            {
                'subject': 'Please review: Quarterly report draft',
                'body': 'I have prepared the quarterly report draft. Could you please review it and provide feedback?',
                'sender': 'analyst@company.com',
                'date': datetime.now(),
                'priority': 'medium'
            },
            {
                'subject': 'Update needed: Project status',
                'body': 'Hi, could you provide an update on the current project status? We need this for the monthly report.',
                'sender': 'coordinator@company.com',
                'date': datetime.now(),
                'priority': 'medium'
            },
            {
                'subject': 'Reminder: Training session next week',
                'body': 'Just a reminder that we have a training session scheduled for next week. Please confirm your attendance.',
                'sender': 'hr@company.com',
                'date': datetime.now(),
                'priority': 'medium'
            }
        ]
        
        # Low priority examples
        low_priority_emails = [
            {
                'subject': 'Newsletter: Monthly company updates',
                'body': 'Here are the latest updates from our company including new hires, achievements, and upcoming events.',
                'sender': 'newsletter@company.com',
                'date': datetime.now(),
                'priority': 'low'
            },
            {
                'subject': 'Social event: Office happy hour',
                'body': 'Join us for our monthly office happy hour next Friday. Food and drinks will be provided.',
                'sender': 'social@company.com',
                'date': datetime.now(),
                'priority': 'low'
            },
            {
                'subject': 'FYI: New parking regulations',
                'body': 'Please note the new parking regulations that will take effect next month. See attached document for details.',
                'sender': 'facilities@company.com',
                'date': datetime.now(),
                'priority': 'low'
            },
            {
                'subject': 'Promotional: 50% off office supplies',
                'body': 'Limited time offer on office supplies. Use code SAVE50 to get 50% off your next order.',
                'sender': 'suppliers@office.com',
                'date': datetime.now(),
                'priority': 'low'
            }
        ]
        
        # Combine all training data
        all_emails = high_priority_emails + medium_priority_emails + low_priority_emails
        
        for email in all_emails:
            features = self.extract_features(email)
            features['priority'] = email['priority']
            training_data.append(features)
        
        return pd.DataFrame(training_data)
    
    def train_model(self):
        """Train the ML-based prioritization model"""
        print("Training email prioritization model...")
        
        # Generate training data
        df = self.generate_training_data()
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col != 'priority']
        X = df[feature_columns]
        y = df['priority']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        print(f"Prioritization Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        self.feature_columns = feature_columns
        
        # Save model
        self.save_model()
        
        return accuracy
    
    def prioritize_email(self, email):
        """Prioritize a single email"""
        # First try rule-based approach
        rule_based_priority = self.rule_based_priority(email)
        
        # If ML model is trained, use it for more accurate prediction
        if self.is_trained and self.model is not None:
            try:
                features = self.extract_features(email)
                feature_vector = pd.DataFrame([features])
                
                # Ensure we have the same features as training
                for col in self.feature_columns:
                    if col not in feature_vector.columns:
                        feature_vector[col] = 0
                
                feature_vector = feature_vector[self.feature_columns]
                ml_priority = self.model.predict(feature_vector)[0]
                
                # Get confidence
                probabilities = self.model.predict_proba(feature_vector)[0]
                max_prob = max(probabilities)
                
                # If confidence is high, use ML prediction, otherwise use rule-based
                if max_prob > 0.6:
                    return ml_priority
                else:
                    return rule_based_priority
            except Exception as e:
                print(f"Error in ML prioritization: {e}")
                return rule_based_priority
        
        return rule_based_priority
    
    def prioritize_batch(self, emails):
        """Prioritize multiple emails"""
        priorities = []
        for email in emails:
            priority = self.prioritize_email(email)
            priorities.append(priority)
        return priorities
    
    def get_priority_score(self, email):
        """Get numerical priority score (0-100)"""
        priority = self.prioritize_email(email)
        
        score_mapping = {
            'high': 90,
            'medium': 60,
            'low': 30
        }
        
        base_score = score_mapping.get(priority, 30)
        
        # Add additional scoring factors
        features = self.extract_features(email)
        
        # Boost score for VIP senders
        if features.get('is_vip_sender', 0):
            base_score += 10
        
        # Boost for recent emails
        if features.get('is_recent', 0):
            base_score += 5
        
        # Boost for action items
        if features.get('has_action_items', 0):
            base_score += 5
        
        return min(100, max(0, base_score))
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.is_trained and self.model is not None:
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            return sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return []
    
    def save_model(self):
        """Save the trained model"""
        if self.model is not None:
            os.makedirs('models', exist_ok=True)
            joblib.dump({
                'model': self.model,
                'feature_columns': self.feature_columns
            }, 'models/email_prioritizer.pkl')
            print("Prioritization model saved successfully!")
    
    def load_model(self):
        """Load pre-trained model"""
        model_path = 'models/email_prioritizer.pkl'
        if os.path.exists(model_path):
            try:
                saved_data = joblib.load(model_path)
                self.model = saved_data['model']
                self.feature_columns = saved_data['feature_columns']
                self.is_trained = True
                print("Pre-trained prioritization model loaded successfully!")
            except Exception as e:
                print(f"Error loading prioritization model: {e}")
                self.is_trained = False
        else:
            print("No pre-trained prioritization model found.")
    
    def add_vip_sender(self, sender_pattern):
        """Add a VIP sender pattern"""
        if sender_pattern not in self.vip_domains:
            self.vip_domains.append(sender_pattern)
    
    def remove_vip_sender(self, sender_pattern):
        """Remove a VIP sender pattern"""
        if sender_pattern in self.vip_domains:
            self.vip_domains.remove(sender_pattern)
    
    def get_priority_explanation(self, email):
        """Get explanation for why an email was given a certain priority"""
        features = self.extract_features(email)
        priority = self.prioritize_email(email)
        
        explanation = []
        
        if features.get('high_priority_keywords', 0) > 0:
            explanation.append("Contains high-priority keywords")
        
        if features.get('is_vip_sender', 0):
            explanation.append("From VIP sender")
        
        if features.get('has_urgent_subject', 0):
            explanation.append("Urgent subject line")
        
        if features.get('has_deadline', 0):
            explanation.append("Contains deadline information")
        
        if features.get('is_recent', 0):
            explanation.append("Recently received")
        
        if features.get('has_action_items', 0):
            explanation.append("Contains action items")
        
        if not explanation:
            explanation.append("Standard priority based on content analysis")
        
        return {
            'priority': priority,
            'reasons': explanation,
            'score': self.get_priority_score(email)
        }

# Example usage
if __name__ == "__main__":
    prioritizer = EmailPrioritizer()
    
    # Test emails
    test_emails = [
        {
            'subject': 'URGENT: Server maintenance tonight',
            'body': 'We need to perform urgent server maintenance tonight. Please prepare for downtime.',
            'sender': 'admin@company.com',
            'date': datetime.now()
        },
        {
            'subject': 'Weekend plans',
            'body': 'Hey, want to grab coffee this weekend?',
            'sender': 'friend@gmail.com',
            'date': datetime.now()
        }
    ]
    
    for email in test_emails:
        priority = prioritizer.prioritize_email(email)
        explanation = prioritizer.get_priority_explanation(email)
        print(f"Subject: {email['subject']}")
        print(f"Priority: {priority}")
        print(f"Explanation: {explanation}")
        print("-" * 50)