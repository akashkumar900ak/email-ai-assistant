import streamlit as st
import pandas as pd
import os
from datetime import datetime
import json
from classifier import EmailClassifier
from prioritize import EmailPrioritizer
from reply_gen import ReplyGenerator
from email_client import EmailClient
import time

# Page configuration
st.set_page_config(
    page_title="AI Email Assistant",
    page_icon="📧",
    layout="wide"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'emails' not in st.session_state:
    st.session_state.emails = []
if 'selected_email' not in st.session_state:
    st.session_state.selected_email = None

# Initialize components
@st.cache_resource
def load_components():
    classifier = EmailClassifier()
    prioritizer = EmailPrioritizer()
    reply_gen = ReplyGenerator()
    return classifier, prioritizer, reply_gen

classifier, prioritizer, reply_gen = load_components()

def main():
    st.title("🤖 AI Email Assistant")
    st.markdown("*Intelligent email classification, prioritization, and reply generation*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Email credentials section
        st.subheader("📧 Email Setup")
        email_address = st.text_input("Email Address", type="default")
        email_password = st.text_input("App Password", type="password", help="Use Gmail App Password, not regular password")
        
        if st.button("Connect to Email"):
            if email_address and email_password:
                try:
                    email_client = EmailClient(email_address, email_password)
                    st.session_state.email_client = email_client
                    st.session_state.authenticated = True
                    st.success("✅ Connected successfully!")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
            else:
                st.warning("Please enter both email and password")
        
        # Demo mode
        st.subheader("🧪 Demo Mode")
        if st.button("Load Sample Emails"):
            st.session_state.emails = load_sample_emails()
            st.session_state.authenticated = True
            st.success("✅ Sample emails loaded!")
        
        # Model training section
        st.subheader("🧠 Model Training")
        if st.button("Train Models"):
            with st.spinner("Training classification model..."):
                classifier.train_model()
            with st.spinner("Training prioritization model..."):
                prioritizer.train_model()
            st.success("✅ Models trained!")

    # Main content area
    if st.session_state.authenticated:
        tab1, tab2, tab3, tab4 = st.tabs(["📨 Inbox", "📊 Analytics", "⚙️ Settings", "🔧 Tools"])
        
        with tab1:
            display_inbox()
        
        with tab2:
            display_analytics()
        
        with tab3:
            display_settings()
            
        with tab4:
            display_tools()
    else:
        st.info("👆 Please connect your email or try demo mode using the sidebar")
        display_features()

def load_sample_emails():
    """Load sample emails for demonstration"""
    sample_emails = [
        {
            'id': 1,
            'subject': 'Urgent: Project Deadline Tomorrow',
            'sender': 'boss@company.com',
            'body': 'Hi, we need to discuss the project status. The deadline is tomorrow and I need an update on your progress. Please send me the latest version by end of day.',
            'date': datetime.now(),
            'classification': 'work',
            'priority': 'high',
            'is_read': False
        },
        {
            'id': 2,
            'subject': 'Weekend Plans',
            'sender': 'friend@email.com',
            'body': 'Hey! Want to grab coffee this weekend? I heard about this new place downtown. Let me know if you\'re free!',
            'date': datetime.now(),
            'classification': 'personal',
            'priority': 'low',
            'is_read': False
        },
        {
            'id': 3,
            'subject': 'Meeting Request - Q4 Planning',
            'sender': 'colleague@company.com',
            'body': 'Hi, I would like to schedule a meeting to discuss Q4 planning. Are you available next Tuesday at 2 PM? We need to align on the quarterly goals.',
            'date': datetime.now(),
            'classification': 'work',
            'priority': 'medium',
            'is_read': False
        },
        {
            'id': 4,
            'subject': 'Congratulations! You won $1000000',
            'sender': 'noreply@suspicious.com',
            'body': 'You have won one million dollars! Click here to claim your prize. Send your bank details immediately.',
            'date': datetime.now(),
            'classification': 'spam',
            'priority': 'low',
            'is_read': False
        }
    ]
    return sample_emails

def display_inbox():
    st.header("📨 Your Inbox")
    
    if hasattr(st.session_state, 'email_client'):
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Fetch New Emails"):
                with st.spinner("Fetching emails..."):
                    try:
                        new_emails = st.session_state.email_client.fetch_emails(limit=10)
                        # Process emails through AI models
                        for email in new_emails:
                            email['classification'] = classifier.classify_email(email['subject'] + ' ' + email['body'])
                            email['priority'] = prioritizer.prioritize_email(email)
                        st.session_state.emails = new_emails
                        st.success(f"✅ Fetched {len(new_emails)} emails")
                    except Exception as e:
                        st.error(f"❌ Error fetching emails: {str(e)}")
    
    if st.session_state.emails:
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            classification_filter = st.selectbox("Filter by Category", 
                                                ["All", "work", "personal", "spam", "promotional"])
        with col2:
            priority_filter = st.selectbox("Filter by Priority", 
                                         ["All", "high", "medium", "low"])
        with col3:
            show_unread = st.checkbox("Show unread only", value=False)
        
        # Apply filters
        filtered_emails = st.session_state.emails.copy()
        if classification_filter != "All":
            filtered_emails = [e for e in filtered_emails if e.get('classification') == classification_filter]
        if priority_filter != "All":
            filtered_emails = [e for e in filtered_emails if e.get('priority') == priority_filter]
        if show_unread:
            filtered_emails = [e for e in filtered_emails if not e.get('is_read', True)]
        
        # Display emails
        for email in filtered_emails:
            display_email_card(email)
    else:
        st.info("No emails to display. Fetch emails or load sample data.")

def display_email_card(email):
    """Display an individual email card"""
    priority_colors = {
        'high': '🔴',
        'medium': '🟡', 
        'low': '🟢'
    }
    
    category_icons = {
        'work': '💼',
        'personal': '👤',
        'spam': '🚫',
        'promotional': '📢'
    }
    
    with st.expander(f"{priority_colors.get(email.get('priority', 'low'), '⚪')} {category_icons.get(email.get('classification', 'personal'), '📧')} {email['subject'][:50]}...", 
                     expanded=False):
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**From:** {email['sender']}")
            st.write(f"**Date:** {email.get('date', 'Unknown')}")
            
        with col2:
            st.write(f"**Category:** {email.get('classification', 'Unknown')}")
            st.write(f"**Priority:** {email.get('priority', 'Unknown')}")
            
        with col3:
            if st.button(f"📝 Reply", key=f"reply_{email['id']}"):
                st.session_state.selected_email = email
                st.rerun()
        
        st.write("**Email Content:**")
        st.write(email['body'])
        
        # Show AI-generated reply if email is selected
        if st.session_state.selected_email and st.session_state.selected_email['id'] == email['id']:
            st.write("---")
            st.subheader("🤖 AI-Generated Reply")
            
            with st.spinner("Generating reply..."):
                suggested_reply = reply_gen.generate_reply(email['body'])
            
            edited_reply = st.text_area("Edit reply:", value=suggested_reply, height=150, key=f"reply_text_{email['id']}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📤 Send Reply", key=f"send_{email['id']}"):
                    try:
                        if hasattr(st.session_state, 'email_client'):
                            st.session_state.email_client.send_reply(email, edited_reply)
                            st.success("✅ Reply sent successfully!")
                        else:
                            st.success("✅ Reply would be sent (demo mode)")
                        st.session_state.selected_email = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error sending reply: {str(e)}")
            
            with col2:
                if st.button("❌ Cancel", key=f"cancel_{email['id']}"):
                    st.session_state.selected_email = None
                    st.rerun()

def display_analytics():
    st.header("📊 Email Analytics")
    
    if st.session_state.emails:
        # Create dataframe for analysis
        df = pd.DataFrame(st.session_state.emails)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Email Categories")
            if 'classification' in df.columns:
                category_counts = df['classification'].value_counts()
                st.bar_chart(category_counts)
            
            st.subheader("📊 Priority Distribution")
            if 'priority' in df.columns:
                priority_counts = df['priority'].value_counts()
                st.bar_chart(priority_counts)
        
        with col2:
            st.subheader("📋 Summary Statistics")
            total_emails = len(df)
            unread_emails = len(df[df.get('is_read', True) == False]) if 'is_read' in df.columns else 0
            
            st.metric("Total Emails", total_emails)
            st.metric("Unread Emails", unread_emails)
            
            if 'priority' in df.columns:
                high_priority = len(df[df['priority'] == 'high'])
                st.metric("High Priority", high_priority)
            
            st.subheader("🎯 Top Senders")
            if 'sender' in df.columns:
                top_senders = df['sender'].value_counts().head(5)
                st.dataframe(top_senders)
    else:
        st.info("No data available for analytics. Please load some emails first.")

def display_settings():
    st.header("⚙️ Settings")
    
    st.subheader("🤖 AI Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Classification Model**")
        model_accuracy = st.slider("Model Confidence Threshold", 0.0, 1.0, 0.8)
        retrain_classifier = st.button("🔄 Retrain Classifier")
        
        if retrain_classifier:
            with st.spinner("Retraining classifier..."):
                time.sleep(2)  # Simulate training
                st.success("✅ Classifier retrained!")
    
    with col2:
        st.write("**Reply Generation Settings**")
        reply_length = st.selectbox("Reply Length", ["Short", "Medium", "Long"])
        reply_tone = st.selectbox("Reply Tone", ["Professional", "Friendly", "Casual"])
        
        st.write("**Email Fetching**")
        fetch_limit = st.number_input("Emails to fetch", min_value=1, max_value=100, value=10)

def display_tools():
    st.header("🔧 Developer Tools")
    
    st.subheader("📤 Export Data")
    if st.session_state.emails:
        if st.button("📥 Download Email Data (JSON)"):
            json_data = json.dumps(st.session_state.emails, default=str, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"email_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    st.subheader("🧪 Test Models")
    test_email = st.text_area("Test Email Content", placeholder="Enter email content to test classification and prioritization...")
    
    if st.button("🔍 Test Classification") and test_email:
        classification = classifier.classify_email(test_email)
        st.write(f"**Predicted Category:** {classification}")
        
        # Create mock email for prioritization
        mock_email = {
            'subject': 'Test Email',
            'body': test_email,
            'sender': 'test@example.com'
        }
        priority = prioritizer.prioritize_email(mock_email)
        st.write(f"**Predicted Priority:** {priority}")
        
        reply = reply_gen.generate_reply(test_email)
        st.write(f"**Suggested Reply:** {reply}")

def display_features():
    st.header("🌟 Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📧 Email Management")
        st.write("• Automatic email fetching")
        st.write("• Smart categorization")
        st.write("• Priority scoring")
        st.write("• Gmail integration")
    
    with col2:
        st.subheader("🤖 AI-Powered")
        st.write("• Intelligent classification")
        st.write("• Priority detection")
        st.write("• Auto-reply generation")
        st.write("• Context understanding")
    
    with col3:
        st.subheader("📊 Analytics")
        st.write("• Email statistics")
        st.write("• Sender analysis")
        st.write("• Category insights")
        st.write("• Performance metrics")

if __name__ == "__main__":
    main()