import streamlit as st
import pandas as pd
import os
from datetime import datetime
import json
from classifier import EmailClassifier
from prioritizer import EmailPrioritizer  # ✅ fixed import
from reply_generator import ReplyGenerator  # ✅ fixed import
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
        st.subheader("📧 Email Setup")

        email_address = st.text_input("Email Address")
        email_password = st.text_input("App Password", type="password", help="Use Gmail App Password")

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

        st.subheader("🧪 Demo Mode")
        if st.button("Load Sample Emails"):
            st.session_state.emails = load_sample_emails()
            st.session_state.authenticated = True
            st.success("✅ Sample emails loaded!")

        st.subheader("🧠 Model Training")
        if st.button("Train Models"):
            with st.spinner("Training classification model..."):
                classifier.train_model()
            with st.spinner("Training prioritization model..."):
                prioritizer.train_model()
            st.success("✅ Models trained!")

    if st.session_state.authenticated:
        tab1, tab2, tab3 = st.tabs(["📨 Inbox", "📊 Analytics", "⚙️ Settings"])

        with tab1:
            display_inbox()
        with tab2:
            display_analytics()
        with tab3:
            display_settings()
    else:
        st.info("👆 Please connect your email or try demo mode using the sidebar")

def load_sample_emails():
    return [
        {
            'id': 1,
            'subject': 'Urgent: Project Deadline Tomorrow',
            'sender': 'boss@company.com',
            'body': 'We need to discuss the project status. The deadline is tomorrow. Please send updates.',
            'date': datetime.now(),
            'classification': 'work',
            'priority': 'high',
            'is_read': False
        },
        {
            'id': 2,
            'subject': 'Weekend Plans',
            'sender': 'friend@email.com',
            'body': 'Coffee this weekend? Let me know!',
            'date': datetime.now(),
            'classification': 'personal',
            'priority': 'low',
            'is_read': False
        }
    ]

def display_inbox():
    st.header("📨 Your Inbox")

    if hasattr(st.session_state, 'email_client'):
        if st.button("🔄 Fetch New Emails"):
            with st.spinner("Fetching emails..."):
                try:
                    new_emails = st.session_state.email_client.fetch_emails(limit=10)
                    for email in new_emails:
                        email['classification'] = classifier.classify_email(email['subject'] + ' ' + email['body'])
                        email['priority'] = prioritizer.prioritize_email(email)
                    st.session_state.emails = new_emails
                    st.success(f"✅ Fetched {len(new_emails)} emails")
                except Exception as e:
                    st.error(f"❌ Error fetching emails: {str(e)}")

    if st.session_state.emails:
        for email in st.session_state.emails:
            show_email_card(email)
    else:
        st.info("No emails to display. Fetch or load sample emails.")

def show_email_card(email):
    with st.expander(f"📧 {email['subject']}"):
        st.write(f"**From:** {email['sender']}")
        st.write(f"**Category:** {email['classification']} | **Priority:** {email['priority']}")
        st.write("**Content:**")
        st.write(email['body'])

        if st.button("✉️ Generate Reply", key=f"reply_{email['id']}"):
            reply = reply_gen.generate_reply(email['body'])
            st.text_area("Suggested Reply:", value=reply, height=150)

def display_analytics():
    st.header("📊 Email Analytics")
    if st.session_state.emails:
        df = pd.DataFrame(st.session_state.emails)
        st.bar_chart(df['classification'].value_counts())
        st.bar_chart(df['priority'].value_counts())
    else:
        st.info("No email data to analyze.")

def display_settings():
    st.header("⚙️ Settings")
    st.slider("Model Confidence Threshold", 0.0, 1.0, 0.8)

# Call the main function
if __name__ == "__main__":
    main()
