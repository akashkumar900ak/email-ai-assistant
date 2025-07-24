import re
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch
import random
from datetime import datetime
import os

class ReplyGenerator:
    def __init__(self, model_name="gpt2"):
        """Initialize the reply generator with a language model"""
        self.model_name = model_name
        self.generator = None
        self.tokenizer = None
        self.model = None
        
        # Initialize the model
        self.load_model()
        
        # Reply templates for different scenarios
        self.templates = {
            'meeting': [
                "Thank you for reaching out. I'm available for the meeting. Please let me know the agenda details.",
                "I can attend the meeting. Could you please share the meeting agenda and dial-in details?",
                "Thank you for the meeting invite. I'll be there. Please send the meeting materials in advance."
            ],
            'urgent': [
                "Thank you for bringing this to my attention. I understand the urgency and will prioritize this matter.",
                "I acknowledge the urgent nature of this request. I'm working on it now and will update you shortly.",
                "Thank you for flagging this as urgent. I'm addressing it immediately and will get back to you soon."
            ],
            'request': [
                "Thank you for your request. I'll review this and get back to you by [timeframe].",
                "I've received your request and will work on it. I'll provide an update within [timeframe].",
                "Thank you for reaching out. I'll look into this and respond with details soon."
            ],
            'information': [
                "Thank you for the information. I'll review it and let you know if I have any questions.",
                "I appreciate you sharing this information. I'll go through it and follow up if needed.",
                "Thanks for keeping me informed. I'll review the details and reach out if I need clarification."
            ],
            'general': [
                "Thank you for your email. I'll review this and get back to you soon.",
                "I've received your message and will respond appropriately. Thank you for reaching out.",
                "Thank you for contacting me. I'll look into this matter and provide a response."
            ]
        }
        
        # Context-aware response patterns
        self.response_patterns = {
            'question': "Thank you for your question. ",
            'request': "I understand your request. ",
            'complaint': "I appreciate you bringing this to my attention. ",
            'compliment': "Thank you for your kind words. ",
            'urgent': "I understand this is urgent. ",
            'meeting': "Regarding the meeting, ",
            'deadline': "I note the deadline mentioned. ",
            'follow_up': "Thank you for following up. "
        }
    
    def load_model(self):
        """Load the language model for text generation"""
        try:
            print("Loading text generation model...")
            # Use a smaller, faster model for better performance
            self.generator = pipeline(
                'text-generation',
                model='distilgpt2',  # Smaller and faster than GPT-2
                tokenizer='distilgpt2',
                device=-1,  # Use CPU
                pad_token_id=50256
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to template-based responses")
            self.generator = None
    
    def preprocess_email(self, email_content):
        """Extract key information from email content"""
        if not isinstance(email_content, str):
            return ""
        
        # Clean the email content
        content = email_content.lower().strip()
        
        # Remove common email artifacts
        content = re.sub(r'from:.*\n', '', content)
        content = re.sub(r'to:.*\n', '', content)
        content = re.sub(r'sent:.*\n', '', content)
        content = re.sub(r'subject:.*\n', '', content)
        
        # Remove excessive whitespace
        content = ' '.join(content.split())
        
        return content
    
    def detect_email_intent(self, email_content):
        """Detect the intent/type of the email"""
        content = email_content.lower()
        
        # Check for different patterns
        if any(word in content for word in ['meeting', 'schedule', 'appointment', 'call']):
            return 'meeting'
        elif any(word in content for word in ['urgent', 'asap', 'immediately', 'emergency']):
            return 'urgent'
        elif any(word in content for word in ['please', 'could you', 'can you', 'request']):
            return 'request'
        elif any(word in content for word in ['?', 'question', 'clarify', 'understand']):
            return 'question'
        elif any(word in content for word in ['complaint', 'issue', 'problem', 'concern']):
            return 'complaint'
        elif any(word in content for word in ['thank', 'appreciate', 'great job', 'excellent']):
            return 'compliment'
        elif any(word in content for word in ['deadline', 'due date', 'expires', 'by tomorrow']):
            return 'deadline'
        elif any(word in content for word in ['follow up', 'following up', 'check in']):
            return 'follow_up'
        elif any(word in content for word in ['fyi', 'for your information', 'letting you know']):
            return 'information'
        else:
            return 'general'
    
    def generate_template_reply(self, email_content, intent):
        """Generate reply using templates"""
        templates = self.templates.get(intent, self.templates['general'])
        
        # Select a random template
        template = random.choice(templates)
        
        # Add context-specific modifications
        if intent == 'meeting':
            if 'when' in email_content.lower() or 'time' in email_content.lower():
                template += " Please confirm the time that works best for you."
        elif intent == 'urgent':
            template += " I'll keep you updated on the progress."
        elif intent == 'request':
            # Try to extract timeframe
            if 'tomorrow' in email_content.lower():
                template = template.replace('[timeframe]', 'tomorrow')
            elif 'today' in email_content.lower():
                template = template.replace('[timeframe]', 'today')
            else:
                template = template.replace('[timeframe]', 'the next 24-48 hours')
        
        # Add appropriate closing
        closings = [
            "\n\nBest regards,",
            "\n\nThank you,",
            "\n\nKind regards,",
            "\n\nSincerely,"
        ]
        
        template += random.choice(closings)
        
        return template
    
    def generate_ai_reply(self, email_content):
        """Generate reply using AI model"""
        if self.generator is None:
            return None
        
        try:
            # Create a prompt for reply generation
            intent = self.detect_email_intent(email_content)
            starter = self.response_patterns.get(intent, "Thank you for your email. ")
            
            # Create prompt
            prompt = f"Email: {email_content[:200]}...\nReply: {starter}"
            
            # Generate response
            generated = self.generator(
                prompt,
                max_length=len(prompt.split()) + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256,
                eos_token_id=50256
            )
            
            # Extract the generated reply
            full_response = generated[0]['generated_text']
            reply_start = full_response.find("Reply: ") + 7
            reply = full_response[reply_start:].strip()
            
            # Clean up the reply
            reply = self.clean_generated_reply(reply)
            
            # Add professional closing if not present
            if not any(closing in reply.lower() for closing in ['regards', 'sincerely', 'thank you']):
                reply += "\n\nBest regards,"
            
            return reply
            
        except Exception as e:
            print(f"Error generating AI reply: {e}")
            return None
    
    def clean_generated_reply(self, reply):
        """Clean and improve the generated reply"""
        # Remove incomplete sentences at the end
        sentences = reply.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            reply = '.'.join(sentences[:-1]) + '.'
        
        # Remove repetitive phrases
        lines = reply.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and line.strip() not in unique_lines:
                unique_lines.append(line.strip())
        
        reply = '\n'.join(unique_lines)
        
        # Ensure proper capitalization
        reply = '. '.join(sentence.strip().capitalize() for sentence in reply.split('.') if sentence.strip())
        
        return reply
    
    def generate_reply(self, email_content, style='professional', length='medium'):
        """Main method to generate email reply"""
        if not email_content or not isinstance(email_content, str):
            return "Thank you for your email. I'll review this and get back to you soon.\n\nBest regards,"
        
        # Preprocess email
        processed_content = self.preprocess_email(email_content)
        
        # Detect intent
        intent = self.detect_email_intent(processed_content)
        
        # Try AI generation first, fall back to templates
        ai_reply = self.generate_ai_reply(processed_content)
        
        if ai_reply and len(ai_reply.strip()) > 20:
            reply = ai_reply
        else:
            reply = self.generate_template_reply(processed_content, intent)
        
        # Adjust reply based on style and length preferences
        reply = self.adjust_reply_style(reply, style, length)
        
        return reply
    
    def adjust_reply_style(self, reply, style, length):
        """Adjust the reply based on style and length preferences"""
        if style == 'casual':
            # Make it more casual
            reply = reply.replace('Thank you for your email', 'Thanks for reaching out')
            reply = reply.replace('Best regards,', 'Thanks,')
            reply = reply.replace('I understand', "I get it")
            reply = reply.replace('I will', "I'll")
        elif style == 'formal':
            # Make it more formal
            reply = reply.replace("I'll", 'I will')
            reply = reply.replace("Thanks", 'Thank you')
            reply = reply.replace("can't", 'cannot')
        
        if length == 'short':
            # Keep only essential parts
            sentences = reply.split('.')
            if len(sentences) > 2:
                reply = '. '.join(sentences[:2]) + '.'
        elif length == 'long':
            # Add more details
            if 'meeting' in reply.lower():
                reply = reply.replace('.', '. I look forward to our discussion.')
            elif 'request' in reply.lower():
                reply = reply.replace('.', '. I will ensure this receives proper attention.')
        
        return reply
    
    def generate_multiple_options(self, email_content, num_options=3):
        """Generate multiple reply options"""
        options = []
        
        for i in range(num_options):
            # Use different approaches for variety
            if i == 0:
                reply = self.generate_reply(email_content, style='professional', length='medium')
            elif i == 1:
                reply = self.generate_reply(email_content, style='friendly', length='short')
            else:
                reply = self.generate_reply(email_content, style='formal', length='long')
            
            options.append(reply)
        
        return options
    
    def get_reply_suggestions(self, email_content):
        """Get contextual suggestions for improving the reply"""
        intent = self.detect_email_intent(email_content)
        
        suggestions = {
            'meeting': [
                "Consider proposing specific times",
                "Ask about the agenda or meeting objectives",
                "Confirm the meeting location or dial-in details"
            ],
            'urgent': [
                "Acknowledge the urgency clearly",
                "Provide a specific timeline for response",
                "Offer interim updates if resolution takes time"
            ],
            'request': [
                "Clarify any requirements or constraints",
                "Provide a realistic timeline",
                "Ask for additional information if needed"
            ],
            'question': [
                "Address each question separately",
                "Provide clear, specific answers",
                "Offer to discuss further if needed"
            ]
        }
        
        return suggestions.get(intent, ["Keep the response clear and professional"])
    
    def analyze_email_sentiment(self, email_content):
        """Basic sentiment analysis of the email"""
        positive_words = ['thank', 'appreciate', 'great', 'excellent', 'wonderful', 'pleased']
        negative_words = ['problem', 'issue', 'complaint', 'disappointed', 'frustrated', 'urgent']
        
        content = email_content.lower()
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

# Example usage
if __name__ == "__main__":
    generator = ReplyGenerator()
    
    test_emails = [
        "Hi, I need to schedule a meeting with you for next week. Are you available on Tuesday?",
        "URGENT: The server is down and we need immediate action. Please help!",
        "Thank you for the great presentation yesterday. The team was very impressed.",
        "Could you please send me the quarterly report by tomorrow? I need it for the board meeting."
    ]
    
    for email in test_emails:
        print(f"Original Email: {email}")
        reply = generator.generate_reply(email)
        print(f"Generated Reply: {reply}")
        print("-" * 80)