import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
import ssl
from datetime import datetime
import re
import base64
import quopri

class EmailClient:
    def __init__(self, email_address, password, imap_server="imap.gmail.com", smtp_server="smtp.gmail.com"):
        """Initialize email client with credentials"""
        self.email_address = email_address
        self.password = password
        self.imap_server = imap_server
        self.smtp_server = smtp_server
        self.imap_port = 993
        self.smtp_port = 587
        
        # Test connection on initialization
        self.test_connection()
    
    def test_connection(self):
        """Test IMAP and SMTP connections"""
        try:
            # Test IMAP connection
            context = ssl.create_default_context()
            with imaplib.IMAP4_SSL(self.imap_server, self.imap_port, ssl_context=context) as imap:
                imap.login(self.email_address, self.password)
                print("✅ IMAP connection successful")
            
            # Test SMTP connection
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as smtp:
                smtp.starttls()
                smtp.login(self.email_address, self.password)
                print("✅ SMTP connection successful")
                
        except Exception as e:
            raise Exception(f"Connection failed: {str(e)}")
    
    def decode_mime_words(self, s):
        """Decode MIME encoded words in email headers"""
        if not s:
            return ""
        
        decoded_parts = []
        for part, encoding in decode_header(s):
            if isinstance(part, bytes):
                try:
                    if encoding:
                        decoded_parts.append(part.decode(encoding))
                    else:
                        decoded_parts.append(part.decode('utf-8', errors='ignore'))
                except:
                    decoded_parts.append(part.decode('utf-8', errors='ignore'))
            else:
                decoded_parts.append(str(part))
        
        return ''.join(decoded_parts)
    
    def extract_email_body(self, msg):
        """Extract the body content from an email message"""
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                # Skip attachments
                if "attachment" in content_disposition:
                    continue
                
                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            body = payload.decode(charset, errors='ignore')
                            break
                    except:
                        continue
                elif content_type == "text/html" and not body:
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            html_body = payload.decode(charset, errors='ignore')
                            # Simple HTML to text conversion
                            body = self.html_to_text(html_body)
                    except:
                        continue
        else:
            content_type = msg.get_content_type()
            if content_type == "text/plain":
                try:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        charset = msg.get_content_charset() or 'utf-8'
                        body = payload.decode(charset, errors='ignore')
                except:
                    body = str(msg.get_payload())
            elif content_type == "text/html":
                try:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        charset = msg.get_content_charset() or 'utf-8'
                        html_body = payload.decode(charset, errors='ignore')
                        body = self.html_to_text(html_body)
                except:
                    body = str(msg.get_payload())
        
        return body.strip()
    
    def html_to_text(self, html):
        """Simple HTML to text conversion"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        
        # Decode HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text
    
    def fetch_emails(self, folder="INBOX", limit=10, unread_only=False):
        """Fetch emails from the specified folder"""
        emails = []
        
        try:
            context = ssl.create_default_context()
            with imaplib.IMAP4_SSL(self.imap_server, self.imap_port, ssl_context=context) as imap:
                imap.login(self.email_address, self.password)
                imap.select(folder)
                
                # Search criteria
                if unread_only:
                    search_criteria = "UNSEEN"
                else:
                    search_criteria = "ALL"
                
                status, message_ids = imap.search(None, search_criteria)
                
                if status == "OK":
                    message_ids = message_ids[0].split()
                    
                    # Get the most recent emails (reverse order)
                    recent_ids = message_ids[-limit:] if len(message_ids) > limit else message_ids
                    recent_ids.reverse()  # Most recent first
                    
                    for i, msg_id in enumerate(recent_ids):
                        try:
                            status, msg_data = imap.fetch(msg_id, "(RFC822)")
                            
                            if status == "OK":
                                email_body = msg_data[0][1]
                                email_message = email.message_from_bytes(email_body)
                                
                                # Extract email information
                                subject = self.decode_mime_words(email_message["Subject"]) or "No Subject"
                                sender = self.decode_mime_words(email_message["From"]) or "Unknown Sender"
                                date_str = email_message["Date"]
                                
                                # Parse date
                                try:
                                    date_obj = email.utils.parsedate_to_datetime(date_str)
                                except:
                                    date_obj = datetime.now()
                                
                                # Extract body
                                body = self.extract_email_body(email_message)
                                
                                # Check if email is read
                                status, flag_data = imap.fetch(msg_id, "(FLAGS)")
                                is_read = b'\\Seen' in flag_data[0] if flag_data else True
                                
                                email_data = {
                                    'id': len(emails) + 1,
                                    'message_id': msg_id.decode(),
                                    'subject': subject,
                                    'sender': sender,
                                    'date': date_obj,
                                    'body': body[:1000] + "..." if len(body) > 1000 else body,  # Truncate long emails
                                    'is_read': is_read,
                                    'folder': folder
                                }
                                
                                emails.append(email_data)
                                
                        except Exception as e:
                            print(f"Error processing email {msg_id}: {str(e)}")
                            continue
                
        except Exception as e:
            raise Exception(f"Error fetching emails: {str(e)}")
        
        return emails
    
    def send_email(self, to_email, subject, body, reply_to_message_id=None):
        """Send an email"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_address
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add reply-to header if this is a reply
            if reply_to_message_id:
                msg['In-Reply-To'] = reply_to_message_id
                msg['References'] = reply_to_message_id
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_address, self.password)
                text = msg.as_string()
                server.sendmail(self.email_address, to_email, text)
            
            print(f"✅ Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            print(f"❌ Error sending email: {str(e)}")
            return False
    
    def send_reply(self, original_email, reply_body):
        """Send a reply to an email"""
        try:
            # Extract sender email from the original email
            sender = original_email['sender']
            
            # Extract email address from sender string (remove name if present)
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', sender)
            if email_match:
                sender_email = email_match.group()
            else:
                sender_email = sender
            
            # Create reply subject
            original_subject = original_email['subject']
            if not original_subject.lower().startswith('re:'):
                reply_subject = f"Re: {original_subject}"
            else:
                reply_subject = original_subject
            
            # Send the reply
            success = self.send_email(
                to_email=sender_email,
                subject=reply_subject,
                body=reply_body,
                reply_to_message_id=original_email.get('message_id')
            )
            
            return success
            
        except Exception as e:
            print(f"❌ Error sending reply: {str(e)}")
            return False
    
    def mark_as_read(self, message_id, folder="INBOX"):
        """Mark an email as read"""
        try:
            context = ssl.create_default_context()
            with imaplib.IMAP4_SSL(self.imap_server, self.imap_port, ssl_context=context) as imap:
                imap.login(self.email_address, self.password)
                imap.select(folder)
                
                imap.store(message_id, '+FLAGS', '\\Seen')
                
            return True
            
        except Exception as e:
            print(f"Error marking email as read: {str(e)}")
            return False
    
    def mark_as_unread(self, message_id, folder="INBOX"):
        """Mark an email as unread"""
        try:
            context = ssl.create_default_context()
            with imaplib.IMAP4_SSL(self.imap_server, self.imap_port, ssl_context=context) as imap:
                imap.login(self.email_address, self.password)
                imap.select(folder)
                
                imap.store(message_id, '-FLAGS', '\\Seen')
                
            return True
            
        except Exception as e:
            print(f"Error marking email as unread: {str(e)}")
            return False
    
    def delete_email(self, message_id, folder="INBOX"):
        """Delete an email (move to trash)"""
        try:
            context = ssl.create_default_context()
            with imaplib.IMAP4_SSL(self.imap_server, self.imap_port, ssl_context=context) as imap:
                imap.login(self.email_address, self.password)
                imap.select(folder)
                
                imap.store(message_id, '+FLAGS', '\\Deleted')
                imap.expunge()
                
            return True
            
        except Exception as e:
            print(f"Error deleting email: {str(e)}")
            return False
    
    def get_folders(self):
        """Get list of available folders"""
        try:
            context = ssl.create_default_context()
            with imaplib.IMAP4_SSL(self.imap_server, self.imap_port, ssl_context=context) as imap:
                imap.login(self.email_address, self.password)
                
                status, folders = imap.list()
                
                if status == "OK":
                    folder_list = []
                    for folder in folders:
                        folder_info = folder.decode().split('"')
                        if len(folder_info) >= 3:
                            folder_name = folder_info[-2]
                            folder_list.append(folder_name)
                    
                    return folder_list
                
        except Exception as e:
            print(f"Error getting folders: {str(e)}")
            return ["INBOX"]
    
    def search_emails(self, query, folder="INBOX"):
        """Search emails by subject or sender"""
        try:
            context = ssl.create_default_context()
            with imaplib.IMAP4_SSL(self.imap_server, self.imap_port, ssl_context=context) as imap:
                imap.login(self.email_address, self.password)
                imap.select(folder)
                
                # Search in subject and from fields
                search_criteria = f'(OR SUBJECT "{query}" FROM "{query}")'
                status, message_ids = imap.search(None, search_criteria)
                
                if status == "OK" and message_ids[0]:
                    message_ids = message_ids[0].split()
                    emails = []
                    
                    for msg_id in message_ids[-10:]:  # Limit to 10 results
                        try:
                            status, msg_data = imap.fetch(msg_id, "(RFC822)")
                            
                            if status == "OK":
                                email_body = msg_data[0][1]
                                email_message = email.message_from_bytes(email_body)
                                
                                subject = self.decode_mime_words(email_message["Subject"]) or "No Subject"
                                sender = self.decode_mime_words(email_message["From"]) or "Unknown Sender"
                                date_str = email_message["Date"]
                                
                                try:
                                    date_obj = email.utils.parsedate_to_datetime(date_str)
                                except:
                                    date_obj = datetime.now()
                                
                                body = self.extract_email_body(email_message)
                                
                                email_data = {
                                    'id': len(emails) + 1,
                                    'message_id': msg_id.decode(),
                                    'subject': subject,
                                    'sender': sender,
                                    'date': date_obj,
                                    'body': body[:500] + "..." if len(body) > 500 else body,
                                    'folder': folder
                                }
                                
                                emails.append(email_data)
                                
                        except Exception as e:
                            continue
                    
                    return emails
                
        except Exception as e:
            print(f"Error searching emails: {str(e)}")
            
        return []
    
    def get_email_count(self, folder="INBOX", unread_only=False):
        """Get count of emails in folder"""
        try:
            context = ssl.create_default_context()
            with imaplib.IMAP4_SSL(self.imap_server, self.imap_port, ssl_context=context) as imap:
                imap.login(self.email_address, self.password)
                imap.select(folder)
                
                if unread_only:
                    status, message_ids = imap.search(None, "UNSEEN")
                else:
                    status, message_ids = imap.search(None, "ALL")
                
                if status == "OK":
                    if message_ids[0]:
                        return len(message_ids[0].split())
                    else:
                        return 0
                        
        except Exception as e:
            print(f"Error getting email count: {str(e)}")
            
        return 0

# Example usage and testing
if __name__ == "__main__":
    # This is for testing - replace with actual credentials
    print("EmailClient class ready for use!")
    print("Remember to use Gmail App Passwords, not regular passwords!")
    print("\nExample usage:")
    print("""
    client = EmailClient('your-email@gmail.com', 'your-app-password')
    emails = client.fetch_emails(limit=5)
    for email in emails:
        print(f"Subject: {email['subject']}")
        print(f"From: {email['sender']}")
        print(f"Date: {email['date']}")
        print("-" * 40)
    """)