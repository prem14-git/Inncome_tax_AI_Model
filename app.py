from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import secrets
import string
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime, timedelta
import hashlib
import os as _os



app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")




load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY :
    raise ValueError("Missing API keys. Please set PINECONE_API_KEY and OPENAI_API_KEY in your environment.")
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY





# PostgreSQL Setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("Missing DATABASE_URL. Please set it in your .env file")

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email Configuration - Development Mode (shows code in console)
# For production, uncomment and configure one of the SMTP options below

# Development Mode - No email sending, code shown in console
DEVELOPMENT_MODE = False

# Gmail SMTP Configuration (uncomment and configure for production)
# app.config['MAIL_SERVER'] = 'smtp.gmail.com'
# app.config['MAIL_PORT'] = 587
# app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USERNAME'] = 'your-gmail@gmail.com'
# app.config['MAIL_PASSWORD'] = 'your-app-password'  # Generate App Password in Gmail settings
# app.config['MAIL_DEFAULT_SENDER'] = 'your-gmail@gmail.com'

# Hostinger SMTP Configuration (alternative)
app.config['MAIL_SERVER'] = 'smtp.hostinger.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = 'verification@lelekart.com'
app.config['MAIL_PASSWORD'] = 'Chinta@1710657'
app.config['MAIL_DEFAULT_SENDER'] = 'verification@lelekart.com'

# Outlook SMTP Configuration (alternative)
# app.config['MAIL_SERVER'] = 'smtp-mail.outlook.com'
# app.config['MAIL_PORT'] = 587
# app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USERNAME'] = 'your-email@outlook.com'
# app.config['MAIL_PASSWORD'] = 'your-password'
# app.config['MAIL_DEFAULT_SENDER'] = 'your-email@outlook.com'

db = SQLAlchemy(app)
mail = Mail(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_verified = db.Column(db.Boolean, default=False, nullable=False)
    verification_code = db.Column(db.String(6), nullable=True)
    verification_expires = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<User {self.email}>'

# Chat Session Model
class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<ChatSession {self.title}>'

# Chat Message Model
class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ChatMessage {self.role}>'

# Shareable Chat Link Model
class ChatShare(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    token = db.Column(db.String(64), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<ChatShare {self.token}>'

# Create tables
with app.app_context():
    try:
        db.create_all()
        print("Successfully connected to PostgreSQL database")
    except Exception as e:
        print(f"Database connection error: {e}")
        print("Please check your DATABASE_URL in .env file")
        raise


# Auth helpers
def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("user_email"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper

def verification_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("user_email"):
            return redirect(url_for("login"))
        user = User.query.filter_by(email=session.get('user_email')).first()
        if not user:
            session.clear()
            return redirect(url_for("login"))
        if not user.is_verified:
            return redirect(url_for("verify_email"))
        return view_func(*args, **kwargs)
    return wrapper

# Email verification helpers
def generate_verification_code():
    """Generate a 6-digit verification code"""
    return ''.join(secrets.choice(string.digits) for _ in range(6))

def send_verification_email(email, name, verification_code):
    """Send verification email to user"""
    try:
        if DEVELOPMENT_MODE:
            # Development mode - just print the code to console
            print("=" * 60)
            print("🔐 VERIFICATION CODE (Development Mode)")
            print("=" * 60)
            print(f"Email: {email}")
            print(f"Name: {name}")
            print(f"Verification Code: {verification_code}")
            print("=" * 60)
            print("⚠️  In development mode - email not actually sent!")
            print("⚠️  Use the code above to verify your account.")
            print("=" * 60)
            return True
        
        # Production mode - send actual email
        print(f"Attempting to send email to: {email}")
        print(f"Using SMTP server: {app.config['MAIL_SERVER']}:{app.config['MAIL_PORT']}")
        print(f"Using sender: {app.config['MAIL_DEFAULT_SENDER']}")
        
        msg = Message(
            subject='Verify Your Email - Tax AI Assistant',
            recipients=[email],
            sender=app.config['MAIL_DEFAULT_SENDER']
        )
        msg.html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #667eea; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; }}
                .code {{ background: #667eea; color: white; font-size: 24px; font-weight: bold; padding: 15px; text-align: center; border-radius: 8px; margin: 20px 0; letter-spacing: 3px; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Tax AI Assistant</h1>
                    <p>Email Verification Required</p>
                </div>
                <div class="content">
                    <h2>Hello {name}!</h2>
                    <p>Thank you for signing up for Tax AI Assistant. To complete your registration, please verify your email address using the code below:</p>
                    
                    <div class="code">{verification_code}</div>
                    
                    <p><strong>Important:</strong></p>
                    <ul>
                        <li>This verification code will expire in <strong>1 minute</strong></li>
                        <li>Enter this code on the verification page to activate your account</li>
                        <li>If you didn't request this verification, please ignore this email</li>
                    </ul>
                    
                    <p>If you have any questions, feel free to contact our support team.</p>
                </div>
                <div class="footer">
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        print("Sending email...")
        mail.send(msg)
        print(f"Email sent successfully to {email}")
        return True
    except Exception as e:
        print(f"Error sending email to {email}: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

embeddings = download_hugging_face_embeddings()

index_name = "lawbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , system_prompt) , 
        ("human" , "{input}") , 
    ]
)

question_answer_chain = create_stuff_documents_chain(llm , prompt)
reg_chain = create_retrieval_chain(retriever , question_answer_chain)





@app.route('/')
@login_required
def index():
    # Check session user exists and is verified
    user = User.query.filter_by(email=session.get('user_email')).first()
    if not user:
        session.clear()
        return redirect(url_for('login'))
    if not user.is_verified:
        return redirect(url_for('verify_email'))
    
    # Get user's chat sessions that have at least one message
    chat_sessions = db.session.query(ChatSession).join(ChatMessage).filter(
        ChatSession.user_id == user.id
    ).distinct().order_by(ChatSession.updated_at.desc()).all()
    return render_template('chat.html', user_email=session.get('user_email'), chat_sessions=chat_sessions)



@app.route("/get", methods=["GET", "POST"])
@verification_required
def chat():
    msg = request.form["msg"]
    session_id = request.form.get("session_id", type=int)
    
    # Get user
    user = User.query.filter_by(email=session.get('user_email')).first()
    
    # Create new session if none provided
    if not session_id:
        chat_session = ChatSession(
            user_id=user.id,
            title=msg[:50] + "..." if len(msg) > 50 else msg
        )
        db.session.add(chat_session)
        db.session.commit()
        session_id = chat_session.id
    else:
        chat_session = ChatSession.query.get(session_id)
        if not chat_session or chat_session.user_id != user.id:
            return jsonify({"error": "Invalid session"}), 400
    
    # Save user message
    user_message = ChatMessage(
        session_id=session_id,
        role='user',
        content=msg
    )
    db.session.add(user_message)
    
    # Get AI response
    response = reg_chain.invoke({"input": msg})
    ai_response = response["answer"]
    
    # Save AI response
    ai_message = ChatMessage(
        session_id=session_id,
        role='assistant',
        content=ai_response
    )
    db.session.add(ai_message)
    
    # Update session title if it's the first message
    if len(chat_session.title) < 10:
        chat_session.title = msg[:50] + "..." if len(msg) > 50 else msg
        chat_session.updated_at = datetime.utcnow()
    
    db.session.commit()
    
    return jsonify({
        "response": ai_response,
        "session_id": session_id
    })


@app.route('/load_chat/<int:session_id>')
@login_required
def load_chat(session_id):
    user = User.query.filter_by(email=session.get('user_email')).first()
    chat_session = ChatSession.query.filter_by(id=session_id, user_id=user.id).first()
    
    if not chat_session:
        return jsonify({"error": "Chat session not found"}), 404
    
    messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp.asc()).all()
    
    return jsonify({
        "title": chat_session.title,
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            } for msg in messages
        ]
    })


@app.route('/new_chat', methods=['POST'])
@login_required
def new_chat():
    user = User.query.filter_by(email=session.get('user_email')).first()
    # Don't create a new session immediately, just return success
    # The session will be created when the first message is sent
    return jsonify({"success": True})


@app.route('/delete_chat/<int:session_id>', methods=['DELETE'])
@login_required
def delete_chat(session_id):
    user = User.query.filter_by(email=session.get('user_email')).first()
    chat_session = ChatSession.query.filter_by(id=session_id, user_id=user.id).first()
    
    if not chat_session:
        return jsonify({"error": "Chat session not found"}), 404
    
    # Delete all messages in the session
    ChatMessage.query.filter_by(session_id=session_id).delete()
    db.session.delete(chat_session)
    db.session.commit()
    
    return jsonify({"success": True})


# Share endpoints
def _generate_share_token() -> str:
    random_bytes = _os.urandom(32)
    return hashlib.sha256(random_bytes).hexdigest()


@app.route('/share/<int:session_id>', methods=['POST'])
@login_required
def create_share_link(session_id):
    user = User.query.filter_by(email=session.get('user_email')).first()
    if not user:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    chat_session = ChatSession.query.filter_by(id=session_id, user_id=user.id).first()
    if not chat_session:
        return jsonify({"success": False, "message": "Chat session not found"}), 404

    # Check if share already exists; if not, create one
    existing = ChatShare.query.filter_by(session_id=session_id, user_id=user.id).first()
    if existing:
        token = existing.token
    else:
        token = _generate_share_token()
        share = ChatShare(session_id=session_id, user_id=user.id, token=token)
        db.session.add(share)
        db.session.commit()

    share_url = url_for('view_share', token=token, _external=True)
    return jsonify({"success": True, "url": share_url})


@app.route('/s/<token>')
@login_required
def view_share(token):
    # login-required view, but anonymize owner
    share = ChatShare.query.filter_by(token=token).first()
    if not share:
        flash('Shared chat not found.', 'danger')
        return redirect(url_for('index'))

    # Fetch messages for the shared session
    messages = ChatMessage.query.filter_by(session_id=share.session_id).order_by(ChatMessage.timestamp.asc()).all()
    anon_messages = [
        {
            'role': m.role,
            'content': m.content,
            'timestamp': m.timestamp
        } for m in messages
    ]
    # Render dedicated read-only shared chat view
    return render_template('shared_chat.html', messages=anon_messages, owner='Anonymous')


# Auth routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            if not name or not email or not password:
                flash('All fields are required.', 'danger')
                return render_template('signup.html', user_email=email, user_name=name)
            if len(password) < 6:
                flash('Password must be at least 6 characters.', 'danger')
                return render_template('signup.html', user_email=email, user_name=name)
            
            # Check if user already exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                if existing_user.is_verified:
                    flash('Email already registered. Please login.', 'warning')
                    return redirect(url_for('login'))
                else:
                    # User exists but not verified, show verify button
                    session['user_email'] = email
                    session['user_name'] = name
                    flash('Account already exists but not verified. Click "Verify Email" to complete registration.', 'warning')
                    return render_template('signup.html', show_verify_button=True, user_email=email, user_name=name)
            
            # Create new user
            password_hash = generate_password_hash(password)
            verification_code = generate_verification_code()
            new_user = User(
                name=name,
                email=email,
                password_hash=password_hash,
                verification_code=verification_code,
                verification_expires=datetime.utcnow() + timedelta(minutes=1)
            )
            db.session.add(new_user)
            db.session.commit()
            
            # Store user in session but don't redirect yet
            session['user_email'] = email
            session['user_name'] = name
            session['verification_code'] = verification_code
            flash('Account created successfully! Click "Verify Email" to complete registration.', 'success')
            return render_template('signup.html', show_verify_button=True, user_email=email, user_name=name)
        except Exception as e:
            print(f"Signup error: {e}")
            db.session.rollback()
            flash('Database error. Please try again.', 'danger')
            return render_template('signup.html', user_email=request.form.get('email',''), user_name=request.form.get('name',''))
    # GET request: do not pre-populate fields
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash('Invalid credentials.', 'danger')
            return render_template('login.html')
        session['user_email'] = email
        session['user_name'] = user.name
        
        # Check if user is verified
        if not user.is_verified:
            return redirect(url_for('verify_email'))
        
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/verify_email', methods=['GET', 'POST'])
def verify_email():
    if not session.get('user_email'):
        return redirect(url_for('login'))
    
    user = User.query.filter_by(email=session.get('user_email')).first()
    if not user:
        session.clear()
        return redirect(url_for('login'))
    
    if user.is_verified:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        verification_code = request.form.get('verification_code', '').strip()
        if not verification_code:
            flash('Please enter the verification code.', 'danger')
            return render_template('verify_email.html', user_email=user.email, user_name=user.name)
        
        # Check if code matches and is not expired
        if (user.verification_code == verification_code and 
            user.verification_expires and 
            datetime.utcnow() < user.verification_expires):
            
            # Mark user as verified
            user.is_verified = True
            user.verification_code = None
            user.verification_expires = None
            db.session.commit()
            
            flash('Email verified successfully! Welcome to Tax AI Assistant.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid or expired verification code. Please try again.', 'danger')
            return render_template('verify_email.html', user_email=user.email, user_name=user.name)
    
    return render_template('verify_email.html', user_email=user.email, user_name=user.name)


@app.route('/send_verification', methods=['POST'])
def send_verification():
    """Send verification email from signup page"""
    if not session.get('user_email'):
        return jsonify({'success': False, 'message': 'No user session found'})
    
    email = session.get('user_email')
    name = session.get('user_name')
    
    # Get or create user
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'success': False, 'message': 'User not found'})
    
    if user.is_verified:
        return jsonify({'success': False, 'message': 'Email already verified'})
    
    # Generate new verification code
    verification_code = generate_verification_code()
    user.verification_code = verification_code
    user.verification_expires = datetime.utcnow() + timedelta(minutes=1)
    db.session.commit()
    
    # Send verification email
    if send_verification_email(email, name, verification_code):
        return jsonify({'success': True, 'message': 'Verification code sent to your email'})
    else:
        return jsonify({'success': False, 'message': 'Failed to send verification email'})

@app.route('/resend_verification', methods=['POST'])
def resend_verification():
    if not session.get('user_email'):
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    user = User.query.filter_by(email=session.get('user_email')).first()
    if not user:
        return jsonify({'success': False, 'message': 'User not found'})
    
    if user.is_verified:
        return jsonify({'success': False, 'message': 'Email already verified'})
    
    # Generate new verification code
    verification_code = generate_verification_code()
    user.verification_code = verification_code
    user.verification_expires = datetime.utcnow() + timedelta(minutes=1)
    db.session.commit()
    
    # Send verification email
    if send_verification_email(user.email, user.name, verification_code):
        return jsonify({'success': True, 'message': 'Verification code sent to your email'})
    else:
        return jsonify({'success': False, 'message': 'Failed to send verification email'})


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/test_email')
def test_email():
    """Test email configuration"""
    try:
        test_email_address = "premkulkarni1407@gmail.com"
        test_code = "123456"
        test_name = "Test User"
        
        print("Testing email configuration...")
        result = send_verification_email(test_email_address, test_name, test_code)
        
        if result:
            return jsonify({
                'success': True, 
                'message': f'Test email sent successfully to {test_email_address}'
            })
        else:
            return jsonify({
                'success': False, 
                'message': 'Failed to send test email. Check console logs for details.'
            })
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error testing email: {str(e)}'
        })


@app.route('/smtp_diagnostics')
def smtp_diagnostics():
    """Diagnose SMTP connectivity for Hostinger on ports 587 (TLS) and 465 (SSL)."""
    import socket, ssl, time
    results = {}
    server = 'smtp.hostinger.com'
    tests = [
        {'port': 587, 'mode': 'TLS', 'use_ssl': False},
        {'port': 465, 'mode': 'SSL', 'use_ssl': True},
    ]
    for t in tests:
        start = time.time()
        try:
            if t['use_ssl']:
                ctx = ssl.create_default_context()
                with socket.create_connection((server, t['port']), timeout=10) as sock:
                    with ctx.wrap_socket(sock, server_hostname=server) as ssock:
                        banner = ssock.recv(1024).decode(errors='ignore')
                        results[f"{server}:{t['port']} ({t['mode']})"] = {
                            'connected': True,
                            'banner': banner.strip(),
                            'elapsed_ms': int((time.time() - start) * 1000)
                        }
            else:
                with socket.create_connection((server, t['port']), timeout=10) as sock:
                    banner = sock.recv(1024).decode(errors='ignore')
                    results[f"{server}:{t['port']} ({t['mode']})"] = {
                        'connected': True,
                        'banner': banner.strip(),
                        'elapsed_ms': int((time.time() - start) * 1000)
                    }
        except Exception as e:
            results[f"{server}:{t['port']} ({t['mode']})"] = {
                'connected': False,
                'error': str(e),
                'elapsed_ms': int((time.time() - start) * 1000)
            }
    return jsonify(results)




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)