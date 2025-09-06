from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime



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

db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
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
    # Get user's chat sessions that have at least one message
    user = User.query.filter_by(email=session.get('user_email')).first()
    chat_sessions = db.session.query(ChatSession).join(ChatMessage).filter(
        ChatSession.user_id == user.id
    ).distinct().order_by(ChatSession.updated_at.desc()).all()
    return render_template('chat.html', user_email=session.get('user_email'), chat_sessions=chat_sessions)



@app.route("/get", methods=["GET", "POST"])
@login_required
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
                return render_template('signup.html')
            
            # Check if user already exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash('Email already registered. Please login.', 'warning')
                return redirect(url_for('login'))
            
            # Create new user
            password_hash = generate_password_hash(password)
            new_user = User(
                name=name,
                email=email,
                password_hash=password_hash
            )
            db.session.add(new_user)
            db.session.commit()
            
            # Auto-login after successful signup
            session['user_email'] = email
            session['user_name'] = name
            flash('Welcome! Your account has been created successfully.', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            print(f"Signup error: {e}")
            db.session.rollback()
            flash('Database error. Please try again.', 'danger')
            return render_template('signup.html')
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
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)