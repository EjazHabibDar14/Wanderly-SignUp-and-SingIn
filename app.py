from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import pandas as pd
from datetime import datetime
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import json
from offers_dict import offers_dict

from werkzeug.security import generate_password_hash, check_password_hash
from transformers import pipeline
from flask_jwt_extended import JWTManager
from flask_jwt_extended import create_access_token, create_refresh_token
from flask_jwt_extended import jwt_required, get_jwt_identity

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:admin123@127.0.0.1:5432/wanderly'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

app.config['JWT_SECRET_KEY'] = 'wanderly'
jwt = JWTManager(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255))
    name = db.Column(db.String(100))
    gender = db.Column(db.String(20))
    current_city = db.Column(db.String(100))
    age = db.Column(db.Integer)
    chat_history = db.Column(db.Text, default=json.dumps([]))
    labels = db.Column(db.JSON, nullable=True)
    scores = db.Column(db.JSON, nullable=True)
    offers = db.Column(db.JSON, nullable=True)
    #labels_scores = db.Column(db.JSON, nullable=True)

    def __init__(self, email, password, name, gender, current_city, age):
        self.email = email
        self.password_hash = generate_password_hash(password)
        self.name = name
        self.gender = gender
        self.current_city = current_city
        self.age = age
        self.chat_history = json.dumps([]) 
    
    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')
    gender = data.get('gender')
    current_city = data.get('current_city')
    age = data.get('age')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'User already registered'}), 409
    
    new_user = User(email=email, password=password, name=name, gender=gender, current_city=current_city, age=age)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully', 'user_id': new_user.id}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    user = User.query.filter_by(email=email).first()
    if user:
        if check_password_hash(user.password_hash, password):
            access_token = create_access_token(identity=email)
            refresh_token = create_refresh_token(identity=email)
            return jsonify({
                'message': 'Login successful',
                'user_id': user.id,
                'access_token': access_token,
                'refresh_token': refresh_token
            }), 200
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
    else:
        return jsonify({'error': 'Register user first'}), 404
    
@app.route('/chat', methods=['POST'])
@jwt_required()
def chat():
    user_identity = get_jwt_identity()  # Extract user identity from JWT
    data = request.get_json()
    message = data.get('message')

    if not message:
        return jsonify({'error': 'Message is required'}), 400

    # Fetch the user based on the email provided in the JWT, not from the request payload
    user = User.query.filter_by(email=user_identity).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    chat_history = json.loads(user.chat_history)
    vector_store = initialize_chatbot()

    if message.lower() in ['exit', 'quit']:
        labels, scores = classify_chat_history(chat_history)
        matched_offers = get_offers(labels, offers_dict)

        user.labels = labels
        user.scores = scores
        user.offers = matched_offers
        db.session.commit()

        return jsonify({'answer': 'Chat session ended by user.', 'Matched Offers': matched_offers})

    # Proceed with chat as usual
    chat_model = ChatOpenAI(temperature=0.0, model_name='gpt-4-turbo')
    model = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )

    response = model.invoke({"question": message, "chat_history": chat_history})
    chat_history.append({"question": message, "answer": response['answer']})
    user.chat_history = json.dumps(chat_history)
    db.session.commit()

    return jsonify({'answer': response['answer']})


@app.route('/update_user', methods=['PUT'])
@jwt_required()
def update_user():
    user_identity = get_jwt_identity()  # Extract user identity from JWT
    data = request.get_json()

    # Fetch the user based on the email provided in the JWT
    user = User.query.filter_by(email=user_identity).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    # Update the user attributes if provided in the request
    user.email = data.get('email', user.email)
    if 'password' in data:
        user.password = generate_password_hash(data['password'])  # Ensure the password is hashed if it's being updated
    user.name = data.get('name', user.name)
    user.gender = data.get('gender', user.gender)
    user.current_city = data.get('current_city', user.current_city)
    user.age = data.get('age', user.age)

    db.session.commit()
    return jsonify({'message': 'User details updated successfully'}), 200



@app.route('/get_chat_history', methods=['GET'])
@jwt_required()
def get_chat_history():
    user_identity = get_jwt_identity()  # Extract user identity from JWT

    # Fetch the user based on the email provided in the JWT
    user = User.query.filter_by(email=user_identity).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    chat_history = json.loads(user.chat_history)
    return jsonify({'user_id': user.id, 'chat_history': chat_history}), 200


@app.route('/get_offers', methods=['GET'])
@jwt_required()
def get_offers_for_user():
    user_identity = get_jwt_identity()  # Extract user identity from JWT

    # Fetch the user based on the email provided in the JWT
    user = User.query.filter_by(email=user_identity).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    offers = user.offers  # Offers are stored as JSON in the database
    return jsonify({'user_id': user.id, 'offers': offers}), 200



def initialize_chatbot():
    with app.app_context():
        load_dotenv()
        embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        persist_directory = "E:/TransData/Internal-wanderly-app-dev/FireCrawl/TravelDataChromaStore"
        vector_store = Chroma(
            collection_name="travel_data",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        return vector_store
    
def classify_chat_history(chat_history):

    candidate_labels = [
        "Direct Flights", "Connected Flights", "Discounted Flights", "Round Trips", "One Way Trips", "Rentals",  
        "1 Star Hotels", "2 Star Hotels", "3 Star Hotels", "4 Star Hotels", "5 Star Hotels",  
        "Restaurants  and Food", "Cultural Activities",
        "Outdoor Activities", "Shopping Malls" 
    ]

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",  multi_label=True)
    sequence = json.dumps(chat_history, indent=4)
    output = classifier(sequence, candidate_labels)

    # Sorting labels and scores by scores in descending order
    results = list(zip(output['labels'], output['scores']))
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

    # Selecting top 3 results
    top_results = results_sorted[:3]
    print("TOP RESULTS", top_results)
    
    # Unzipping the top results
    top_labels, top_scores = zip(*top_results)
    return top_labels, top_scores

def get_offers(top_labels, offers_dict):
    matched_offers = []
    for label in top_labels:
        if label in offers_dict:
            print(f"Matched label: {label}")
            matched_offers.extend(offers_dict[label])
    return matched_offers

# def run_conversation(vector_store, chat_history):
#     chat_model = ChatOpenAI(temperature=0.0, model_name='gpt-4-turbo')
#     model = ConversationalRetrievalChain.from_llm(
#         llm=chat_model,
#         retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
#         memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     )
#     print("Begin your conversation:")
#     while True:
#         print("****************")
#         print("Chat History:", chat_history)
#         print("****************")
#         question = input("You: ")
#         if question.lower() in ['exit', 'quit']:
#             break
#         response = model.invoke({"question": question, "chat_history": chat_history})
#         chat_history.append({"question": question, "answer": response['answer']})
#         print("Bot:", response['answer'])
#     return chat_history

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create database tables if they don't exist
    app.run(debug=True)