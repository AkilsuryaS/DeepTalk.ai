import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import groq
from datetime import datetime, timedelta
import random

# Initialize Groq client
groq_api_key = "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"
client = groq.Client(api_key=groq_api_key)

@st.cache_resource
def load_knowledge_base():
    """Load the preprocessed FAISS database"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Load the persisted FAISS database
    knowledge_base = FAISS.load_local(
        folder_path="data/faiss_db",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return knowledge_base

def call_groq_api(prompt, simplify=False):
    try:
        if simplify:
            prompt = f"Explain the following in short and easy-to-understand way: {prompt}"
        response = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def generate_quiz(context):
    """Generate a quiz based on the conversation context"""
    prompt = f"""Based on the following context, generate 3 multiple choice questions to test understanding.
    Format each question as a dictionary with 'question', 'options' (list of 4 choices), 'correct_answer' (index 0-3),
    and 'explanation'. Make questions challenging but fair.
    
    Context: {context}
    
    Return only the Python list of dictionaries, no other text."""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        # Safety evaluation of the response
        quiz_data = eval(response.choices[0].message.content)
        if not isinstance(quiz_data, list) or len(quiz_data) != 3:
            raise ValueError("Invalid quiz format")
        return quiz_data
    except Exception as e:
        return None

def answer_question(knowledge_base, question, simplify=False):
    docs = knowledge_base.similarity_search(question)
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = call_groq_api(prompt, simplify=simplify)
    return response, context

# Initialize session states
if 'quiz_active' not in st.session_state:
    st.session_state.quiz_active = False
if 'current_quiz' not in st.session_state:
    st.session_state.current_quiz = None
if 'quiz_responses' not in st.session_state:
    st.session_state.quiz_responses = []
if 'last_context' not in st.session_state:
    st.session_state.last_context = ""

# Streamlit app
st.title("Deep Learning with PyTorch Chatbot")
st.write("Learn Deep Learning concepts interactively and test your knowledge with quizzes!")

# Load the preprocessed knowledge base
try:
    knowledge_base = load_knowledge_base()
    st.success("Successfully loaded the knowledge base!")
except Exception as e:
    st.error(f"Error loading knowledge base: {str(e)}")
    st.stop()

# Initialize session state for chat history
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "current_chat" not in st.session_state:
    st.session_state.current_chat = {"title": "New Chat", "messages": []}

# Sidebar for chat history
st.sidebar.title("Chat History")

# Button to start a new chat
if st.sidebar.button("New Chat"):
    st.session_state.current_chat = {"title": f"New Chat {len(st.session_state.chat_sessions) + 1}", "messages": []}
    st.session_state.quiz_active = False
    st.session_state.current_quiz = None
    st.session_state.quiz_responses = []

# [Previous chat history grouping code remains the same]

# Chat interface
st.subheader(st.session_state.current_chat["title"])

# Display current chat messages
for message in st.session_state.current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Quiz section
if st.session_state.current_chat["messages"] and st.button("Generate Quiz"):
    st.session_state.quiz_active = True
    with st.spinner("Generating quiz..."):
        st.session_state.current_quiz = generate_quiz(st.session_state.last_context)

if st.session_state.quiz_active and st.session_state.current_quiz:
    st.subheader("Quiz Time! 🎯")
    
    for i, question in enumerate(st.session_state.current_quiz):
        st.write(f"\n**Question {i+1}:** {question['question']}")
        
        # Create a unique key for each radio button
        response = st.radio(
            "Select your answer:",
            question['options'],
            key=f"quiz_{i}",
            index=None
        )
        
        # If user has selected an answer
        if response:
            selected_index = question['options'].index(response)
            if selected_index == question['correct_answer']:
                st.success("✅ Correct!")
            else:
                st.error("❌ Incorrect")
            
            st.info(f"Explanation: {question['explanation']}")
            
        st.write("---")

# Chat input
if prompt := st.chat_input("Ask me anything about Deep Learning:"):
    # Add user message to current chat
    st.session_state.current_chat["messages"].append({"role": "user", "content": prompt})
    
    # Generate response
    with st.spinner("Thinking..."):
        response, context = answer_question(knowledge_base, prompt, simplify=True)
        st.session_state.last_context = context  # Store context for quiz generation
    
    # Add assistant response to current chat
    st.session_state.current_chat["messages"].append({"role": "assistant", "content": response})

    # Reset quiz state when new question is asked
    st.session_state.quiz_active = False
    st.session_state.current_quiz = None
    st.session_state.quiz_responses = []

    # Save current chat to chat sessions if it's new
    if st.session_state.current_chat not in st.session_state.chat_sessions:
        st.session_state.current_chat["date"] = datetime.now().date()
        st.session_state.chat_sessions.append(st.session_state.current_chat)

# Display current chat messages (if needed again)
for message in st.session_state.current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])