import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import groq
from datetime import datetime, timedelta

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
            prompt = f"Explain the following in a very simple and easy-to-understand way: {prompt}"
        response = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def answer_question(knowledge_base, question, simplify=False):
    docs = knowledge_base.similarity_search(question)
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = call_groq_api(prompt, simplify=simplify)
    return response

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

# Display chat sessions grouped by date
today = datetime.now().date()
yesterday = today - timedelta(days=1)
seven_days_ago = today - timedelta(days=7)

# Group chats by date
chats_today = []
chats_yesterday = []
chats_7_days = []
older_chats = []

for chat in st.session_state.chat_sessions:
    chat_date = chat.get("date", today)
    if chat_date == today:
        chats_today.append(chat)
    elif chat_date == yesterday:
        chats_yesterday.append(chat)
    elif chat_date >= seven_days_ago:
        chats_7_days.append(chat)
    else:
        older_chats.append(chat)

# Display chats in the sidebar
if chats_today:
    st.sidebar.subheader("Today")
    for chat in chats_today:
        if st.sidebar.button(chat["title"]):
            st.session_state.current_chat = chat

if chats_yesterday:
    st.sidebar.subheader("Yesterday")
    for chat in chats_yesterday:
        if st.sidebar.button(chat["title"]):
            st.session_state.current_chat = chat

if chats_7_days:
    st.sidebar.subheader("7 Days")
    for chat in chats_7_days:
        if st.sidebar.button(chat["title"]):
            st.session_state.current_chat = chat

if older_chats:
    st.sidebar.subheader("Older")
    for chat in older_chats:
        if st.sidebar.button(chat["title"]):
            st.session_state.current_chat = chat

# Chat interface
st.subheader(st.session_state.current_chat["title"])

# Display current chat messages
for message in st.session_state.current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about Deep Learning:"):
    # Add user message to current chat
    st.session_state.current_chat["messages"].append({"role": "user", "content": prompt})
    
    # Generate response
    with st.spinner("Thinking..."):
        response = answer_question(knowledge_base, prompt, simplify=True)
    
    # Add assistant response to current chat
    st.session_state.current_chat["messages"].append({"role": "assistant", "content": response})

    # Save current chat to chat sessions if it's new
    if st.session_state.current_chat not in st.session_state.chat_sessions:
        st.session_state.current_chat["date"] = today
        st.session_state.chat_sessions.append(st.session_state.current_chat)

# Display current chat messages
for message in st.session_state.current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])