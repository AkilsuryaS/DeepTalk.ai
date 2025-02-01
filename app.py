import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Replace Chroma with FAISS
import groq

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
            model="deepseek-r1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
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

def generate_quiz(knowledge_base):
    # Sample context from the knowledge base
    docs = knowledge_base.similarity_search("deep learning fundamentals", k=3)
    context = " ".join([doc.page_content for doc in docs])
    
    quiz_prompt = f"""Based on this context about deep learning: {context}
    Generate a quiz with 3 multiple-choice questions.
    Each question should have 4 options, with only one correct answer.
    Format the quiz as follows:
    
    Question 1: [Question text]
    A) [Option 1]
    B) [Option 2]
    C) [Option 3]
    D) [Option 4]
    Correct Answer: [Correct option]

    Question 2: [Question text]
    A) [Option 1]
    B) [Option 2]
    C) [Option 3]
    D) [Option 4]
    Correct Answer: [Correct option]

    Question 3: [Question text]
    A) [Option 1]
    B) [Option 2]
    C) [Option 3]
    D) [Option 4]
    Correct Answer: [Correct option]
    """
    response = call_groq_api(quiz_prompt)
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

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about Deep Learning:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.spinner("Thinking..."):
        response = answer_question(knowledge_base, prompt, simplify=True)

    # Add assistant response to chat history
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Quiz button
if st.button("Generate Quiz"):
    with st.spinner("Generating quiz..."):
        quiz = generate_quiz(knowledge_base)
        st.session_state.quiz = quiz

# Display quiz
if "quiz" in st.session_state:
    st.subheader("Quiz")
    st.write(st.session_state.quiz)