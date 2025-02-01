import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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
            model="deepseek-r1-distill-llama-70b",
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

def generate_quiz(knowledge_base, question):
    # Generate quiz based on the user's question
    docs = knowledge_base.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in docs])
    
    quiz_prompt = f"""Based on this context: {context}
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
    quiz = call_groq_api(quiz_prompt)
    return quiz

def parse_quiz(quiz):
    """Parse the quiz string into questions, options, and correct answers."""
    questions = []
    lines = quiz.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("Question"):
            question = lines[i].split(":")[1].strip()
            options = [lines[i + j].strip() for j in range(1, 5)]
            correct_answer = lines[i + 5].split(":")[1].strip()
            questions.append({
                "question": question,
                "options": options,
                "correct_answer": correct_answer
            })
            i += 6
        else:
            i += 1
    return questions

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

# Sidebar for chat history
st.sidebar.title("Chat History")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history in the sidebar
for message in st.session_state.messages:
    st.sidebar.write(f"{message['role'].capitalize()}: {message['content']}")

# Chat interface
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
    if "messages" in st.session_state and st.session_state.messages:
        # Use the last user question to generate the quiz
        last_user_question = next(
            (msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "user"),
            None
        )
        if last_user_question:
            with st.spinner("Generating quiz..."):
                quiz = generate_quiz(knowledge_base, last_user_question)
                st.session_state.quiz = quiz
                st.session_state.quiz_questions = parse_quiz(quiz)
        else:
            st.warning("No user question found to generate a quiz.")
    else:
        st.warning("No chat history found to generate a quiz.")

# Display quiz and handle user answers
if "quiz_questions" in st.session_state:
    st.subheader("Quiz")
    for i, q in enumerate(st.session_state.quiz_questions):
        st.write(f"**Question {i + 1}:** {q['question']}")
        user_answer = st.radio(
            f"Select an answer for Question {i + 1}:",
            q["options"],
            key=f"quiz_{i}"
        )
        if st.button(f"Submit Answer for Question {i + 1}"):
            if user_answer.startswith(q["correct_answer"]):
                st.success("Correct! 🎉")
            else:
                st.error(f"Incorrect. The correct answer is: {q['correct_answer']}")
            st.write(f"Explanation: {call_groq_api(f'Explain why the correct answer is {q["correct_answer"]} for the question: {q["question"]}')}")