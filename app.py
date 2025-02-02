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

def call_groq_api(prompt, simplify=False, concise=False):
    try:
        if simplify:
            prompt = f"Explain the following in a very simple and easy-to-understand way in 1-2 sentences: {prompt}"
        elif concise:
            prompt = f"Provide a concise and crisp answer to the following in 1-2 sentences: {prompt}"
        response = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def answer_question(knowledge_base, question, simplify=False, concise=False):
    docs = knowledge_base.similarity_search(question)
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = call_groq_api(prompt, simplify=simplify, concise=concise)
    return response

def generate_quiz(knowledge_base, context, user_prompt):
    prompt = f"Context: {context}\n\nBased on the user's question: '{user_prompt}', generate a quiz with 3 multiple-choice questions. Each question should be concise and have 4 options with one correct answer. Format the quiz as follows:\n\nQ1: [Question]\nA) [Option A]\nB) [Option B]\nC) [Option C]\nD) [Option D]\nAnswer: [Correct Option]\n\nQ2: [Question]\nA) [Option A]\nB) [Option B]\nC) [Option C]\nD) [Option D]\nAnswer: [Correct Option]\n\nQ3: [Question]\nA) [Option A]\nB) [Option B]\nC) [Option C]\nD) [Option D]\nAnswer: [Correct Option]"
    quiz = call_groq_api(prompt, concise=True)
    return quiz

def parse_quiz(quiz_text):
    questions = []
    current_question = None  # Initialize as None
    for line in quiz_text.split("\n"):
        line = line.strip()  # Remove leading/trailing whitespace
        if line.startswith("Q"):
            # If there's a current question, add it to the list
            if current_question:
                questions.append(current_question)
            # Initialize a new question
            try:
                question_text = line.split(": ")[1]  # Extract the question text
                current_question = {"question": question_text, "options": [], "answer": ""}
            except IndexError:
                st.error(f"Error parsing question: {line}")
                continue
        elif line.startswith("A)") or line.startswith("B)") or line.startswith("C)") or line.startswith("D)"):
            # Ensure current_question is initialized before appending options
            if current_question is not None:
                current_question["options"].append(line)
            else:
                st.error(f"Option found without a question: {line}")
        elif line.startswith("Answer:"):
            # Ensure current_question is initialized before setting the answer
            if current_question is not None:
                try:
                    current_question["answer"] = line.split(": ")[1]  # Extract the correct answer
                except IndexError:
                    st.error(f"Error parsing answer: {line}")
            else:
                st.error(f"Answer found without a question: {line}")
    # Add the last question if it exists
    if current_question is not None:
        questions.append(current_question)
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

# Initialize session state for chat history and quiz
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "current_chat" not in st.session_state:
    st.session_state.current_chat = {"title": "New Chat", "messages": []}
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}

# Sidebar for chat history
st.sidebar.title("Chat History")

# Button to start a new chat
if st.sidebar.button("New Chat"):
    st.session_state.current_chat = {"title": f"New Chat {len(st.session_state.chat_sessions) + 1}", "messages": []}
    st.session_state.quiz = None
    st.session_state.user_answers = {}

# Display chat sessions grouped by date
today = datetime.now().date()
yesterday = today - timedelta(days=1)
seven_days_ago = today - timedelta(days=7)

# Group chats by date
chats_today = []
chats_yesterday = []
chats_7_days = []
older_chats = []

for idx, chat in enumerate(st.session_state.chat_sessions):
    chat_date = chat.get("date", today)
    if chat_date == today:
        chats_today.append((idx, chat))
    elif chat_date == yesterday:
        chats_yesterday.append((idx, chat))
    elif chat_date >= seven_days_ago:
        chats_7_days.append((idx, chat))
    else:
        older_chats.append((idx, chat))

# Display chats in the sidebar
if chats_today:
    st.sidebar.subheader("Today")
    for idx, chat in chats_today:
        if st.sidebar.button(chat["title"], key=f"today_{idx}"):  # Unique key for each button
            st.session_state.current_chat = chat
            st.session_state.quiz = None
            st.session_state.user_answers = {}

if chats_yesterday:
    st.sidebar.subheader("Yesterday")
    for idx, chat in chats_yesterday:
        if st.sidebar.button(chat["title"], key=f"yesterday_{idx}"):  # Unique key for each button
            st.session_state.current_chat = chat
            st.session_state.quiz = None
            st.session_state.user_answers = {}

if chats_7_days:
    st.sidebar.subheader("7 Days")
    for idx, chat in chats_7_days:
        if st.sidebar.button(chat["title"], key=f"7days_{idx}"):  # Unique key for each button
            st.session_state.current_chat = chat
            st.session_state.quiz = None
            st.session_state.user_answers = {}

if older_chats:
    st.sidebar.subheader("Older")
    for idx, chat in older_chats:
        if st.sidebar.button(chat["title"], key=f"older_{idx}"):  # Unique key for each button
            st.session_state.current_chat = chat
            st.session_state.quiz = None
            st.session_state.user_answers = {}

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
        response = answer_question(knowledge_base, prompt, concise=True)
    
    # Add assistant response to current chat
    #st.session_state.current_chat["messages"].append({"role": "assistant", "content": response})
    st.session_state.current_chat["messages"].append({"role": "assistant", "content": f"**Answer:** {response}"})



    # Save current chat to chat sessions if it's new
    if st.session_state.current_chat not in st.session_state.chat_sessions:
        st.session_state.current_chat["date"] = today
        st.session_state.chat_sessions.append(st.session_state.current_chat)

# Display current chat messages
for message in st.session_state.current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Quiz section
if len(st.session_state.current_chat["messages"]) > 0:  # Check if there are any messages
    if st.button("Generate Quiz"):
        context = " ".join([doc.page_content for doc in knowledge_base.similarity_search(st.session_state.current_chat["messages"][-1]["content"])])
        quiz_text = generate_quiz(knowledge_base, context, st.session_state.current_chat["messages"][-1]["content"])
        st.session_state.quiz = parse_quiz(quiz_text)
        st.session_state.user_answers = {}
else:
    st.warning("Please provide a prompt about what you want to learn related to Deep Learning with PyTorch. After that, I can generate a quiz for you.")

if st.session_state.quiz:
    st.subheader("Quiz")
    for i, question in enumerate(st.session_state.quiz):
        st.write(f"**Q{i+1}: {question['question']}**")
        user_answer = st.radio(f"Select an answer for Q{i+1}:", question["options"], key=f"q{i}")
        st.session_state.user_answers[i] = user_answer

    if st.button("Submit Quiz"):
        correct_answers = 0
        for i, question in enumerate(st.session_state.quiz):
            if st.session_state.user_answers[i].startswith(question["answer"]):
                correct_answers += 1
                st.success(f"Q{i+1}: Correct! {question['answer']} is the right answer.")
            else:
                st.error(f"Q{i+1}: Incorrect. The correct answer is {question['answer']}.")
        st.write(f"**You got {correct_answers} out of {len(st.session_state.quiz)} questions correct!**")