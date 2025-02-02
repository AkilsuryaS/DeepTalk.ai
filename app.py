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
####
def is_deep_learning_related(question, knowledge_base):
    """
    Check if the question is related to deep learning by comparing similarity scores
    with the knowledge base content
    """
    # Get the most similar documents and their scores
    docs_and_scores = knowledge_base.similarity_search_with_score(question)
    
    # If the best match has a high similarity score (lower score means more similar)
    # We consider it related to deep learning
    if docs_and_scores and docs_and_scores[0][1] < 1.0:  # Threshold can be adjusted
        return True
    return False
#####

def call_groq_api(prompt, simplify=False, concise=False):
    try:
        if simplify:
            prompt = f"Explain the following in a very simple and easy-to-understand way in 5-6 sentences: {prompt}"
        elif concise:
            prompt = f"Provide a concise and crisp answer to the following in 5-6 sentences: {prompt}"
        response = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def answer_question(knowledge_base, question, simplify=False, concise=False):
    #### First check if the question is related to deep learning
    if not is_deep_learning_related(question, knowledge_base):
        return "I can only answer questions related to deep learning. Please ask a question about deep learning concepts, neural networks, or PyTorch."
    #####


    # If it is related, proceed with answering
    docs = knowledge_base.similarity_search(question)
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = call_groq_api(prompt, simplify=simplify, concise=concise)
    return response

def generate_quiz(knowledge_base, context, user_prompt):
    """
    Generate a quiz only if the topic is related to deep learning.
    """
    # Check if the topic is related to deep learning
    if not is_deep_learning_related(user_prompt, knowledge_base):
        return "I can only generate quizzes about deep learning topics. Please ask about deep learning concepts first."
    
    # Generate the quiz prompt
    prompt = f"""Context: {context}

Based on the user's question: '{user_prompt}', generate a quiz with 3 multiple-choice questions. 
Follow this EXACT format for each question:

Q1: [Question text here?]
A) [Option A text]
B) [Option B text]
C) [Option C text]
D) [Option D text]
Answer: [A/B/C/D]

Q2: [Question text here?]
A) [Option A text]
B) [Option B text]
C) [Option C text]
D) [Option D text]
Answer: [A/B/C/D]

Q3: [Question text here?]
A) [Option A text]
B) [Option B text]
C) [Option C text]
D) [Option D text]
Answer: [A/B/C/D]"""
    
    # Call the Groq API to generate the quiz
    quiz = call_groq_api(prompt, concise=True)
    
    # Validate the quiz format
    if not quiz or not quiz.strip():
        return "Failed to generate a valid quiz. Please try again."
        
    if not all(x in quiz for x in ['Q1:', 'Q2:', 'Q3:', 'Answer:']):
        return "Generated quiz is not in the correct format. Please try again."
    
    return quiz

def parse_quiz(quiz_text):
    """
    Parse quiz text into structured format with better error handling.
    """
    questions = []
    current_question = None
    question_found = False

    # Split the text into lines and clean them
    lines = [line.strip() for line in quiz_text.split('\n') if line.strip()]
    
    for line in lines:
        # Handle question lines
        if line.startswith('Q') and ':' in line:
            question_found = True
            # If there's a current question, save it before starting new one
            if current_question and current_question['options']:
                questions.append(current_question)
            
            # Extract question text
            try:
                question_text = line.split(': ', 1)[1]
                current_question = {
                    "question": question_text,
                    "options": [],
                    "answer": ""
                }
            except IndexError:
                st.error(f"Error parsing question: {line}")
                continue
                
        # Handle option lines
        elif line.startswith(('A)', 'B)', 'C)', 'D)')):
            if not question_found:
                continue  # Skip options if no question has been found yet
            if current_question is not None:
                current_question["options"].append(line)
            else:
                st.error(f"Option found without a question: {line}")
                
        # Handle answer lines
        elif line.startswith('Answer:'):
            if current_question is not None:
                try:
                    current_question["answer"] = line.split(': ')[1]
                except IndexError:
                    st.error(f"Error parsing answer: {line}")
            else:
                st.error(f"Answer found without a question: {line}")
    
    # Add the last question if it exists and has options
    if current_question and current_question['options']:
        questions.append(current_question)
    
    # Validate the parsed quiz
    if not questions:
        st.error("No valid questions were parsed from the quiz text")
        return None
        
    for i, q in enumerate(questions):
        if not q['options']:
            st.error(f"Question {i+1} has no options")
            return None
        if not q['answer']:
            st.error(f"Question {i+1} has no answer")
            return None
    
    return questions

# Streamlit app
st.title("Deep Learning with PyTorch Chatbot")
st.caption("Learn Deep Learning concepts from the d2l.pdf textbook")

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

# First, display existing messages
if st.session_state.current_chat["messages"]:
    for message in st.session_state.current_chat["messages"]:
        with st.chat_message(message["role"]):
            if message.get("type") == "text":
                st.markdown(message["content"])
            elif message.get("type") == "quiz":
                with st.expander("Show Quiz Result", expanded=True):
                    st.write("**Quiz Results:**")
                    for i, question in enumerate(message["quiz"]):
                        st.write(f"**Q{i+1}: {question['question']}**")
                        st.write(f"Your answer: {message['user_answers'][i]}")
                        st.write(f"Correct answer: {question['answer']}")
                        if message['user_answers'][i].startswith(question['answer']):
                            st.success("✅ Correct!")
                        else:
                            st.error("❌ Incorrect!")
                    st.write(f"**Score: {message['correct_answers']} out of {len(message['quiz'])} correct**")

# Then handle new input
if prompt := st.chat_input("Ask me anything about Deep Learning:"):
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to current chat
    st.session_state.current_chat["messages"].append({
        "role": "user",
        "type": "text",
        "content": prompt
    })

    # Generate and show response
    with st.spinner("Thinking..."):
        response = answer_question(knowledge_base, prompt, concise=True)
        
    with st.chat_message("assistant"):
        st.markdown(f"**Answer:** {response}")
    
    # Add assistant response to current chat
    st.session_state.current_chat["messages"].append({
        "role": "assistant",
        "type": "text",
        "content": f"**Answer:** {response}"
    })

    # Save current chat to chat sessions if it's new
    if st.session_state.current_chat not in st.session_state.chat_sessions:
        st.session_state.current_chat["date"] = today
        st.session_state.chat_sessions.append(st.session_state.current_chat)

# Quiz section
if len(st.session_state.current_chat["messages"]) > 0:  # Check if there are any messages
    # Get the last message that isn't from the assistant
    last_user_message = None
    for message in reversed(st.session_state.current_chat["messages"]):
        if message["role"] == "user" and message["type"] == "text":
            last_user_message = message["content"]
            break
    
    if last_user_message:
        # Check if the last user question was deep learning related
        if is_deep_learning_related(last_user_message, knowledge_base):
            # Show generate quiz button only for deep learning questions
            col1, col2 = st.columns([1, 4])  # Create columns for better spacing
            with col1:
                if st.button("Generate Quiz", key="generate_quiz_btn"):
                    context = " ".join([doc.page_content for doc in knowledge_base.similarity_search(last_user_message)])
                    quiz_text = generate_quiz(knowledge_base, context, last_user_message)
                    st.session_state.quiz = parse_quiz(quiz_text)
                    st.session_state.user_answers = {}
        else:
            # Show warning for non-deep learning questions
            st.warning("Please ask a question about deep learning concepts to generate a quiz. The last question was not related to the deep learning content from d2l.pdf.")
else:
    st.info("Start by asking a question about deep learning concepts from d2l.pdf. Then you can generate a quiz to test your understanding.")

# Display the quiz if it exists
if st.session_state.quiz:
    st.subheader("Quiz")
    for i, question in enumerate(st.session_state.quiz):
        st.write(f"**Q{i+1}: {question['question']}**")
        user_answer = st.radio(f"Select an answer for Q{i+1}:", question["options"], key=f"q{i}")
        st.session_state.user_answers[i] = user_answer

    if st.button("Submit Quiz"):
        correct_answers = 0
        quiz_results = []
        for i, question in enumerate(st.session_state.quiz):
            is_correct = st.session_state.user_answers[i].startswith(question["answer"])
            if is_correct:
                correct_answers += 1
            quiz_results.append({
                "question": question["question"],
                "user_answer": st.session_state.user_answers[i],
                "correct_answer": question["answer"],
                "is_correct": is_correct
            })
        
        # Save the quiz result as a chat message
        quiz_message = {
            "role": "assistant",
            "type": "quiz",
            "quiz": st.session_state.quiz,
            "user_answers": st.session_state.user_answers,
            "correct_answers": correct_answers,
            "quiz_results": quiz_results
        }
        st.session_state.current_chat["messages"].append(quiz_message)

        # Clear the current quiz state
        st.session_state.quiz = None
        st.session_state.user_answers = {}