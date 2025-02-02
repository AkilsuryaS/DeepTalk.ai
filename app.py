import streamlit as st
from streamlit_webrtc import webrtc_streamer
import speech_recognition as sr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import groq
from datetime import datetime, timedelta

# Initialize Groq client
groq_api_key = "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"
client = groq.Client(api_key=groq_api_key)

# Load knowledge base (same as before)
@st.cache_resource
def load_knowledge_base():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    knowledge_base = FAISS.load_local(
        folder_path="data/faiss_db",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return knowledge_base

# Speech-to-text function
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)  # You can also use Whisper or other APIs
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Sorry, there was an issue with the speech recognition service."

# Voice input component
def voice_input():
    st.write("Click the button below to start speaking:")
    webrtc_ctx = webrtc_streamer(
        key="voice-input",
        mode="audio",
        audio_receiver_size=1024,
    )
    if webrtc_ctx.audio_receiver:
        audio_frames = webrtc_ctx.audio_receiver.get_frames()
        if audio_frames:
            # Save audio to a file
            audio_file = "user_audio.wav"
            with open(audio_file, "wb") as f:
                for frame in audio_frames:
                    f.write(frame.to_ndarray().tobytes())
            
            # Convert audio to text
            user_input = speech_to_text(audio_file)
            return user_input
    return None

# Main app
st.title("Deep Learning with PyTorch Chatbot")
st.caption("Learn Deep Learning concepts from the [d2l.pdf textbook](https://d2l.ai/d2l-en.pdf)")

# Load knowledge base
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

# Sidebar for chat history (same as before)
st.sidebar.title("Chat History")
if st.sidebar.button("New Chat"):
    st.session_state.current_chat = {"title": f"New Chat {len(st.session_state.chat_sessions) + 1}", "messages": []}
    st.session_state.quiz = None
    st.session_state.user_answers = {}

# Chat interface
st.subheader(st.session_state.current_chat["title"])

# Display existing messages (same as before)
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

# Voice or text input
input_mode = st.radio("Choose input mode:", ("Text", "Voice"))

if input_mode == "Voice":
    user_input = voice_input()
    if user_input:
        st.write(f"**You said:** {user_input}")
        prompt = user_input
else:
    prompt = st.chat_input("Ask me anything about Deep Learning:")

# Handle user input
if prompt:
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

# Quiz section (same as before)
if len(st.session_state.current_chat["messages"]) > 0:
    last_user_message = None
    for message in reversed(st.session_state.current_chat["messages"]):
        if message["role"] == "user" and message["type"] == "text":
            last_user_message = message["content"]
            break
    
    if last_user_message:
        if is_deep_learning_related(last_user_message, knowledge_base):
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Generate Quiz", key="generate_quiz_btn"):
                    context = " ".join([doc.page_content for doc in knowledge_base.similarity_search(last_user_message)])
                    quiz_text = generate_quiz(knowledge_base, context, last_user_message)
                    st.session_state.quiz = parse_quiz(quiz_text)
                    st.session_state.user_answers = {}
        else:
            st.warning("Please ask a question about deep learning concepts to generate a quiz. The last question was not related to the deep learning content from d2l.pdf.")
else:
    st.info("Start by asking a question about deep learning concepts from d2l.pdf. Then you can generate a quiz to test your understanding.")

# Display the quiz if it exists (same as before)
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
        
        quiz_message = {
            "role": "assistant",
            "type": "quiz",
            "quiz": st.session_state.quiz,
            "user_answers": st.session_state.user_answers,
            "correct_answers": correct_answers,
            "quiz_results": quiz_results
        }
        st.session_state.current_chat["messages"].append(quiz_message)

        st.session_state.quiz = None
        st.session_state.user_answers = {}