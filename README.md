# Deep Learning Interactive Learning Assistant 🤖

An AI-powered interactive learning assistant that helps users understand deep learning concepts through conversation and automated quizzes. Built with Streamlit, LangChain, and Groq LLM.

## 🌟 Features

- **Intelligent Q&A**: Get precise answers to your deep learning questions using content from the d2l.ai textbook
- **Interactive Quizzes**: Automatically generated multiple-choice questions to test your understanding
- **Conversation History**: Keep track of your learning sessions with organized chat history
- **Smart Context Understanding**: Utilizes FAISS similarity search for accurate and relevant responses
- **Real-time Learning Assessment**: Immediate feedback on quiz performance

## 🛠️ Technical Architecture

- **Frontend**: Streamlit
- **Language Model**: Groq API (deepseek-r1-distill-llama-70b)
- **Embeddings**: HuggingFace's sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Document Processing**: LangChain
- **Text Processing**: PyPDF2

## 📋 Prerequisites

- Python 3.8+
- Groq API key
- Required Python packages (see requirements.txt)

## 🚀 Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd deep-learning-assistant
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Prepare the knowledge base:
```bash
python preprocess.py
```

3. Run the application:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
deep-learning-assistant/
├── app.py              # Main Streamlit application
├── preprocess.py       # Data preprocessing script
├── requirements.txt    # Project dependencies
├── data/              # Directory for processed data
│   └── faiss_db/      # FAISS vector store
└── README.md
```

## 💻 Usage

1. Start by running the preprocessing script to create the FAISS database
2. Launch the Streamlit app
3. Ask questions about deep learning concepts
4. Generate and take quizzes to test your understanding
5. Review your chat history and track your progress


## 🛡️ Limitations

- The assistant is specifically designed for deep learning topics and will not answer unrelated questions
- Responses are based on the content from d2l.ai textbook
- Internet connection required for Groq API calls

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- [d2l.ai](https://d2l.ai/) for the comprehensive deep learning content
- Daniel Bourke and Krish Naik for inspiration
- Groq for providing the LLM API
- HuggingFace for the sentence transformers
- The Streamlit team for the amazing framework

## 📧 Contact

Akilsurya Sivakumar - akilsurya20399@gmail.com

---
Don't forget to ⭐ the repo if you find this project useful!
