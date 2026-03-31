import streamlit as st
from vectorstore import VectorStore
from chatbot import Chatbot
import os

def get_cohere_key():
    try:
        secrets_paths = [
            os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
            os.path.join(os.path.dirname(__file__), "..", ".streamlit", "secrets.toml")
        ]
        if any(os.path.exists(p) for p in secrets_paths):
            return st.secrets.get("COHERE_API_KEY", "")
    except Exception:
        pass
    return ""
def show_about():
    st.markdown("## About This Project")
    st.markdown("**Developed by:** Tammali Suresh")
    st.markdown("---")

    st.markdown("### Problem Statement")
    st.write("""
    Team members often struggle to quickly find relevant information from internal documents,
    notes, and past project materials. As a result, knowledge is not reused effectively,
    work gets repeated, and onboarding new team members takes longer than it should.
    """)

    st.markdown("### Problem Understanding")
    st.write("""
    Organizations accumulate large volumes of documents over time — reports, meeting notes,
    project files, and manuals. Without an intelligent search system, employees waste hours
    manually scanning through files, leading to duplicated effort and slower onboarding.
    """)

    st.markdown("### Proposed Solution")
    st.write("""
    An AI-powered Retrieval-Augmented Generation (RAG) system that allows users to upload
    any PDF document and ask natural language questions. The system retrieves the most
    relevant sections from the document and uses a large language model (Cohere command-r-plus)
    to generate accurate, context-aware answers.
    """)

    st.markdown("### How It Works")
    st.write("""
    1. Upload a PDF document
    2. The document is split into chunks and indexed using TF-IDF vectorization
    3. When a question is asked, the most relevant chunks are retrieved using cosine similarity
    4. The retrieved context is passed to Cohere's LLM to generate a precise answer
    """)

    st.markdown("### Evaluation")
    st.write("""
    - Answers are grounded in the actual document content, reducing hallucinations
    - Fast retrieval using local TF-IDF — no external vector database required
    - Chat history is maintained within the session for multi-turn conversations
    - PDF is processed only once per session for efficiency
    """)

    st.markdown("### Reflection")
    st.write("""
    This prototype demonstrates how RAG can solve real knowledge management challenges.
    Future improvements could include support for multiple documents, semantic embeddings
    for better retrieval accuracy, and user authentication for team-wide deployment.
    """)

    st.markdown("### Submission Format")
    st.write("""
    - Short write-up covering all sections above
    - Source code (app.py, chatbot.py, vectorstore.py)
    - Live demo via Streamlit
    - Walkthrough: Upload a PDF → Ask a question → Get an AI-generated answer
    """)

def main():
    st.title("AI-Powered Knowledge Retrieval Assistant for Intelligent Document Search")

    page = st.radio("Navigation", ["Chat", "About Project"], horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    if page == "About Project":
        show_about()
        return

    st.write("Upload a PDF and ask questions about its content.")

    if "chat_history" not in st.session_state or any(len(h) != 2 for h in st.session_state["chat_history"]):
        st.session_state["chat_history"] = []
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "last_file" not in st.session_state:
        st.session_state["last_file"] = None

    with st.sidebar:
        st.header("Tammali Suresh 👤")
        st.write("Project by Tammali Suresh")
        st.markdown("---")
        st.header("API Keys 🔑")
        default_key = get_cohere_key()
        cohere_api_key = st.text_input("Cohere API Key", value=default_key, type="password")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    user_query = st.text_input("Ask a question based on the document")

    if st.button("Submit"):
        if not uploaded_file or not cohere_api_key or not user_query:
            st.error("Please fill in all fields: Cohere API key, upload a PDF, and enter a question.")
        else:
            if st.session_state["last_file"] != uploaded_file.name:
                with st.spinner("Processing PDF..."):
                    with open("uploaded_document.pdf", "wb") as f:
                        f.write(uploaded_file.read())
                    st.session_state["vectorstore"] = VectorStore("uploaded_document.pdf", cohere_api_key)
                    st.session_state["last_file"] = uploaded_file.name

            with st.spinner("Generating response..."):
                chatbot = Chatbot(st.session_state["vectorstore"], cohere_api_key)
                response, _ = chatbot.respond(user_query)
                accumulated_response = ""
                for event in response:
                    if hasattr(event, 'type') and event.type == 'content-delta':
                        accumulated_response += event.delta.message.content.text
                st.session_state["chat_history"].append((user_query, accumulated_response))

    if st.session_state["chat_history"]:
        for q, a in st.session_state["chat_history"]:
            st.write(f"**You:** {q}")
            st.write(f"**Bot:** {a}")

if __name__ == "__main__":
    main()
