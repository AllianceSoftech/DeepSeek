import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Inject custom CSS
st.markdown("""
<style>
    /* Sidebar background color */
    section[data-testid="stSidebar"] > div {
        background-color: #8cc6f4; /* Sky Blue */
        padding: 10px;
    }

    /* Text color for labels in the sidebar */
    section[data-testid="stSidebar"] label {
        color: white !important;
    }

    /* Text input box styling */
    div.stTextInput > div > div > input {
        background-color: blue !important; /* Blue background */
        border: 2px solid #8cc6f4 !important; /* SkyBlue border */
        color: white !important; /* Text color */
    }

    /* Reduce title font size to H2/H3 */
    h1 {
        font-size: 24px !important; /* Adjust size as needed */
    }

    /* Logo and title in the same row */
    .logo-title-container {
        display: flex;
        align-items: center;
        gap: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Logo and title in the same row
col1, col2 = st.columns([1, 4])
with col1:
    st.image(
        "https://skillpedia.ai/uploads/018f3926-3e32-7200-9b95-dbb0b652dadb.png",
        width=350,  # Adjust logo size as needed
    )
with col2:
    st.title("DeepSeek Powered ChatBot")  # Title with reduced font size
    st.caption("🚀 Your AI Assistant with Debugging Powers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:latest", "deepseek-r1:latest"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")
    st.markdown("Try Generative AI Models at [SkillPediaAI](https://skillpedia.ai/)")

# Initiate the chat engine
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct answers "
    "with print statements for debugging. Always respond in English or Language specified."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()
