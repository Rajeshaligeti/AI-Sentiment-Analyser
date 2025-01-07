import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import chardet  # Library to detect encoding

# Set custom page configuration
st.set_page_config(
    page_title="🚀 AI Sentiment Analyzer",
    page_icon="",
    layout="wide"
)

# Custom CSS for complete UI redesign
st.markdown(
    """
    <style>
    /* General styling */
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f4f7fc;
        margin: 0;
        padding: 0;
    }
    .stApp {
        background: #f4f7fc;
        border-radius: 15px;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background: #2D2E32;
        color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
    }
    h1, h2, h3 {
        font-weight: 600;
        color: #2B2D42;
        text-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #6C63FF, #3A3D99);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0px 5px 15px rgba(108, 99, 255, 0.3);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        background: #3A3D99;
    }
    .stTextArea textarea, .stTextInput input {
        background: #ffffff;
        color: #333;
        border: 2px solid #3A3D99;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        padding: 10px;
        margin-top: 30px;
        background: #2D2E32;
        color: #FFFFFF;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown("<h1 style='text-align: center;'>🚀 AI Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.write("""
This advanced app leverages AI to assess the sentiment of your input text, providing insights into whether it is *Positive, **Negative, **Neutral, or **Mixed*.
""")

with st.sidebar:
    st.header("Navigation")
    analysis_mode = st.radio("Choose Analysis Mode", ["Single Input", "Batch Input", "Chatbot"])
    st.write("Navigate and analyze your text seamlessly.")

# Load Ollama model
@st.cache_resource
def load_sentiment_model():
    return OllamaLLM(model="llama3.2")  # Replace with actual model if necessary

sentiment_model = load_sentiment_model()

def chatbot_mode():
    st.subheader("Chat with AI")
    st.write("Engage in a conversation with the AI model.")
    
    # Session state for chatbot history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Your message:")
    if st.button("Send"):
        if user_input.strip():
            with st.spinner("AI is thinking..."):
                try:
                    # Build the chat history into the prompt
                    conversation_history = "\n".join(
                        f"User: {chat['user']}\nAI: {chat['ai']}"
                        for chat in st.session_state["chat_history"]
                    )
                    prompt = (
                        f"You are a conversational assistant. Engage in a friendly and helpful discussion with the user.\n\n"
                        f"{conversation_history}\n"
                        f"User: {user_input}\nAI:"
                    )
                    
                    # Get response from the model
                    response = sentiment_model.invoke(prompt)
                    st.session_state["chat_history"].append({"user": user_input, "ai": response})
                    
                    # Display the conversation
                    for chat in st.session_state["chat_history"]:
                        st.markdown(f"*User:* {chat['user']}")
                        st.markdown(f"*AI:* {chat['ai']}")
                except Exception as e:
                    st.error(f"Error during chat: {e}")
        else:
            st.warning("Please enter a message to send.")


# Preprocess function
def preprocess_input(text):
    if not text.strip() or re.match(r'^[0-9a-fA-F]+$', text.strip()):
        st.warning("Please provide a valid text input for sentiment analysis.")
        return None
    return text

# Function to parse sentiment and confidence from raw responses
def parse_sentiment_response(response, user_input):
    try:
        # Debugging: Log raw response
        st.write(f"Raw Response: {response}")

        # Handle the case when the response has an error message (but try to provide a fallback)
        if not response.strip():
            return {"sentiment": "Neutral", "confidence": 0.5}  # Default neutral if no response is given
        elif "I can't provide assistance" in response:
            return {"sentiment": "Neutral", "confidence": 0.5}  # Default neutral as fallback

        # Match sentiment and confidence using regex
        match = re.search(r"Sentiment:\s*(\w+).?Confidence\s*Score?:?\s([\d.]+)", response, re.DOTALL)
        if match:
            sentiment = match.group(1).strip().capitalize()
            confidence = float(match.group(2).strip())
            return {"sentiment": sentiment, "confidence": confidence}

        # If we can't parse the sentiment properly, fallback to Neutral with medium confidence
        return {"sentiment": "Neutral", "confidence": 0.5}

    except Exception as e:
        st.error(f"Error during response parsing: {e}")
        return {"sentiment": "Neutral", "confidence": 0.5}  # Default to neutral if error occurs


# Function to analyze sentiment of a single input
def analyze_sentiment(text):
    preprocessed_text = preprocess_input(text)
    if not preprocessed_text:
        return None
    template = (
    "You are an AI sentiment analyzer that interprets all types of inputs, including informal, rude, or slang language. "
    "Please return the sentiment (positive, negative, neutral, or mixed) and a confidence score (0 to 1), regardless of the input.\n"
    "Input: {input_text}\nResponse:"
)


    # Format prompt
    prompt = ChatPromptTemplate.from_template(template)
    prompt_value = prompt.format(input_text=preprocessed_text)

    try:
        response = sentiment_model.invoke(prompt_value)
        result = parse_sentiment_response(response, text)

        # Return results or notify user about unrecognized input
        if result['sentiment'] == "unknown":
            st.warning("The model couldn't determine the sentiment of your input.")
        return result
    except Exception as e:
        st.error(f"Error occurred during analysis: {e}")
        return None


# Input Handling
if analysis_mode == "Single Input":
    st.subheader("Enter Text for Sentiment Analysis")
    user_input = st.text_area("Type your text here", height=200)
    
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                result = analyze_sentiment(user_input)
                
                if result:
                    if 'raw_response' in result:
                        st.write(f"*Model Explanation*: {result['raw_response']}")
                    else:
                        sentiment = result['sentiment']
                        confidence = result['confidence']
                        st.success(f"*Sentiment: {sentiment.capitalize()} | **Confidence*: {confidence:.2f}")
                    
                    st.subheader("🌀 Word Cloud of Input Text")
                    wordcloud = WordCloud(background_color="white").generate(user_input)
                    plt.imshow(wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    st.pyplot()
                else:
                    st.warning("Unable to analyze sentiment. Please try again.")
        else:
            st.warning("Please enter some text to analyze.")

# Batch input handling
def detect_encoding(file):
    raw_data = file.read(10000)
    result = chardet.detect(raw_data)
    file.seek(0)
    return result['encoding']

# Batch input option with line number and clickable link
if analysis_mode == "Batch Input":
    st.subheader("Upload a File for Batch Analysis")
    uploaded_file = st.file_uploader("Upload a CSV or TXT file (one text per line)", type=["csv", "txt"])
    
    if uploaded_file:
        encoding = detect_encoding(uploaded_file)
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, encoding=encoding)
                texts = df.iloc[:, 0].tolist()
            else:
                texts = uploaded_file.read().decode(encoding).splitlines()
            
            st.subheader("Batch Sentiment Analysis Results")
            confidence_threshold = st.slider("Minimum Confidence Score", 0.0, 1.0, 0.5, 0.05)
            
            if st.button("Analyze Batch Sentiment"):
                with st.spinner("Analyzing..."):
                    results = []
                    for i, text in enumerate(texts):
                        result = analyze_sentiment(text)
                        if result and result.get('confidence') >= confidence_threshold:
                            # Create a link pointing to the line number in the batch (using anchor in HTML)
                            line_link = f'<a href="javascript:void(0)" onclick="window.scrollTo(0, document.getElementById(\'line-{i}\').offsetTop)">Line {i + 1}</a>'
                            results.append((line_link, text, result['sentiment'], result['confidence']))
                    
                    if results:
                        # Display the results in a table format with clickable links
                        batch_df = pd.DataFrame(results, columns=["Line", "Text", "Sentiment", "Confidence"])
                        batch_df["Line"] = batch_df["Line"].apply(lambda x: f"{x}")
                        st.write(batch_df.to_html(escape=False), unsafe_allow_html=True)
                    else:
                        st.warning("No results met the threshold.")
        
        except UnicodeDecodeError:
            st.error("Unsupported file encoding.")

elif analysis_mode == "Chatbot":
    chatbot_mode()


# Footer
st.markdown("<div class='footer'>Built using Streamlit</div>", unsafe_allow_html=True)