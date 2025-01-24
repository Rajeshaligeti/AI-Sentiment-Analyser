# Streamlit Sentiment Analyzer

A web-based application built using Streamlit that performs sentiment analysis on text or batch files. The app uses natural language processing techniques to classify text as positive, negative, or neutral. It also generates visual insights such as word clouds based on the text provided.

## Features

- **Sentiment Analysis**: Classifies text into positive, negative, or neutral sentiment.
- **Batch File Support**: Allows users to upload text files and perform sentiment analysis on multiple lines of text.
- **Word Cloud Generation**: Visual representation of the most frequent words in the input text.
- **User-friendly Interface**: Simple and interactive UI built using Streamlit.

## Requirements

Make sure you have the following dependencies installed:

- Python 3.x
- Streamlit
- LangChain
- Ollama
- Matplotlib

You can install the required dependencies using `pip`:

```bash
pip install streamlit langchain ollama matplotlib
```
## Usage
Clone the repository:
```
git clone https://github.com/yourusername/streamlit-sentiment-analyzer.git

cd streamlit-sentiment-analyzer
```
Run the Streamlit app:
```
streamlit run app.py
```
Open the provided URL in your browser (usually http://localhost:8501) to interact with the app.

## Input
**Text Input**: Enter the text you want to analyze in the text box.

**File Input**: Upload a .txt file containing multiple lines of text for batch sentiment analysis.

---

After entering or uploading text, the app will display the sentiment classification (positive, negative, or neutral) and show a word cloud of the most frequently used words.

## License
This project is licensed under the Apache License - see the LICENSE file for details.

## Acknowledgments
**Streamlit**: Used to create the interactive web application.

**LangChain & Ollama**: Used for natural language processing and sentiment analysis.
