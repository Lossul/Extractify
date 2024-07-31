# Two API keys are being used:
#1) PANDASAI API KEY and 2) HUGGINGFACE API KEY
#Please refresh these keys and copy paste it in the .env file and then use the chatbot if you are facing issues
import fitz
import re
import pandas as pd
from pandasai import SmartDataframe
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import base64
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import RetrievalQA
import tempfile
import pickle
import os
from PIL import Image
import pytesseract
import time
import uuid
from dotenv import load_dotenv
from difflib import SequenceMatcher

# Load environment variables
load_dotenv()
CSV_FILE = "chat_history.csv"

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

def ocr_core(image_file):
    """Perform OCR on the given image file."""
    text = pytesseract.image_to_string(Image.open(image_file))
    return text

def get_button_label(chat_df, chat_id):
    """Generate a button label based on the first user message of a specific chat ID."""
    user_messages = chat_df[(chat_df["ChatID"] == chat_id) & (chat_df["Role"] == "User")]
    
    if user_messages.empty:
        return "No message"
    
    first_message = user_messages.iloc[0]["Content"]
    return f"{' '.join(first_message.split()[:5])}..."  # A summary like text is formed for our button


def save_chat_history(chat_df):
    chat_df.to_csv(CSV_FILE, index=False)
    
def load_chat_history():
    try:
        if os.path.exists(CSV_FILE):
            return pd.read_csv(CSV_FILE)  # Return DataFrame directly
        else:
            return pd.DataFrame(columns=["ChatID", "Role", "Content"])
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        return pd.DataFrame(columns=["ChatID", "Role", "Content"])

def load_pdf(file_path):
    """Load text from a PDF file using PyMuPDF."""
    pdf_document = fitz.open(file_path)
    page_texts = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)  # Use load_page to access the page
        text = page.get_text("text")
        page_texts.append((page_num + 1, text))  # Store page number and text
    return pdf_document, page_texts

def highlight_text_in_pdf(file_path, search_text, page_number):
    pdf_document = fitz.open(file_path)
    page = pdf_document.load_page(page_number - 1)

    # Search for the text and get areas to highlight
    areas = page.search_for(search_text)

    # Add highlights
    for area in areas:
        highlight = page.add_highlight_annot(area)
        highlight.update()

    # Save the highlighted PDF temporarily
    highlighted_pdf_path = "highlighted.pdf"
    pdf_document.save(highlighted_pdf_path)

    # Render the page with highlights
    pix = page.get_pixmap(dpi=120)
    img = pix.tobytes()
    
    # Save the highlighted page as a temporary image file to display
    with open("highlighted_page.png", "wb") as img_file:
        img_file.write(img)
    
    st.image("highlighted_page.png", use_column_width=True)
    
    # Close the PDF document
    pdf_document.close()

    return highlighted_pdf_path
    
def clean_text(text):
    """Clean extracted text."""
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    # Fix concatenated words
    text = re.sub(r'(\w)([A-Z])', r'\1 \2', text)
    # Additional formatting fixes can be added here
    return text

def chat_with_csv(df, prompt):
    os.getenv("PANDASAI_API_KEY")
    pandas_ai = SmartDataframe(df)
    try:
        result = pandas_ai.chat(prompt)
        return result
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

def save_vectorstore(store_name, chunks, embeddings):
    """Save the vectorstore to a pickle file."""
    vectorstore = FAISS.from_texts(chunks, embeddings)
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
    return vectorstore

def load_vectorstore(store_name):
    """Load the vectorstore from a pickle file."""
    with open(f"{store_name}.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    return vectorstore

def stream_response(content):
    """Stream the response word by word."""
    placeholder = st.empty()
    streamed_response = ""

    content = str(content)  # Ensure content is a string

    for word in content.split():
        streamed_response += word + " "
        placeholder.success(streamed_response)
        try:
            time.sleep(0.1)  # Adjust sleep time for faster/slower streaming
        except KeyboardInterrupt:
            break  # Exit loop if user interrupts

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
def extract_relevant_snippet(text, query):
    """Extract the most relevant sentence or paragraph based on the query."""
    # Split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    # Find the sentence that best matches the query
    best_match_ratio = 0
    best_match_sentence = sentences[0]
    for sentence in sentences:
        match_ratio = SequenceMatcher(None, query, sentence).ratio()
        if match_ratio > best_match_ratio:
            best_match_ratio = match_ratio
            best_match_sentence = sentence
    
    return best_match_sentence

def main():
    if "history" not in st.session_state:
        st.session_state.history = []

    add_bg_from_local('bg_image.png')
    with st.sidebar:
        add_bg_from_local('sidebar_bg.jpg')
    left_co, middle_co, right_co = st.columns(3)
    with middle_co:
        st.image("logo.png", width=250)
    st.title(":orange[Extractify]: A :green[RAG] based Chatbot")
    footer = """<style>.footer {position: fixed;left: 0;bottom: 0;width: 100%;background-color: #000;color: white;text-align: center;}
        </style><div class='footer'><p>Developed with ❤️ using Streamlit and Langchain</p></div>"""
    st.markdown(footer, unsafe_allow_html=True)

    # Sidebar for file upload
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your PDF, Image, or CSV", type=['pdf', 'png', 'jpg', 'jpeg', 'csv'], accept_multiple_files=True)
        st.sidebar.write("****************************************")
        st.sidebar.header(":green[**Chit Chats from Before**]")
        chat_history_df = load_chat_history()

        for chat_id in chat_history_df["ChatID"].unique():
            button_label = get_button_label(chat_history_df, chat_id)
            if st.sidebar.button(button_label, key=chat_id):
                loaded_chat = chat_history_df[chat_history_df["ChatID"] == chat_id]
                loaded_chat_string = "\n\n".join(f"{'You' if row['Role'] == 'User' else 'Bot'}: {row['Content']}"
                                                for _, row in loaded_chat.iterrows())
                st.text_area(":green[**Conversation Thread**]", value=loaded_chat_string, height=200)

        clear_button = st.sidebar.button(":red[Clear All]", key="clear_all")
        if clear_button:
            st.session_state.history.clear()
            save_chat_history(pd.DataFrame(columns=["ChatID", "Role", "Content"]))
            st.sidebar.error("*Chat history is currently empty.*")
            st.experimental_rerun()

    text = ""
    df = None
    combined_chunks = []
    
    if uploaded_files is not None:
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                if uploaded_file.type == "application/pdf":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    pdf_document, page_texts = load_pdf(tmp_file_path)
                    cleaned_texts = [(page_num, clean_text(text)) for page_num, text in page_texts]
                    combined_chunks.extend([text for _, text in cleaned_texts])
                
                elif uploaded_file.type in ["image/png", "image/jpeg"]:
                    text = ocr_core(uploaded_file)
                    text = clean_text(text)
                    combined_chunks.append(text)

                elif uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                    with st.expander(":green[**View Some of the Rows**]"):
                        st.write(df.head())

            except Exception as e:
                st.error(f"Error processing the file: {e}")

        if df is None:
            if not combined_chunks:
                st.info("Please upload a file from the sidebar.")
                return
            
            # Combine all text chunks
            text = " ".join(combined_chunks)
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)
            
            if not chunks:
                st.error("No valid text chunks created.")
                return

            # Store file name
            store_name = "combined_files"

            if os.path.exists(f"{store_name}.pkl"):
                vectorstore = load_vectorstore(store_name)
            else:
                try:
                    embeddings = HuggingFaceHubEmbeddings(
                        model="sentence-transformers/all-MiniLM-L6-v2",
                        huggingfacehub_api_token=os.getenv("API_KEY")
                    )
                    vectorstore = save_vectorstore(store_name, chunks, embeddings)
                except Exception as e:
                    st.error(f"Error getting embeddings from HuggingFace API: {e}")
                    return

            # Initialize retriever
            vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
            keyword_retriever = BM25Retriever.from_texts(chunks)
            keyword_retriever.k = 1

            # Set up ensemble retriever with individual retrievers
            ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retriever, keyword_retriever], weights=[0.3, 0.7])

            # User query input
            container = st.container()
            with container:
                with st.form(key='combined_query_form', clear_on_submit=True):
                    query = st.text_input("Ask questions about your uploaded files", placeholder="Ask me something related to the uploaded files")
                    c1, c2 = st.columns([1,8.5])

                    with c1:
                        send = st.form_submit_button(label=":green[Send]")
                    with c2:
                        chat_history = st.form_submit_button(":orange[Chat History]")

                    if query or send:
                        docs = ensemble_retriever.invoke(input=query, top_k=1)
                        try:
                            context = "\n".join([entry["content"] for entry in st.session_state.history[-10:]])
                            combined_input = context + "\nUser: " + query

                            chain = RetrievalQA.from_chain_type(llm=HuggingFaceEndpoint(
                                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                                temperature=0.1,
                                huggingfacehub_api_token=os.getenv("API_KEY")
                            ), chain_type="stuff", retriever=ensemble_retriever)
                            response = chain.invoke(input=combined_input)

                            result = response.get('result', 'No result found.')
                            st.session_state.history.append({"content": f"User: {query}\nBot: {result}"})

                            with st.spinner("In Progress"):
                                stream_response(result)

                            # Generate a unique ChatID for each session
                            chat_id = uuid.uuid4().hex

                            # Load the current chat history DataFrame
                            chat_history_df = load_chat_history()

                            # Append the new conversation to the DataFrame
                            new_data = pd.DataFrame([
                                {"ChatID": chat_id, "Role": "User", "Content": query},
                                {"ChatID": chat_id, "Role": "AI", "Content": result}
                            ])

                            chat_history_df = pd.concat([chat_history_df, new_data], ignore_index=True)

                            # Save the updated chat history DataFrame
                            save_chat_history(chat_history_df)

                            if docs:
                                doc = docs[0]
                                found_page_number = None

                                # Improved logic to find the page number accurately
                                best_match_ratio = 0
                                for page_num, text in cleaned_texts:
                                    match_ratio = SequenceMatcher(None, doc.page_content, text).ratio()
                                    if match_ratio > best_match_ratio:
                                        best_match_ratio = match_ratio
                                        found_page_number = page_num

                                with st.expander(":orange[**See Retrieved Source**]"):
                                    st.warning(doc.page_content)
                                    if found_page_number:
                                        st.warning(f"Source: Page {found_page_number}")

                                        # Extract a relevant snippet from the page content
                                        relevant_snippet = extract_relevant_snippet(doc.page_content, query)

                                        # Highlight the relevant snippet in the PDF
                                        highlighted_pdf_path = highlight_text_in_pdf(tmp_file_path, relevant_snippet, found_page_number)
                                        st.info(f"Highlighted PDF saved at: {highlighted_pdf_path}")
                                    else:
                                        st.warning("Source page not found.")

                        except Exception as e:
                            st.error(f"Error running the question-answering chain: {e}")

        else:
            # Handle CSV-based query
            container = st.container()
            with container:
                with st.form(key='csv_query_form', clear_on_submit=True):
                    query = st.text_input("Ask questions about your CSV file", placeholder="Ask me something related to the uploaded CSV file")
                    c1, c2 = st.columns([1,8.5])

                    with c1:
                        send = st.form_submit_button(label=":green[Send]")
                    with c2:
                        chat_history = st.form_submit_button(":orange[Chat History]")

                    if query or send:
                        context = "\n".join([entry["content"] for entry in st.session_state.history[-10:]])
                        combined_input = context + "\nUser: " + query

                        response = chat_with_csv(df, combined_input)
                        if response:
                            with st.spinner("In Progress"):
                                stream_response(response)
                        else:
                            st.error("No valid response received from the CSV query.")

                        # Update chat history for CSV interactions
                        chat_id = uuid.uuid4().hex
                        chat_history_df = load_chat_history()
                        new_data = pd.DataFrame([
                            {"ChatID": chat_id, "Role": "User", "Content": query},
                            {"ChatID": chat_id, "Role": "AI", "Content": response}
                        ])
                        chat_history_df = pd.concat([chat_history_df, new_data], ignore_index=True)
                        save_chat_history(chat_history_df)

if __name__ == "__main__":
    main()
