Extractify: A RAG based chatbot will take in any of the uploaded file given by the user and frame answers based on the queries related to any PDF document, image or even CSV files. Below are the libraries used for each type of file:
PDF : PyMuPDF(fitz)
IMAGES : pytesseract
CSV : PANDASAI

PANDASAI has been implemented using its API key, while the LLM- MISTRAL 7B has been used via the open source collection of LLMs: HUGGINGFACE.
Both of these API keys can be found in the .env file, and can be updated as and when the api key is invalidated

Happy extracting and remember you are just a prompt away!
