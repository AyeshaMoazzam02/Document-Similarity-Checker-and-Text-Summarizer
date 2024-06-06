import os
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image  # PIL library for image processing
import pytesseract  # Pytesseract for OCR
from jinja2 import Template
import io
import numpy as np
import docx
import re

# Function to extract text from DOCX files
def extract_text_from_docx(file_path):
    text = ""
    try:
        # Check if the filename starts with "~$", indicating a temporary file
        if not os.path.basename(file_path).startswith("~$"):
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
    except Exception as e:
        print("Error extracting text from Word file:", e)
    return text

# Function to extract text from PDF files (including images)
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as pdf:
            for page in pdf:
                # Check if the page contains images
                images = page.get_images(full=True)
                if images:
                    # Extract text from images using OCR
                    for img_index, img in enumerate(images):
                        xref = img[0]
                        base_image = pdf.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(image_bytes))
                        img_text = pytesseract.image_to_string(image)
                        text += img_text + "\n"  # Add a newline after each image's text
                else:
                    # Extract text from text-based pages
                    text += page.get_text()
    except Exception as e:
        print("Error extracting text from PDF:", e)
    return text

# Function to read text from text files
def read_text_from_txt(file_path):
    text = ""
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print("File not found:", file_path)
    return text

# Function to calculate cosine similarity
def calculate_similarity(document_paths):
    documents = []
    document_names = []
    for file_path in document_paths:
        base_name = os.path.basename(file_path)
        if file_path.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_path.endswith('.txt'):
            text = read_text_from_txt(file_path)
        elif file_path.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        else:
            continue
        documents.append(text)
        document_names.append(base_name)

    # Configure CountVectorizer to be less restrictive
    count_vectorizer = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b\w+\b")

    # Try to create a sparse matrix
    try:
        sparse_matrix = count_vectorizer.fit_transform(documents)
    except ValueError as e:
        print("Error during vectorization:", e)
        return None, None

    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(
        doc_term_matrix,
        columns=count_vectorizer.get_feature_names_out(),
        index=document_names
    )
    cosine_sim = cosine_similarity(df, df)
    return cosine_sim, df, document_names, documents

def generate_heatmap(cosine_sim,df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_sim, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=df.index, yticklabels=df.index)
    plt.title('Cosine Similarity Heatmap')
    plt.xlabel('')
    plt.ylabel('')
    heatmap_file = 'static/heatmap.png'  # Save file in static directory accessible by Flask
    plt.savefig(heatmap_file)  # Save heatmap as image
    plt.close()  # Close the plot to free up memory
    return heatmap_file

# HISTOGRAM VISUALIZATION
def generate_histogram(cosine_sim):
    plt.figure(figsize=(8, 6))
    similarity_scores = cosine_sim.flatten()
    plt.hist(similarity_scores, bins=20, color='blue', alpha=0.7)
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of Document Similarity Scores')
    histogram_file = 'static/histogram.png'  # Save file in static directory accessible by Flask
    plt.savefig(histogram_file)  # Save histogram as image
    plt.close()  # Close the plot to free up memory
    return histogram_file

def calculate_similarity_percentage(cosine_sim, document_names):
    num_docs = cosine_sim.shape[0]
    similarities = []

    for i in range(num_docs):
        for j in range(i+1, num_docs):
            similarity_percentage = round(cosine_sim[i, j] * 100, 2)
            similarity_percentage_str = f"{similarity_percentage}%"
            similarities.append((document_names[i], document_names[j], similarity_percentage_str))

    similarity_df = pd.DataFrame(similarities, columns=["Document 1", "Document 2", "Similarity"])
    return similarity_df

def highlight_similarities(text1, text2):
    # Simple example of highlighting common words
    words1 = set(text1.split())
    words2 = set(text2.split())
    common_words = words1 & words2

    for word in common_words:
        text1 = re.sub(r'\b' + re.escape(word) + r'\b', f'<mark>{word}</mark>', text1)
        text2 = re.sub(r'\b' + re.escape(word) + r'\b', f'<mark>{word}</mark>', text2)

    return text1, text2
