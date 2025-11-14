import streamlit as st
# BEST PRACTICE: Using the modern 'pypdf' library instead of 'PyPDF2'
from pypdf import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to extract text from pdf
def extract_text_from_pdf(file):
    # The file object passed by Streamlit needs to be read.
    # pypdf can handle the file-like object directly.
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() if page.extract_text() else ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {file.name}. Ensure it is not corrupted. Error: {e}")
        return ""

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    if not resumes:
        return np.array([])
        
    # FIX 1: Correctly define the documents list for TF-IDF
    # The job description must be the first document for proper vector indexing
    documents = [job_description] + resumes
    
    # Check if we have documents to process
    if len(documents) < 2:
        return np.array([])

    # Create and transform the documents
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    
    # Convert vectors to array (optional but useful for consistency)
    vectors_array = vectors.toarray()

    # Calculate cosine similarity
    # Job description is at index 0
    job_description_vector = vectors_array[0].reshape(1, -1)
    # Resumes start from index 1
    resume_vectors = vectors_array[1:]
    
    # Calculate similarities and FIX 2: Corrected typo from .flatter to .flatten()
    cosine_similarities = cosine_similarity(job_description_vector, resume_vectors).flatten()

    return cosine_similarities

# Streamlit app
st.title("AI Resume Screening and Ranking System")
st.markdown("Use this tool to rank resumes based on their relevance to a job description.")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description here (text or bullet points work well):", height=200)

# File Uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files for screening", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.markdown("---")
    st.header("Ranking Results")
    
    # Check for empty job description
    if not job_description.strip():
        st.warning("Please enter a non-empty Job Description to start the ranking.")
    elif len(uploaded_files) == 0:
        st.warning("Please upload at least one PDF file to screen.")
    else:
        # 1. Extract text from resumes
        resumes = []
        resume_names = []
        with st.spinner('Extracting text from resumes...'):
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                if text: # Only append if extraction was successful
                    resumes.append(text)
                    resume_names.append(file.name)

        if not resumes:
            st.error("Could not extract readable text from any uploaded PDF. Please check the files.")
        else:
            # 2. Rank resumes
            with st.spinner('Calculating relevance scores...'):
                scores = rank_resumes(job_description, resumes)

            # 3. Display Scores
            results = pd.DataFrame({
                "Resume": resume_names, 
                "Relevance Score (0-1)": scores
            })
            
            # Format and Sort
            results['Relevance Score (0-1)'] = results['Relevance Score (0-1)'].map('{:.4f}'.format)
            results = results.sort_values(by="Relevance Score (0-1)", ascending=False, key=lambda x: x.astype(float))
            
            st.dataframe(
                results,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Relevance Score (0-1)": st.column_config.ProgressColumn(
                        "Relevance Score (0-1)",
                        help="Cosine similarity score (0.00 to 1.00)",
                        format="%f",
                        min_value=0.0,
                        max_value=1.0,
                    ),
                },
            )

            st.success("Ranking complete!")

elif uploaded_files or job_description:
    st.info("Please provide both a Job Description and upload at least one resume to run the screening.")
