# 🤖 AI-Powered Resume Screening and Ranking System

An automated HR-tech intelligence dashboard designed to streamline talent acquisition workflows. This platform extracts unstructured textual content from PDF resumes, translates candidate backgrounds into mathematical vectors, and evaluates profile matches against precise job descriptions using natural language processing (NLP) architectures.

## 🚀 Live Demo
🔗 **Explore the live screening dashboard here:** [AI Resume Screening & Ranking App](https://ai-resume-screening-and-ranking-system-project-cozhfc3yqe3pxld.streamlit.app/)

---

## 📌 Problem Statement & Project Scope
### The Challenge
Hiring teams face major operational bottlenecks when manually filtering through hundreds of applications to extract relevant skills. Unstructured file variations, inconsistent formatting, and subjective manual scoring protocols make traditional screening processes slow, prone to bias, and inefficient.

### The Solution
This system provides an unbiased, algorithmic parsing and ranking application. Built as a capstone framework, it provides immediate alignment insights through semantic calculation engines—instantly distilling candidate scores down to explicit percentage relevance weights.

---

## ✨ System Features
* **Multi-Format Text Extraction:** Fully transparent, stream-based extraction pipelines utilizing `PyPDF2` to strip and clean structural raw data out of varying resume documents.
* **Vectorized Term Weighting:** Deploys a Text Frequency-Inverse Document Frequency (`TF-IDF`) vectorizer matrix to capture structural industry keywords while suppressing common grammatical stop-words.
* **Cosine Proximity Math:** Evaluates geometric angles between multidimensional document vectors to yield clean, normalized matching percentages between 0% and 100%.
* **Enterprise Deliverables Backed:** Contains institutional internship documentation matrices, standardized deployment presentation formats, and model evaluation guidelines.

---

## 🛠️ Tech Stack & Dependencies
* **Dashboard Front-End:** Streamlit (Clean Dataframe Layering)
* **NLP Vector Engine:** Scikit-Learn (`TfidfVectorizer`)
* **Proximity Computations:** Scikit-Learn (`cosine_similarity`)
* **File Processing Layers:** PyPDF2 (PDF String Serialization), Python-docx (`.docx` reference loaders)
* **Data Matrices:** Pandas, NumPy

---

## 📂 Project Structure
```text
├── .devcontainer/                         # Pre-configured isolated cloud workspace architecture
├── Data Analyst jobdescription.docx       # Standard evaluation baseline file for system calibration
├── AICTE_Internship_2024_Project_Report_Template_ convert to pdf before submission.pdf # Institutional validation report
├── Mridul Singh Rajput_certificate.pdf    # Certified validation matrix documents
├── TSP 4.0 Capstone Project PPT Template.pptx # Explanatory capstone technical presentation slide deck
├── AICTE Internship Video.mp4             # Step-by-step operational software walkthrough video
├── resume_app.py                          # Streamlit application portal and vector routing controller
├── requirements.txt                       # Software package configuration requirements
└── README.md                              # Project documentation
