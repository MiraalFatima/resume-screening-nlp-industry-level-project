# File: app.py

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from utils.text_extract import extract_text_from_file
import time

# Set page configuration with modern theme
st.set_page_config(
    page_title="🚀 AI Resume Matcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for futuristic styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #00f5ff, #0080ff, #8000ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
        margin-bottom: 0.5rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(0, 245, 255, 0.3)); }
        to { filter: drop-shadow(0 0 30px rgba(0, 245, 255, 0.8)); }
    }
    
    .sub-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.3rem;
        text-align: center;
        color: #a0a0ff;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .tech-card {
        background: linear-gradient(145deg, rgba(26, 26, 46, 0.8), rgba(22, 33, 62, 0.8));
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 20px rgba(0, 245, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .upload-section {
        border: 2px dashed rgba(0, 245, 255, 0.5);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(145deg, rgba(26, 26, 46, 0.5), rgba(22, 33, 62, 0.5));
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: rgba(0, 245, 255, 0.8);
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .status-success {
        background: linear-gradient(90deg, rgba(0, 255, 127, 0.2), rgba(0, 245, 255, 0.2));
        border-left: 4px solid #00ff7f;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
    }
    
    .status-info {
        background: linear-gradient(90deg, rgba(0, 245, 255, 0.2), rgba(128, 0, 255, 0.2));
        border-left: 4px solid #00f5ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
    }
    
    .results-header {
        font-family: 'Orbitron', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #00f5ff;
        text-align: center;
        margin: 2rem 0 1rem 0;
        text-shadow: 0 0 15px rgba(0, 245, 255, 0.5);
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(0, 245, 255, 0.1), rgba(128, 0, 255, 0.1));
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 245, 255, 0.2);
        border-color: rgba(0, 245, 255, 0.6);
    }
    
    .processing-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }
    
    .loader {
        width: 50px;
        height: 50px;
        border: 3px solid rgba(0, 245, 255, 0.3);
        border-top: 3px solid #00f5ff;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .stDataFrame {
        background: rgba(26, 26, 46, 0.8) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 245, 255, 0.3) !important;
    }
    
    .stDataFrame > div {
        background: rgba(26, 26, 46, 0.8) !important;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stFileUploader > div > div > div > div {
        background: linear-gradient(145deg, rgba(26, 26, 46, 0.8), rgba(22, 33, 62, 0.8));
        border: 2px dashed rgba(0, 245, 255, 0.5);
        border-radius: 15px;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #00f5ff, #0080ff);
        color: #000;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 245, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 245, 255, 0.4);
        background: linear-gradient(45deg, #0080ff, #8000ff);
    }
    
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<h1 class="main-header">🚀 AI RESUME MATCHER</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Neural Network Analysis • Upload Your Resume • Get Precision Matching Scores</p>', unsafe_allow_html=True)

# --- MODEL AND DATA LOADING ---
@st.cache_resource
def load_model_and_data():
    """Loads the model and job data, computing job embeddings once."""
    with st.spinner("🔄 Initializing AI Models..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load and clean dataset
        try:
            job_df = pd.read_csv("data/jobs.csv")
        except FileNotFoundError:
            st.error("⚠️ Error: `data/jobs.csv` not found. Please make sure the file exists.")
            return None, None

        job_df.columns = job_df.columns.str.strip()
        
        # Use the actual column names from your CSV: 'Category' and 'Resume'
        job_df = job_df.rename(columns={
            'Category': 'job_title', 
            'Resume': 'job_description'
        })
        
        # Create a combined text for embedding
        job_df['job_text'] = job_df['job_title'] + ". " + job_df['job_description']
        
        # Compute embeddings for all job descriptions
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        embeddings = []
        for i, text in enumerate(job_df['job_text']):
            embedding = model.encode(text, convert_to_tensor=True)
            embeddings.append(embedding)
            progress = (i + 1) / len(job_df)
            progress_bar.progress(progress)
            status_text.text(f"Processing job descriptions... {i+1}/{len(job_df)}")
        
        job_df['job_embedding'] = embeddings
        progress_bar.empty()
        status_text.empty()
        
        return model, job_df

# Initialize system
st.markdown('<div class="status-info">🤖 <strong>SYSTEM STATUS:</strong> Initializing Neural Networks...</div>', unsafe_allow_html=True)
model, job_df = load_model_and_data()

if model is not None and job_df is not None:
    st.markdown('<div class="status-success">✅ <strong>SYSTEM READY:</strong> AI Models Loaded Successfully</div>', unsafe_allow_html=True)
    
    # Display system metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: #00f5ff; margin: 0;">🎯 MODEL</h3>
            <p style="margin: 0.5rem 0; font-family: 'Rajdhani', sans-serif;">SentenceTransformer</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: #00f5ff; margin: 0;">📊 JOBS LOADED</h3>
            <p style="margin: 0.5rem 0; font-family: 'Rajdhani', sans-serif; font-size: 1.5rem; font-weight: 700;">{len(job_df)}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: #00f5ff; margin: 0;">🧠 STATUS</h3>
            <p style="margin: 0.5rem 0; font-family: 'Rajdhani', sans-serif; color: #00ff7f;">ACTIVE</p>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # File Upload Section
    st.markdown('''
    <div class="tech-card">
        <h2 style="color: #00f5ff; font-family: 'Orbitron', monospace; text-align: center; margin-bottom: 1rem;">
            📄 RESUME UPLOAD PORTAL
        </h2>
        <p style="text-align: center; color: #a0a0ff; margin-bottom: 2rem;">
            Drag & drop your resume or click to browse • Supported formats: PDF, DOCX
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=["pdf", "docx"],
        help="Upload your resume in PDF or DOCX format for AI analysis"
    )

    if uploaded_file is not None:
        st.markdown('<div class="status-info">📂 <strong>FILE DETECTED:</strong> Processing resume...</div>', unsafe_allow_html=True)
        
        try:
            # Show processing animation
            with st.spinner("🔍 Analyzing Resume with AI..."):
                time.sleep(1)  # Small delay for dramatic effect
                
                # Extract text directly from the uploaded file buffer
                resume_text = extract_text_from_file(uploaded_file)
                
                # Show text extraction success
                st.markdown('<div class="status-success">✅ <strong>TEXT EXTRACTION:</strong> Complete</div>', unsafe_allow_html=True)
                
                with st.spinner("🧠 Computing Neural Embeddings..."):
                    time.sleep(0.5)
                    # Encode the resume text
                    resume_emb = model.encode(resume_text, convert_to_tensor=True)

                st.markdown('<div class="status-success">✅ <strong>EMBEDDING GENERATION:</strong> Complete</div>', unsafe_allow_html=True)

                # --- CALCULATE AND DISPLAY SCORES ---
                with st.spinner("⚡ Matching Against Job Database..."):
                    scores = []
                    for index, row in job_df.iterrows():
                        job_emb = row['job_embedding']
                        score = util.pytorch_cos_sim(resume_emb, job_emb).item() * 100
                        scores.append({
                            "🎯 Job Title": row['job_title'],
                            "📊 Match Score": f"{score:.2f}%",
                            "Score": score  # Keep numeric score for sorting
                        })

                    # Create a DataFrame from the scores
                    results_df = pd.DataFrame(scores)
                    results_df = results_df.sort_values(by="Score", ascending=False).drop(columns="Score")

                # Display Results
                st.markdown('<h2 class="results-header">🎯 MATCHING ANALYSIS COMPLETE</h2>', unsafe_allow_html=True)
                
                # Show top match highlight
                if len(results_df) > 0:
                    top_match = results_df.iloc[0]
                    st.markdown(f'''
                    <div class="tech-card" style="border-color: #00ff7f; box-shadow: 0 0 30px rgba(0, 255, 127, 0.3);">
                        <h3 style="color: #00ff7f; text-align: center; margin: 0;">🏆 TOP MATCH</h3>
                        <h2 style="color: #ffffff; text-align: center; margin: 0.5rem 0;">{top_match["🎯 Job Title"]}</h2>
                        <h1 style="color: #00ff7f; text-align: center; margin: 0; font-size: 3rem;">{top_match["📊 Match Score"]}</h1>
                    </div>
                    ''', unsafe_allow_html=True)

                st.markdown('<div class="tech-card">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #00f5ff; text-align: center; margin-bottom: 1rem;">📈 COMPLETE RESULTS MATRIX</h3>', unsafe_allow_html=True)
                
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<div style="background: linear-gradient(90deg, rgba(255, 0, 0, 0.2), rgba(255, 100, 100, 0.2)); border-left: 4px solid #ff0000; padding: 1rem; border-radius: 8px; margin: 1rem 0;">⚠️ <strong>ERROR:</strong> {str(e)}</div>', unsafe_allow_html=True)

else:
    st.markdown('<div style="background: linear-gradient(90deg, rgba(255, 0, 0, 0.2), rgba(255, 100, 100, 0.2)); border-left: 4px solid #ff0000; padding: 1rem; border-radius: 8px; margin: 1rem 0;">❌ <strong>SYSTEM ERROR:</strong> Failed to initialize AI models</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('''
<div style="text-align: center; color: #666; font-family: 'Rajdhani', sans-serif; padding: 2rem;">
    <p>🚀 Powered by Advanced AI • Neural Network Analysis • Professional Resume Matching</p>
</div>
''', unsafe_allow_html=True)
