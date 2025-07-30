# File: match.py

import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
# Ensure you have a 'utils' folder with 'text_extract.py' inside
from utils.text_extract import extract_text_from_file 

def run_matching():
    """
    Screens all resumes in 'data/resumes' against all jobs in 'data/jobs.csv'.
    """
    print("🚀 Starting Resume Matcher...")

    # --- Load Model (can be slow on first run) ---
    print("\n⏳ Loading the machine learning model...")
    print("   (This may take a few minutes on the first run as it downloads the model).")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Critical Error: Failed to load the Sentence Transformer model: {e}")
        print("   Please check your internet connection and that 'sentence-transformers' is installed correctly.")
        return

    # --- Load and Prepare Job Data ---
    job_data_path = "data/jobs.csv"
    print(f"\n⏳ Loading job data from '{job_data_path}'...")
    try:
        job_df = pd.read_csv(job_data_path)
        print("✅ Job data loaded.")
    except FileNotFoundError:
        print(f"❌ Error: Job data file not found at '{job_data_path}'.")
        print(f"   Please make sure you are running this script from the 'resume-screening-nlp' root folder and the file exists.")
        return

    # --- IMPORTANT: Column Renaming & Validation ---
    # These are the *original* column names from your CSV file.
    # We will rename them to the names the script uses internally.
    original_col_names = {
        'Category': 'job_title', 
        'Resume': 'job_description'
    }
    
    # Check if the expected original columns exist
    if not all(col in job_df.columns for col in original_col_names.keys()):
        print("❌ Error: The CSV file does not have the expected columns.")
        print(f"   The script expects columns named: {list(original_col_names.keys())}")
        print(f"   But your file only has these columns: {job_df.columns.to_list()}")
        print("   Please update the 'original_col_names' dictionary in this script to match your CSV.")
        return

    job_df = job_df.rename(columns=original_col_names)
    job_df['job_text'] = job_df['job_title'].astype(str) + ". " + job_df['job_description'].astype(str)
    
    # --- Pre-calculate Job Embeddings ---
    print("\n⏳ Calculating embeddings for all job descriptions...")
    job_embeddings = model.encode(job_df['job_text'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    print("✅ Job embeddings are ready.")

    # --- Process Resumes ---
    resume_folder = "data/resumes"
    print(f"\n📂 Looking for resumes in '{resume_folder}' folder...")
    if not os.path.exists(resume_folder):
        print(f"   -> The folder '{resume_folder}' doesn't exist. Please create it and add your resume files.")
        return
        
    resume_files = [f for f in os.listdir(resume_folder) if f.lower().endswith(('.pdf', '.docx'))]
    if not resume_files:
        print(f"   -> No .pdf or .docx resumes found in '{resume_folder}'.")
        return

    print(f"   Found {len(resume_files)} resume(s) to process.")
    for resume_file in resume_files:
        resume_path = os.path.join(resume_folder, resume_file)
        print(f"\n--- Processing: {resume_file} ---")
        
        try:
            resume_text = extract_text_from_file(resume_path)
            resume_emb = model.encode(resume_text, convert_to_tensor=True)
            
            # Calculate scores against all jobs at once
            all_scores = util.pytorch_cos_sim(resume_emb, job_embeddings)[0] * 100
            
            # Create a DataFrame for easy sorting and display
            results_df = pd.DataFrame({
                'Job Title': job_df['job_title'],
                'Score': all_scores.cpu().numpy() # Move tensor to CPU and convert to numpy
            })
            
            # Get the top 5 matches
            top_matches = results_df.sort_values(by="Score", ascending=False).head(5)
            
            print("🎯 Top 5 Matches:")
            for index, row in top_matches.iterrows():
                print(f"   - {row['Job Title']}: {row['Score']:.2f}%")

        except Exception as e:
            print(f"   ❌ An error occurred while processing {resume_file}: {e}")

if __name__ == "__main__":
    run_matching()