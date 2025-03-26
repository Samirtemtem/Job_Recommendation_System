import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MongoDB setup - Update with your database connection string
client = MongoClient('mongodb+srv://root:root@cluster0.wa1te.mongodb.net/recruitpro?retryWrites=true&w=majority')
db = client['recruitpro']  # Replace with your actual database name if different

# Load data from multiple collections
users_collection = db['users']
profiles_collection = db['profiles']
jobposts_collection = db['jobposts']
skills_collection = db['skills']
educations_collection = db['educations']
experiences_collection = db['experiences']

logging.info("Loading data from MongoDB collections...")

# Load data into DataFrames
users = pd.DataFrame(list(users_collection.find()))
profiles = pd.DataFrame(list(profiles_collection.find()))
jobposts = pd.DataFrame(list(jobposts_collection.find()))
skills = pd.DataFrame(list(skills_collection.find()))
educations = pd.DataFrame(list(educations_collection.find()))
experiences = pd.DataFrame(list(experiences_collection.find()))

logging.info(f"Loaded {len(users)} users, {len(profiles)} profiles, {len(jobposts)} job posts")

# Process profiles to create consolidated candidate data
logging.info("Processing profiles and creating consolidated candidate data...")

candidates = profiles.copy()
if not candidates.empty:
    # Add user information to profiles
    if '_id' in users.columns and 'user' in candidates.columns:
        candidates = candidates.merge(
            users[['_id', 'firstName', 'lastName', 'email', 'phoneNumber', 'role']], 
            left_on='user', 
            right_on='_id', 
            how='left',
            suffixes=('', '_user')
        )
    
    # Process skills
    if not skills.empty and 'skills' in candidates.columns:
        # Create a dictionary mapping skill IDs to skill names
        skill_dict = {}
        for _, row in skills.iterrows():
            if '_id' in row and 'name' in row:
                skill_dict[str(row['_id'])] = row['name']
        
        # Map skill IDs to skill names for each candidate
        def map_skills(skill_ids):
            if not isinstance(skill_ids, list):
                return []
            return [skill_dict.get(str(skill_id), '') for skill_id in skill_ids if str(skill_id) in skill_dict]
        
        candidates['skill_names'] = candidates['skills'].apply(map_skills)
    else:
        candidates['skill_names'] = candidates.apply(lambda x: [], axis=1)
    
    # Process education
    if not educations.empty and 'education' in candidates.columns:
        # Create a dictionary mapping education IDs to education details
        education_dict = {}
        for _, row in educations.iterrows():
            if '_id' in row and 'diploma' in row:
                education_dict[str(row['_id'])] = row['diploma']
        
        # Map education IDs to education details for each candidate
        def map_education(edu_ids):
            if not isinstance(edu_ids, list):
                return []
            return [education_dict.get(str(edu_id), '') for edu_id in edu_ids if str(edu_id) in education_dict]
        
        candidates['education_details'] = candidates['education'].apply(map_education)
    else:
        candidates['education_details'] = candidates.apply(lambda x: [], axis=1)
    
    # Process experience
    if not experiences.empty and 'experience' in candidates.columns:
        # Create a dictionary mapping experience IDs to experience details and years
        experience_dict = {}
        experience_years_dict = {}
        for _, row in experiences.iterrows():
            if '_id' in row and 'position' in row:
                experience_dict[str(row['_id'])] = row['position']
                # Calculate years of experience (simplified)
                if 'startDate' in row and 'endDate' in row and pd.notna(row['startDate']) and pd.notna(row['endDate']):
                    try:
                        start = pd.to_datetime(row['startDate'])
                        end = pd.to_datetime(row['endDate'])
                        years = (end - start).days / 365.25  # Approximate years
                        experience_years_dict[str(row['_id'])] = years
                    except Exception as e:
                        logging.warning(f"Error calculating experience years: {e}")
                        experience_years_dict[str(row['_id'])] = 0
                else:
                    experience_years_dict[str(row['_id'])] = 0
        
        # Map experience IDs to details for each candidate
        def map_experience(exp_ids):
            if not isinstance(exp_ids, list):
                return []
            return [experience_dict.get(str(exp_id), '') for exp_id in exp_ids if str(exp_id) in experience_dict]
        
        # Calculate total years of experience
        def calculate_total_experience(exp_ids):
            if not isinstance(exp_ids, list):
                return 0
            return sum(experience_years_dict.get(str(exp_id), 0) for exp_id in exp_ids)
        
        candidates['experience_details'] = candidates['experience'].apply(map_experience)
        candidates['total_experience_years'] = candidates['experience'].apply(calculate_total_experience)
    else:
        candidates['experience_details'] = candidates.apply(lambda x: [], axis=1)
        candidates['total_experience_years'] = candidates.apply(lambda x: 0, axis=1)

# Process job posts
logging.info("Processing job posts...")
if not jobposts.empty:
    # Extract requirements as skills
    jobposts['extracted_skills'] = jobposts['requirements'].apply(
        lambda x: x if isinstance(x, list) else []
    )

# Rename columns to match the expected format for the recommendation algorithm
# For candidates
candidates = candidates.rename(columns={
    '_id': 'CandidateID',
    'address': 'Location',
    'skill_names': 'Skills',
    'education_details': 'Education',
    'total_experience_years': 'Experience',
    'description': 'JobDescription'
})

# For jobposts
jobposts = jobposts.rename(columns={
    '_id': 'JobID',
    'title': 'JobTitle',
    'experience': 'Experience',
    'requirements': 'Requirements',
    'extracted_skills': 'Skills',
    'description': 'JobDescription'
})

# Ensure both dataframes have the needed columns
required_columns = ['Skills', 'Experience', 'Education', 'Location']
for column in required_columns:
    if column not in candidates.columns:
        candidates[column] = candidates.apply(lambda x: [], axis=1) if column == 'Skills' or column == 'Education' else ''
    if column not in jobposts.columns:
        jobposts[column] = jobposts.apply(lambda x: [], axis=1) if column == 'Skills' else ''

# Ensure skills are in list format
jobposts['Skills'] = jobposts['Skills'].apply(lambda x: x if isinstance(x, list) else ([x] if pd.notna(x) else []))
candidates['Skills'] = candidates['Skills'].apply(lambda x: x if isinstance(x, list) else ([x] if pd.notna(x) else []))

logging.info("Starting feature encoding...")

# Handle categorical features
def combine_and_fit_labelencoder(df1, df2, column):
    # Handle missing values
    df1[column] = df1[column].fillna('Unknown')
    df2[column] = df2[column].fillna('Unknown')
    
    # Convert to string to ensure LabelEncoder works properly
    df1[column] = df1[column].astype(str)
    df2[column] = df2[column].astype(str)
    
    combined_data = pd.concat([df1[column], df2[column]], axis=0).unique()
    le = LabelEncoder()
    le.fit(combined_data)
    return le

label_encoders = {}
for col in ['Experience', 'Location']:
    if col in candidates.columns and col in jobposts.columns:
        try:
            label_encoders[col] = combine_and_fit_labelencoder(candidates, jobposts, col)
            candidates[col] = label_encoders[col].transform(candidates[col])
            jobposts[col] = label_encoders[col].transform(jobposts[col])
        except Exception as e:
            logging.error(f"Error encoding {col}: {e}")
            # Use a default value if encoding fails
            candidates[col] = 0
            jobposts[col] = 0

# Education needs special handling as it might be a list
if 'Education' in candidates.columns and 'Education' in jobposts.columns:
    # Flatten education lists to single string
    candidates['Education_flat'] = candidates['Education'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else str(x)
    )
    jobposts['Education_flat'] = jobposts['Education'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else str(x)
    )
    
    try:
        label_encoders['Education'] = combine_and_fit_labelencoder(
            candidates[['Education_flat']].rename(columns={'Education_flat': 'Education'}),
            jobposts[['Education_flat']].rename(columns={'Education_flat': 'Education'}),
            'Education'
        )
        candidates['Education_encoded'] = label_encoders['Education'].transform(candidates['Education_flat'])
        jobposts['Education_encoded'] = label_encoders['Education'].transform(jobposts['Education_flat'])
    except Exception as e:
        logging.error(f"Error encoding Education: {e}")
        candidates['Education_encoded'] = 0
        jobposts['Education_encoded'] = 0

# Extract all unique skills from both jobposts and candidates
all_skills = []
for skills_list in candidates['Skills'].tolist() + jobposts['Skills'].tolist():
    if isinstance(skills_list, list):
        all_skills.extend(skills_list)
skills_set = list(set(all_skills))

logging.info(f"Found {len(skills_set)} unique skills")

def encode_features(df, feature_set, column_name):
    encoded_features = []
    for features in df[column_name]:
        if not isinstance(features, list):
            features = []
        encoded = [1 if feature in features else 0 for feature in feature_set]
        encoded_features.append(encoded)
    return np.array(encoded_features)

logging.info("Encoding skills...")
jobposts_skills_encoded = encode_features(jobposts, skills_set, 'Skills')
candidates_skills_encoded = encode_features(candidates, skills_set, 'Skills')

# Create feature vectors combining categorical and skills data
feature_columns = []

# Try to include Education and Experience data
if 'Education_encoded' in candidates.columns and 'Education_encoded' in jobposts.columns:
    feature_columns.append('Education_encoded')
elif 'Education' in candidates.columns and 'Education' in jobposts.columns:
    # Try to use Education directly if encoded version not available
    try:
        # Convert to numeric if possible
        candidates['Education_numeric'] = pd.to_numeric(candidates['Education'], errors='coerce').fillna(0)
        jobposts['Education_numeric'] = pd.to_numeric(jobposts['Education'], errors='coerce').fillna(0)
        feature_columns.append('Education_numeric')
    except:
        logging.warning("Could not convert Education to numeric feature")

# Add Experience feature
if 'Experience' in candidates.columns and 'Experience' in jobposts.columns:
    # Make sure Experience is numeric
    candidates['Experience'] = pd.to_numeric(candidates['Experience'], errors='coerce').fillna(0)
    jobposts['Experience'] = pd.to_numeric(jobposts['Experience'], errors='coerce').fillna(0)
    feature_columns.append('Experience')

logging.info(f"Using feature columns: {feature_columns}")

# Add Location if available
if 'Location' in jobposts.columns and 'Location' in candidates.columns:
    if not jobposts['Location'].isnull().all() and not candidates['Location'].isnull().all():
        feature_columns.append('Location')

# Improve skill matching by giving more weight to skills
# This makes skills more important in the matching
skill_weight = 3.0  # Adjust this weight to make skills more/less important

if feature_columns:
    # Prepare categorical features
    jobposts_cat = jobposts[feature_columns].values
    candidates_cat = candidates[feature_columns].values
    
    # Apply skill weight to skill vectors
    jobposts_skills_weighted = jobposts_skills_encoded * skill_weight
    candidates_skills_weighted = candidates_skills_encoded * skill_weight
    
    # Combine categorical and weighted skill features
    jobposts_combined = np.hstack((jobposts_cat, jobposts_skills_weighted))
    candidates_combined = np.hstack((candidates_cat, candidates_skills_weighted))
else:
    # If no categorical features available, just use weighted skills
    jobposts_combined = jobposts_skills_encoded * skill_weight
    candidates_combined = candidates_skills_encoded * skill_weight

# Add debug info
logging.info(f"Feature vector composition: {len(feature_columns)} categorical features + {jobposts_skills_encoded.shape[1]} skills")
logging.info(f"Applied skill weight of {skill_weight}x to increase importance of matching skills")

logging.info("Calculating similarity matrices...")

# Debug information about feature vectors
logging.info(f"Shape of jobposts_combined: {jobposts_combined.shape}")
logging.info(f"Shape of candidates_combined: {candidates_combined.shape}")

# Check if vectors are all zeros (which would result in 0% similarity)
jobposts_zeros = np.sum(np.all(jobposts_combined == 0, axis=1))
candidates_zeros = np.sum(np.all(candidates_combined == 0, axis=1))
logging.info(f"Job posts with all-zero feature vectors: {jobposts_zeros} out of {jobposts_combined.shape[0]}")
logging.info(f"Candidates with all-zero feature vectors: {candidates_zeros} out of {candidates_combined.shape[0]}")

# Sample a few vectors to see their content
if jobposts_combined.shape[0] > 0:
    logging.info(f"Sample job post feature vector: {jobposts_combined[0]}")
    logging.info(f"Non-zero elements: {np.count_nonzero(jobposts_combined[0])}")

if candidates_combined.shape[0] > 0:
    logging.info(f"Sample candidate feature vector: {candidates_combined[0]}")
    logging.info(f"Non-zero elements: {np.count_nonzero(candidates_combined[0])}")

# Calculate similarity matrices
similarity_matrix = cosine_similarity(jobposts_combined, candidates_combined)

# Debug similarity scores
if similarity_matrix.size > 0:
    logging.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    logging.info(f"Min similarity: {np.min(similarity_matrix)}, Max similarity: {np.max(similarity_matrix)}")
    logging.info(f"Average similarity: {np.mean(similarity_matrix)}")
    logging.info(f"Number of zero similarities: {np.sum(similarity_matrix == 0)}")
    
    # Print a sample of the similarity matrix
    if similarity_matrix.shape[0] > 0 and similarity_matrix.shape[1] > 0:
        sample_size = min(5, similarity_matrix.shape[0], similarity_matrix.shape[1])
        logging.info(f"Sample of similarity matrix (5x5 or smaller):\n{similarity_matrix[:sample_size, :sample_size]}")

logging.info("Saving data to pickle files...")
# Save similarity matrix and label encoders to pickle files
with open('similarity_matrix.pkl', 'wb') as f:
    pickle.dump(similarity_matrix, f)
    
with open('candidate_similarity_matrix.pkl', 'wb') as f:
    pickle.dump(similarity_matrix.T, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Save jobposts and candidates data to pickle files for use in app.py
with open('jobposts.pkl', 'wb') as f:
    pickle.dump(jobposts, f)

with open('candidates.pkl', 'wb') as f:
    pickle.dump(candidates, f)

logging.info("Data processing and model creation completed successfully")
print("Data processing and model creation completed successfully")
