# Job Recommendation System

A recommendation system that matches job postings with candidate profiles based on skills, experience, and education.

## Overview

This system leverages machine learning to create personalized job recommendations for candidates. It analyzes profiles across multiple collections in MongoDB and calculates similarity scores to suggest the most relevant jobs for each candidate.

## Features

- **Job Recommendations for Candidates**: Suggests suitable job postings based on candidate profiles
- **Similar Candidates**: Identifies similar candidates for each job posting
- **Interactive Web Interface**: Simple web UI to input a candidate ID and view recommendations

## Database Structure

The system works with the following MongoDB collections:

- **users**: Basic user information and authentication details
- **profiles**: Candidate profiles with references to education, experience, and skills
- **jobposts**: Available job positions with requirements and descriptions
- **skills**: Skills with proficiency levels
- **educations**: Educational backgrounds
- **experiences**: Work experience records
- **sociallinks**: Social media profile links

## How It Works

1. **Data Collection**:
   - Loads data from MongoDB collections
   - Joins related collections to create comprehensive candidate and job profiles

2. **Feature Engineering**:
   - Extracts and processes skills from both candidates and job posts
   - Encodes categorical features (education, experience, location)
   - Represents skills as binary vectors

3. **Similarity Calculation**:
   - Combines encoded features to create feature vectors
   - Calculates cosine similarity between job and candidate vectors

4. **Recommendation Generation**:
   - Recommends jobs to candidates based on similarity scores
   - Identifies similar candidates for job positions

## Technical Details

- **Backend**: Python, Flask
- **Database**: MongoDB
- **ML Libraries**: scikit-learn, pandas, numpy
- **Similarity Algorithm**: Cosine similarity
- **Web Interface**: HTML, CSS, Jinja2 templates

## How Recommendations Are Calculated

The recommendation scores in the system are calculated using the following process:

1. **Feature Vector Creation**:
   - Each candidate and job post is represented as a numerical vector
   - Vectors contain encoded categorical data (experience, education, location)
   - Skills are represented as binary values (1 = has skill, 0 = doesn't have skill)
   - A weight multiplier (default: 3.0) is applied to skills to prioritize skill matching

2. **Cosine Similarity**:
   - The system calculates cosine similarity between job and candidate vectors
   - Cosine similarity measures the angle between vectors, not magnitude
   - Values range from 0 (completely different) to 1 (identical)
   - Final scores are presented as percentages (e.g., 0.75 = 75% match)

3. **Ranking**:
   - Job recommendations are ranked by similarity score in descending order
   - The top 5 jobs with highest similarity scores are presented
   - Similar candidates are identified using the transposed similarity matrix

4. **Factors Affecting Scores**:
   - Skill overlap significantly impacts similarity due to weighting
   - Limited skill overlap results in lower similarity scores
   - Consistent skill terminology improves matching accuracy
   - Categorical features (experience, education) provide baseline similarity

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/job-recommendation-system.git
   cd job-recommendation-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up MongoDB connection:
   - Ensure MongoDB is running
   - Update the connection string in `main.py` and `app.py` if needed

## Usage

1. First, generate the recommendation model:
   ```
   python main.py
   ```

2. Start the Flask application:
   ```
   python app.py
   ```

3. Access the web interface at `http://localhost:5000`

4. Enter a candidate ID to get personalized job recommendations

## Extending the System

To add more features to the recommendation system:

1. **Real-time updates**: Implement webhooks to update recommendations when profiles or jobs change
2. **Advanced filtering**: Add filters for job type, salary range, or location preferences
3. **Feedback mechanism**: Collect user feedback to improve recommendation accuracy 