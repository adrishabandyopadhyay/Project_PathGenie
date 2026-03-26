from flask import Flask, render_template, request
import joblib
import numpy as np
from utils.preprocess import clean_text
import PyPDF2
import nltk
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

app = Flask(__name__)

# MBTI to Career mapping
MBTI_CAREER_MAP = {
    'INTJ': ['Data Scientist', 'Software Architect', 'Research Scientist', 'Cybersecurity Analyst'],
    'INTP': ['Machine Learning Engineer', 'Software Developer', 'Data Analyst', 'Research Scientist'],
    'ENTJ': ['Project Manager', 'Business Analyst', 'Management Consultant', 'Product Manager'],
    'ENTP': ['Entrepreneur', 'Product Manager', 'Marketing Strategist', 'UX Designer'],
    'INFJ': ['Psychologist', 'Content Writer', 'HR Manager', 'UX Researcher'],
    'INFP': ['Graphic Designer', 'Content Creator', 'Social Worker', 'Writer'],
    'ENFJ': ['HR Manager', 'Teacher', 'Public Relations', 'Marketing Manager'],
    'ENFP': ['Marketing Specialist', 'Journalist', 'UX Designer', 'Event Manager'],
    'ISTJ': ['Accountant', 'Database Administrator', 'Financial Analyst', 'Civil Engineer'],
    'ISFJ': ['Nurse', 'Social Worker', 'Administrative Manager', 'Teacher'],
    'ESTJ': ['Operations Manager', 'Financial Advisor', 'Logistics Manager', 'Administrator'],
    'ESFJ': ['Event Planner', 'Nurse', 'HR Coordinator', 'Sales Manager'],
    'ISTP': ['Mechanical Engineer', 'Network Engineer', 'Data Analyst', 'Forensic Scientist'],
    'ISFP': ['Fashion Designer', 'Photographer', 'Graphic Designer', 'Veterinarian'],
    'ESTP': ['Sales Executive', 'Entrepreneur', 'Police Officer', 'Sports Coach'],
    'ESFP': ['Actor', 'Event Planner', 'Sales Representative', 'Tour Guide'],
}

RESUME_MODEL_PATH = 'models/resume_model.pkl'
PERSONALITY_MODEL_PATH = 'models/personality_model_logistic.pkl'
RESUME_VECTORIZER_PATH = 'utils/resume_vectorizer.pkl'
PERSONALITY_VECTORIZER_PATH = 'utils/personality_vectorizer.pkl'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    resume_model = joblib.load(RESUME_MODEL_PATH)
    personality_model = joblib.load(PERSONALITY_MODEL_PATH)
    resume_vectorizer = joblib.load(RESUME_VECTORIZER_PATH)
    personality_vectorizer = joblib.load(PERSONALITY_VECTORIZER_PATH)

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/assess')
def home():
    return render_template('assessment.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        interests = request.form.getlist('interests')
        skills = request.form.getlist('skills')
        user_text = request.form.get('user_text', '')
        combined_text = " ".join(interests + skills + [user_text])

        personality_prediction = None
        if combined_text.strip():
            cleaned_text = clean_text(combined_text)
            vectorized_text = personality_vectorizer.transform([cleaned_text])

            try:
                probs = personality_model.predict_proba(vectorized_text)[0]
                top_idx = np.argsort(probs)[::-1][:4]
                classes = personality_model.classes_

                # Get top MBTI type and map to careers
                top_mbti = classes[top_idx[0]]
                careers = MBTI_CAREER_MAP.get(top_mbti, ['Software Developer', 'Data Analyst', 'Project Manager', 'Business Analyst'])

                personality_prediction = {
                    'primary': {'name': careers[0], 'match': round(probs[top_idx[0]] * 100, 1)},
                    'alternates': [
                        {'name': careers[i], 'match': round((probs[top_idx[0]] * 100 - (i * 5)), 1)}
                        for i in range(1, len(careers))
                    ]
                }
            except Exception:
                pred = personality_model.predict(vectorized_text)[0]
                careers = MBTI_CAREER_MAP.get(pred, ['Software Developer', 'Data Analyst', 'Project Manager'])
                personality_prediction = {
                    'primary': {'name': careers[0], 'match': 85.0},
                    'alternates': [
                        {'name': careers[i], 'match': round(85.0 - (i * 5), 1)}
                        for i in range(1, len(careers))
                    ]
                }

        # Handle Resume Upload
        resume_prediction = None
        if 'resume' in request.files:
            file = request.files['resume']
            if file and file.filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(file)
                cleaned_resume_text = clean_text(resume_text)
                vectorized_resume = resume_vectorizer.transform([cleaned_resume_text])

                try:
                    probs = resume_model.predict_proba(vectorized_resume)[0]
                    top_idx = np.argsort(probs)[::-1][:4]
                    classes = resume_model.classes_
                    resume_prediction = {
                        'primary': {
                            'name': classes[top_idx[0]],
                            'match': round(probs[top_idx[0]] * 100, 1)
                        },
                        'alternates': [
                            {
                                'name': classes[top_idx[i]],
                                'match': round(probs[top_idx[i]] * 100, 1)
                            }
                            for i in range(1, 4) if i < len(classes)
                        ]
                    }
                except Exception:
                    pred = resume_model.predict(vectorized_resume)[0]
                    resume_prediction = {
                        'primary': {'name': pred, 'match': 85.0},
                        'alternates': []
                    }

        combined_prediction = None
        if personality_prediction and resume_prediction:
            combined_scores_map = {}
            
            for pred in [personality_prediction, resume_prediction]:
                if pred and 'primary' in pred:
                    name = str(pred['primary']['name'])
                    score = float(pred['primary']['match'])
                    combined_scores_map.setdefault(name, []).append(score)
                
                if pred and 'alternates' in pred:
                    for alt in pred['alternates']:
                        name = str(alt['name'])
                        score = float(alt['match'])
                        combined_scores_map.setdefault(name, []).append(score)
            
            combined_scores = {}
            for name, scores in combined_scores_map.items():
                # If a career appeared in both lists, give it a significant boost
                if len(scores) > 1:
                    final_score = min(99.1, max(scores) + 8.5)
                else:
                    # Otherwise, keep its score and give a slight confidence curve
                    final_score = min(96.5, scores[0] + 3.0)
                combined_scores[name] = final_score
            
            if combined_scores:
                sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                combined_prediction = {
                    'primary': {'name': str(sorted_combined[0][0]), 'match': float(round(sorted_combined[0][1], 1))},
                    'alternates': [{'name': str(name), 'match': float(round(score, 1))} for name, score in sorted_combined[1:4]]
                }

        return render_template('result.html',
                               personality_prediction=personality_prediction,
                               resume_prediction=resume_prediction,
                               combined_prediction=combined_prediction)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('result.html',
                               personality_prediction=None,
                               resume_prediction=None,
                               error=str(e))

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

