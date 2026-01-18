from flask import Flask, request, render_template
import os
import re
import PyPDF2
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= TEXT EXTRACTION =================
def extract_text_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.lower()

def extract_text_docs(file_path):
    return docx2txt.process(file_path).lower()

def extract_text_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().lower()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_docs(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_txt(file_path)
    else:
        return ""

# ================= FLASK APP =================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# ================= INDEX PAGE =================
@app.route('/')
def index():
    return render_template("index.html")

# ================= RANKING PAGE (YOUR PAGE) =================
@app.route('/rank')
def rank_page():
    return render_template("app.html")

# ================= FEATURE 1: RESUME RANKING (UNCHANGED LOGIC) =================
@app.route('/upload', methods=['POST'])
def upload():
    jd = request.form.get('resumeText')
    resume_files = request.files.getlist('resumeFile')

    resumes = []
    for resume_file in resume_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
        resume_file.save(file_path)
        resumes.append(extract_text(file_path))

    if not resumes or not jd:
        return render_template('app.html', message="Please upload asume(s) and job description")

    vec = TfidfVectorizer().fit_transform([jd] + resumes)
    vecs = vec.toarray()
    j_V = vecs[0]
    r_V = vecs[1:]

    sim = cosine_similarity([j_V], r_V)[0]
    top_in = sim.argsort()[-3:][::-1]
    top_r = [resume_files[i].filename for i in top_in]
    similarity_scores = [round(sim[i], 2) for i in top_in]

    return render_template(
        'app.html',
        message="Top matching resumes:",
        top_r=top_r,
        similarity_scores=similarity_scores
    )

# ================= FEATURE 2: ATS SCORE =================
SKILLS = [
    "python","java","c++","machine learning","deep learning",
    "data science","sql","html","css","javascript","flask","django"
]

@app.route('/ats', methods=['GET', 'POST'])
def ats():
    if request.method == 'POST':
        jd = request.form['jd']
        resume = request.files['resume']

        path = os.path.join(app.config['UPLOAD_FOLDER'], resume.filename)
        resume.save(path)

        text = extract_text(path)

        vec = TfidfVectorizer().fit_transform([jd, text])
        score = round(cosine_similarity(vec[0:1], vec[1:2])[0][0] * 100, 2)

        skills_found = [s for s in SKILLS if s in text]
        exp_match = re.findall(r'(\d+)\s+years?', text)
        experience = max([int(x) for x in exp_match], default=0)

        return render_template(
            "ats.html",
            score=score,
            skills=skills_found,
            exp=experience
        )

    return render_template("ats.html")

# ================= FEATURE 3: ROLE PREDICTION =================
ROLE_MAP = {
    "Data Scientist": ["python", "data science", "machine learning"],
    "ML Engineer": ["python", "machine learning", "deep learning"],
    "Web Developer": ["html", "css", "javascript"],
    "Software Engineer": ["java", "c++", "python"]
}

@app.route('/role', methods=['GET', 'POST'])
def role():
    if request.method == 'POST':
        resume = request.files['resume']
        path = os.path.join(app.config['UPLOAD_FOLDER'], resume.filename)
        resume.save(path)

        text = extract_text(path)

        scores = {}
        for role, skills in ROLE_MAP.items():
            scores[role] = sum(1 for s in skills if s in text)

        ranked_roles = sorted(scores, key=scores.get, reverse=True)

        return render_template("role.html", roles=ranked_roles)

    return render_template("role.html")

# ================= RUN SERVER =================
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)
