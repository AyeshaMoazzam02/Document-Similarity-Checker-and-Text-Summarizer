from flask import Flask, render_template, request, jsonify, url_for, make_response, render_template_string, redirect
from werkzeug.utils import secure_filename
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from nltk.tokenize import sent_tokenize
import os
import numpy as np
from services.similarity import calculate_similarity, generate_heatmap, generate_histogram,calculate_similarity_percentage, highlight_similarities
import logging
import shutil
import pdfkit
import nltk
nltk.download('punkt')
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from database import db, init_app
from models import User

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/' 
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

init_app(app)

logging.basicConfig(level=logging.INFO)

model = "F:\SE\FYP\Summarization\checkpoint-120000"  
tokenizer = PegasusTokenizer.from_pretrained(model)
model = PegasusForConditionalGeneration.from_pretrained(model)

@app.route("/summarize", methods=["POST"])
def summarize_text():
    input_text = request.json.get("text")
    
    # Check if the text is too long
    if len(input_text.split()) > 6500:  # Roughly 15 pages * 500 words/page
        return jsonify({"error": "Text too long."}), 400

    # Segment the text into manageable parts
    sentences = sent_tokenize(input_text)
    segments = []
    current_segment = ""

    for sentence in sentences:
        if len(tokenizer.encode(current_segment + " " + sentence, add_special_tokens=True)) > 512:
            segments.append(current_segment)
            current_segment = sentence
        else:
            current_segment += (" " + sentence)
    if current_segment:
        segments.append(current_segment)

    # Summarize each segment and combine the summaries
    summarized_text = []
    for segment in segments:
        tokens = tokenizer(segment, return_tensors='pt', truncation=True, max_length=512)
        summary_ids = model.generate(tokens['input_ids'], max_length=128, num_beams=8, early_stopping=True)
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summarized_text.append(output)

    summary_text = " ".join(summarized_text)
    return jsonify({"summary": summary_text})

def clean_upload_folder():
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # This removes each file
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # This removes directories
    logging.info("Upload folder has been cleaned up.")

@app.route('/upload-files', methods=['POST'])
def upload_files():
    files = request.files.getlist('file[]')
    document_paths = []

    for file in files:
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            document_paths.append(file_path)
            logging.info(f"Saved file {filename} at {file_path}")

    similarity_results, df, document_names, documents = calculate_similarity(document_paths)
    if similarity_results is not None and df is not None:
        heatmap_path = generate_heatmap(similarity_results, df)
        histogram_path = generate_histogram(similarity_results)
        similarity_index = np.mean(similarity_results.flatten())
        similarity_percentage = calculate_similarity_percentage(similarity_results, document_names)
        if similarity_percentage is not None:
            similarity_percentage_data = similarity_percentage.to_dict(orient='records') 

        clean_upload_folder()  # Clean up after processing

        return jsonify({
            'heatmap_image': url_for('static', filename=os.path.basename(heatmap_path)),
            'histogram_image': url_for('static', filename=os.path.basename(histogram_path)),
            'similarity_index': similarity_index,
            'document_names': document_names,
            'similarity_percentage' : similarity_percentage_data
        })
    else:
        logging.error("No valid similarity results generated.")
        return jsonify({'error': 'Failed to generate similarity results.'}), 500

@app.route('/download-report', methods=['POST'])
def download_report():
    data = request.get_json()
    similarity_percentages = data.get('similarity_percentages', [])
    print(similarity_percentages)
    # Ensure the paths are correct; this is a sample, modify as needed
    heatmap_path = url_for('static', filename='heatmap.png', _external=True)
    histogram_path = url_for('static', filename='histogram.png', _external=True)
    rendered = render_template_string('''
    <html>
    <head><title>Document Similarity Report</title></head>
    <body>
        <h1>Document Similarity Report</h1>
        <p style="font-size: 25px;">Percentage Based Scores</p>
        <ul style="font-size: 20px;">
            {{similarity_percentages}}
        </ul>
        <img src="{{ heatmap_path }}" alt="Cosine Similarity Heatmap">
        <img src="{{ histogram_path }}" alt="Histogram of Document Similarity Scores">
        <p style="font-size: 25px;">Similarity Index: {{ similarity_index }}</p>
    </body>
    </html>
    ''', heatmap_path=heatmap_path, histogram_path=histogram_path, similarity_index=data['similarity_index'], similarity_percentages=similarity_percentages)

    pdf = pdfkit.from_string(rendered, False)

    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=report.pdf'
    return response

@app.route("/", methods=["GET", "POST"])
def signIn():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            return redirect(url_for('Home'))
        else:
            return 'Invalid email or password'
    return render_template("signIn.html")

@app.route("/signUp", methods=["GET", "POST"])
def signUp():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            return "Passwords do not match!"

        hashed_password = generate_password_hash(password)
        new_user = User(first_name=first_name, last_name=last_name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('Home'))
    return render_template("signUp.html")

@app.route("/home", methods=["GET", "POST"])
def Home():
    return render_template("home.html")

@app.route("/similarity", methods=["GET", "POST"])
def Similarity():
    return render_template("similarity.html")

@app.route("/summarization", methods=["GET", "POST"])
def Summarization():
    return render_template("summarization.html")

@app.route("/about", methods=["GET", "POST"])
def About():
    return render_template("about.html")

@app.route("/pp", methods=["GET", "POST"])
def PP():
    return render_template("pp.html")

if __name__ == '__main__':
    app.run(debug=True)
