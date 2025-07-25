import os
import numpy as np
import librosa
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ABSOLUTE PATHS - EXACTLY AS YOU SPECIFIED
UPLOAD_FOLDER = r"D:\Java prg\project\AIML\Project(AIML)\models\uploads"
MODEL_PATH = r"D:\Java prg\project\AIML\Project(AIML)\models\genre_model.h5"
TEMPLATE_FOLDER = r"D:\Java prg\project\AIML\Project(AIML)\models\templates"

# Configure app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = 'your-secret-key-123'  # Required for flash messages

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)

# Allowed audio extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

# Debug print paths
print("\n=== PATH VERIFICATION ===")
print(f"Upload folder: {UPLOAD_FOLDER} | Exists: {os.path.exists(UPLOAD_FOLDER)}")
print(f"Model path: {MODEL_PATH} | Exists: {os.path.exists(MODEL_PATH)}")
print(f"Templates folder: {TEMPLATE_FOLDER} | Exists: {os.path.exists(TEMPLATE_FOLDER)}")
print("=======================\n")

# Load model with error handling
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_genre(file_path):
    try:
        print(f"\nProcessing file: {file_path}")
        y, sr = librosa.load(file_path, sr=22050, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Pad/truncate to 130 frames
        if mfcc.shape[1] < 130:
            mfcc = np.pad(mfcc, ((0, 0), (0, 130 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :130]

        mfcc = mfcc.T[np.newaxis, ..., np.newaxis]
        prediction = model.predict(mfcc)
        genre = GENRES[np.argmax(prediction)]
        print(f"Predicted genre: {genre}")
        return genre
    except Exception as e:
        print(f"Prediction error: {e}")
        raise

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print("\nForm submitted! Checking file...")
        
        if 'file' not in request.files:
            flash('No file part in request', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                print(f"Saving to: {file_path}")
                file.save(file_path)
                
                if not model:
                    flash('Model failed to load', 'error')
                    return redirect(request.url)
                
                genre = predict_genre(file_path)
                return render_template("index.html", 
                                     genre=genre, 
                                     filename=filename,
                                     success="File processed successfully!")
                
            except Exception as e:
                flash(f'Processing error: {str(e)}', 'error')
                print(f"ERROR: {str(e)}")
                return redirect(request.url)
        else:
            flash('Invalid file type. Allowed: .wav, .mp3, .ogg, .flac', 'error')
            return redirect(request.url)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)