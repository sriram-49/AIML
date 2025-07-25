import numpy as np
import librosa
from tensorflow.keras.models import load_model

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

def predict(file_path, model_path="D:\Java prg\project\AIML\Project(AIML)\models\genre_model.h5", max_pad_len=130):
    model = load_model(r"D:\Java prg\project\AIML\Project(AIML)\models\genre_model.h5")
    signal, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)  # Corrected the way we call mfcc()

    # Pad or truncate MFCCs to ensure consistent input length
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    mfcc = mfcc.T  # Transpose to shape (time_steps, n_mfcc)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction)
    return GENRES[predicted_index]

if __name__ == "__main__":
    genre = predict("D:\Java prg\project\AIML\Project(AIML)\data\gtzan\genres_original\hiphop\hiphop.00000.wav")
    print(f"Predicted Genre: {genre}")
