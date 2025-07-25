import os
import librosa
import numpy as np

# Updated dataset and features paths
DATASET_PATH = r"D:\Java prg\project\AIML\Project(AIML)\data\gtzan\genres_original"
FEATURES_PATH = r"D:\Java prg\project\AIML\Project(AIML)\features"

# List genres from the dataset directory
GENRES = os.listdir(DATASET_PATH)

def extract_and_save_features():
    # Create features directory if it doesn't exist
    if not os.path.exists(FEATURES_PATH):
        os.makedirs(FEATURES_PATH)

    # Iterate through each genre folder
    for genre in GENRES:
        genre_path = os.path.join(DATASET_PATH, genre)
        
        # Iterate through each file in the genre folder
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)
            
            # Check if the file is an audio file (e.g., .wav)
            if file.lower().endswith('.wav'):
                try:
                    # Load the audio file
                    signal, sr = librosa.load(file_path, sr=22050)
                    
                    # Extract MFCC features from the audio signal
                    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
                    mfcc = mfcc.T  # Transpose to match shape (time_steps, 13)
                    
                    # Get genre label (index of the genre in the list)
                    label = GENRES.index(genre)
                    
                    # Define the save path for the extracted features
                    save_path = os.path.join(FEATURES_PATH, f"{genre}_{file.replace('.wav', '.npy')}")
                    
                    # Save the MFCC features and label as a .npy file
                    np.save(save_path, {'mfcc': mfcc, 'label': label})
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

if __name__ == "__main__":
    extract_and_save_features()
