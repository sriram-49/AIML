# 🎵 Music Genre Classification Using Machine Learning 🎶

This project leverages machine learning techniques to classify music genres based on audio features. It aims to provide an intelligent system capable of automatically predicting the genre of a given music file using trained AI models.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Dataset Used](#dataset-used)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Model and Evaluation](#model-and-evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [License](#license)

---

## 🎯 Project Overview

This AI/ML-based project classifies music files into various genres like Rock, Classical, Jazz, Pop, etc., using extracted features such as MFCCs, chroma, and mel spectrograms. It demonstrates how supervised learning techniques and deep learning models can be applied in audio signal processing.

---

## 🛠️ Tech Stack

- Python 🐍
- Librosa 🎶 (for feature extraction)
- Scikit-learn 🔬
- TensorFlow / Keras 🧠
- Flask 🌐 (for web deployment)
- HTML/CSS + GSAP (for stylish frontend animations)
- SQLite (for data storage)

---

## ✨ Features

- Upload MP3/WAV files
- Extracts features like MFCCs, Spectral Centroid, Zero-Crossing Rate
- Predicts the genre using pre-trained model
- Beautiful web UI with animations (GSAP)
- Displays model confidence scores and prediction

---

## 🎧 Dataset Used

- **GTZAN Genre Collection**  
  - 10 genres
  - 1000 audio tracks (each 30 seconds long)
  - Source: [http://marsyas.info/downloads/datasets.html](http://marsyas.info/downloads/datasets.html)

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music-genre-classification.git
   cd music-genre-classification
