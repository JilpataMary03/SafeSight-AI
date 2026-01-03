# Real-Time Face Recognition using InsightFace

This project implements a real-time face registration and recognition system using
InsightFace (buffalo_s), OpenCV, and cosine similarity.

## Features
- Real-time webcam face detection
- Face registration with multiple samples per person
- Embedding averaging for robust recognition
- ROI-based recognition using mouse selection
- Cosine similarity matching
- JSON-based face database

## Tech Stack
- Python
- OpenCV
- InsightFace
- NumPy
- SciPy

## How it works
1. Register a person by capturing multiple face embeddings
2. Average embeddings are stored in `faces.json`
3. During recognition, embeddings are compared using cosine distance
4. Recognition is performed only inside a user-defined ROI

## Usage
```bash
pip install -r requirements.txt
python face_register.py
python face_recognize.py
