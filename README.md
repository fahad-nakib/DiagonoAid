# DiagonoAid

**Blood Disease Risk Assessment from Visual Medical Reports Using ML and Generative AI**  
Author: *Md. Fahad Nakib*

## Overview

DiagonoAid is a web-based diagnostic assistant that enables users to upload scanned blood reports and receive AI-powered health insights. The system combines machine learning and generative AI to extract medical metrics from images, predict disease probabilities, and generate personalized summaries and health tips.

## Features

- Upload visual medical reports (PDF or image)
- Extract key health metrics using multimodal generative AI
- Normalize values against clinical reference ranges
- Predict disease risk using a trained XGBoost classifier
- Generate human-readable summaries and health tips
- Interactive web interface for non-technical users

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript, Chart.js  
- **Backend**: Node.js, Express, Multer  
- **ML Pipeline**: Python, XGBoost, scikit-learn, pandas  
- **AI Integration**: Google Generative AI (Gemini)

## Project Structure

- `public/` – Frontend files  
- `uploads/` – Temporary storage for uploaded reports  
- `python/` – ML and AI processing scripts  
- `.pkl` files – Trained model and encoder  
- `server.js` – Node.js backend logic

## How to Run

1. Install Node.js dependencies:
   
   npm install

2. Install Python dependencies:

   pip install -r requirements.txt

   

