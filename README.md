#  Credit Eligibility Prediction App

This project is a web-based application that predicts whether a person is eligible for a bank loan based on basic information like credit history, marital status, and co-applicant income.

##  Technologies Used

- Python 🐍
- Flask 
- Scikit-learn 
- HTML/CSS for the frontend
- Jupyter Notebook 

## Project Objectives

- Develop a machine learning model to classify loan eligibility.
- Serve the model using a Flask web server.
- Design a clean and simple user interface.
- Enable real-time predictions using form input.

## 🧠 What You’ll Learn

- How to prepare data for training.
- How to train a classification model.
- How to deploy the model with Flask.
- How to create a simple interface connected to a Python backend.

## 🛠 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/RaneemBenMrad/credit-eligibility-app.git
cd credit-eligibility-app

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
