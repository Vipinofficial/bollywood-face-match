# Bollywood Celebrity Face Recognition 🎬

A **Streamlit web app** that recognizes Bollywood celebrities from uploaded face images.  
Uses **PyTorch deep learning models** for:

- Celebrity classification
- Embedding-based similarity matching
- Facial attribute analysis (skin tone, complexion, texture)

---

## Features
- Upload an image and get:
  - **Top celebrity predictions** with confidence scores
  - **Most similar celebrities** using embedding similarity
  - **Facial attributes** (skin tone, brightness, complexion, texture)
- Interactive **Plotly visualizations**
- Clean, responsive UI with LinkedIn and GitHub links

---

## Folder Structure
```bash Bollywood-face-train/
│── best_bollywood_model.pth
│── bollywood_attribute_model.pth
│── celebrity_embeddings.pkl
│── model_metadata.json
│── label_encoder.pkl

```


---

## Installation

1. Clone the repo:
```bash
git clone https://github.com/your-username/bollywood-face-recognition.git
cd bollywood-face-recognition
Create a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Place your trained models in the Bollywood-face-train/ folder.

Run the App
bash
Copy
Edit
streamlit run app.py
The app will launch in your browser at http://localhost:8501

Notes
Requires PyTorch ≥ 2.0

Uses torch.serialization.add_safe_globals([LabelEncoder]) to load sklearn objects safely.

Tested on CPU and GPU.
