import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import pickle, json
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import torch.serialization
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download

# Allow sklearn LabelEncoder for torch.load
torch.serialization.add_safe_globals([LabelEncoder])

# -------------------------
# Model Definitions
# -------------------------
class BollywoodFaceNet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(2048, 1024), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(1024, 512), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(512, num_classes)
        )
        self.feature_extractor = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(2048, 512), nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x, return_features=False):
        features = self.features(x)
        features = torch.flatten(features, 1)
        if return_features:
            emb = self.feature_extractor(features)
            logits = self.classifier(features)
            return logits, emb
        return self.classifier(features)

class AttributeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=False)
        self.backbone.classifier = nn.Identity()
        self.skin_tone_head = nn.Sequential(
            nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 3)
        )
        self.complexion_head = nn.Sequential(
            nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 3)
        )
        self.texture_head = nn.Sequential(
            nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2)
        )

    def forward(self, x):
        f = self.backbone(x)
        return {
            'skin_tone': self.skin_tone_head(f),
            'complexion': self.complexion_head(f),
            'texture': self.texture_head(f)
        }

# -------------------------
# Load Models & Data
# -------------------------
@st.cache_resource
def load_predictor():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    repo_id = "vipinkumarsoni/Bollywood-face-trained-ResNet50"

    model_path = hf_hub_download(repo_id=repo_id, filename="best_bollywood_model.pth")
    attr_model_path = hf_hub_download(repo_id=repo_id, filename="bollywood_attribute_model.pth")
    embeddings_path = hf_hub_download(repo_id=repo_id, filename="celebrity_embeddings.pkl")
    metadata_path = hf_hub_download(repo_id=repo_id, filename="model_metadata.json")
    label_encoder_path = hf_hub_download(repo_id=repo_id, filename="label_encoder.pkl")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    num_classes = metadata['num_classes']
    img_size = metadata['img_size']

    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    with open(embeddings_path, 'rb') as f:
        celeb_embeddings = pickle.load(f)

    model = BollywoodFaceNet(num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    attr_model = AttributeNet().to(device)
    attr_ckpt = torch.load(attr_model_path, map_location=device, weights_only=False)
    attr_model.load_state_dict(attr_ckpt['model_state_dict'])
    attr_model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return model, attr_model, label_encoder, celeb_embeddings, transform, device, metadata

model, attr_model, label_encoder, celeb_embeddings, transform, device, metadata = load_predictor()

# -------------------------
# Helpers
# -------------------------
def preprocess_image(image):
    return transform(image.convert('RGB')).unsqueeze(0).to(device)

def predict_celebrity(image_tensor, top_k=5):
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        top_idx = np.argsort(probs)[-top_k:][::-1]
        return [(label_encoder.inverse_transform([i])[0], probs[i]) for i in top_idx]

def find_similar(image_tensor, top_k=5):
    with torch.no_grad():
        _, emb = model(image_tensor, return_features=True)
        emb = emb.cpu().numpy()
    sims = {c: cosine_similarity(emb.reshape(1, -1), e.reshape(1, -1))[0][0] 
            for c, e in celeb_embeddings.items()}
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k]

def extract_attributes(image_tensor):
    with torch.no_grad():
        out = attr_model(image_tensor)
        skin_tone_raw = out['skin_tone'].cpu().numpy()[0]
        category = "Warm" if skin_tone_raw[0] > skin_tone_raw[1] and skin_tone_raw[0] > skin_tone_raw[2] \
            else "Neutral" if skin_tone_raw[1] > skin_tone_raw[0] else "Cool"
        return {
            "skin_tone": category,
            "brightness": float(np.mean(skin_tone_raw)),
            "complexion": np.argmax(out['complexion'].cpu().numpy()[0]),
            "texture": np.argmax(out['texture'].cpu().numpy()[0])
        }

# -------------------------
# Streamlit Glam UI
# -------------------------
st.set_page_config(page_title="Bollywood Celebrity Face Recognition", layout="wide")

st.markdown(
    """
    <style>
    .main {background-color: #000;}
    .stApp {color: #FFD700;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: #FFD700;'>✨ Bollywood Celebrity Face Recognition ✨</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'>Upload an image to see your celebrity matches!</h3>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("About")
st.sidebar.info("Recognize Bollywood celebrities with AI.\n\n**Made by Vipin Kumar Soni**")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/your-link) | [GitHub](https://github.com/your-username)")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    tensor = preprocess_image(image)
    attrs = extract_attributes(tensor)
    top_preds = predict_celebrity(tensor)
    similars = find_similar(tensor)

    st.markdown("### 🎭 Top Celebrity Predictions")
    fig = go.Figure(go.Bar(
        x=[p[1]*100 for p in top_preds],
        y=[p[0] for p in top_preds],
        orientation='h',
        marker=dict(color='gold')
    ))
    fig.update_layout(xaxis_title="Confidence (%)", yaxis_title="Celebrity")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🌟 Most Similar Celebrities")
    fig2 = go.Figure(go.Bar(
        x=[s[1]*100 for s in similars],
        y=[s[0] for s in similars],
        orientation='h',
        marker=dict(color='orange')
    ))
    fig2.update_layout(xaxis_title="Similarity (%)", yaxis_title="Celebrity")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 💎 Facial Attributes")
    st.write(f"**Skin Tone:** {attrs['skin_tone']} (Brightness: {attrs['brightness']:.2f})")
    st.write(f"**Complexion:** {['Fair', 'Medium', 'Dark'][attrs['complexion']]}")
    st.write(f"**Texture:** {['Smooth', 'Rough'][attrs['texture']]}")
