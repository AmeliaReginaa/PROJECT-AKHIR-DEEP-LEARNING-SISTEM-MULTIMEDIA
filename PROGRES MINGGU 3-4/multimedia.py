import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

# ==================================
# CUSTOM CSS
# ==================================
st.markdown("""
<style>

.main{
    background:#0f172a;
}

.title{
    font-size:40px;
    font-weight:700;
    color:white;
}

.subtitle{
    color:#94a3b8;
    font-size:18px;
}

.pred-box{
    background:#1e293b;
    padding:20px;
    border-radius:15px;
    border:1px solid #334155;
}

.stButton>button{
    width:100%;
    background:#2563eb;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
    border:none;
}

.stButton>button:hover{
    background:#1d4ed8;
}

</style>
""", unsafe_allow_html=True)

# ==================================
# MODEL
# ==================================
class EmotionCNNDeepFC(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )

    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x


# ==================================
# LOAD MODEL
# ==================================
@st.cache_resource
def load_model(path):

    model = EmotionCNNDeepFC()

    checkpoint = torch.load(
        path,
        map_location='cpu'
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


# ==================================
# TRANSFORM
# ==================================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

classes = [
    "Negative Emotion",
    "Positive Emotion"
]

# ==================================
# HEADER
# ==================================
st.markdown(
    '<p class="title">Emotion Recognition System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">CNN-based Facial Emotion Classification</p>',
    unsafe_allow_html=True
)

st.divider()

# ==================================
# SIDEBAR
# ==================================
st.sidebar.title("Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "FP32 Model",
        "FP16 Model"
    ]
)

if model_choice=="FP32 Model":
    model_path="best_emotion_model(2).pth"
else:
    model_path="emotion_cnn_final_fp16.pth"

model = load_model(model_path)

# ==================================
# MAIN
# ==================================
col1,col2 = st.columns([1,1])

with col1:

    uploaded = st.file_uploader(
        "Upload Facial Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

with col2:

    st.markdown(
        '<div class="pred-box">',
        unsafe_allow_html=True
    )

    st.subheader("Prediction Result")

    if uploaded:

        img = transform(image)
        img = img.unsqueeze(0)

        with torch.no_grad():

            output = model(img)

            prob = torch.softmax(
                output,
                dim=1
            )

            confidence, pred = torch.max(
                prob,
                1
            )

        st.metric(
            "Prediction",
            classes[pred.item()]
        )

        st.progress(
            float(confidence)
        )

        st.write(
            f"Confidence : "
            f"{confidence.item()*100:.2f}%"
        )

    else:
        st.info(
            "Upload image first"
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )