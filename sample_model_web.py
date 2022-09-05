from json import load
import streamlit as st
from skimage import io
from skimage.transform import resize
import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@st.cache(allow_output_mutation=True)
def load_model():
    return torch.load('./MNIST-Model.pt').to(device)

model = load_model()
uploaded_file = st.file_uploader('Upload Image (.png)', type='png')
if uploaded_file is not None:
    image1 = io.imread(uploaded_file, as_gray=True)
    image_resized = resize(image1, (28, 28), anti_aliasing=True)
    X1 = image_resized.reshape(1, 28, 28)
    X1 = torch.FloatTensor(1-X1).to(device)
    predictions = torch.softmax(model(X1), dim=1)
    st.write(f'### Prediction Result:{np.argmax(predictions.detach().cpu().numpy())}')
    st.image(image1)