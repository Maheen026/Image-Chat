import streamlit as st
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch

# Load the model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def chat_with_image(image_path, question):
    # Load image
    image = Image.open(image_path)
    
    # Prepare inputs
    inputs = processor(image, question, return_tensors="pt")
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted answer
    logits = outputs.logits
    predicted_answer = logits.argmax(-1).item()
    
    return processor.tokenizer.decode(predicted_answer)

import streamlit as st
from PIL import Image

st.title("Image Chat System")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # User input: Ask a question about the image
    question = st.text_input("Ask a question about the image:")
    
    if question:
        # Get the answer from the model
        answer = chat_with_image(uploaded_file, question)
        st.write(f"Answer: {answer}")
