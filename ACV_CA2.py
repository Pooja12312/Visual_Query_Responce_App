pip install transformers torch streamlit pillow

!streamlit run app.py & npx localtunnel --port 8501

pip install --upgrade streamlit


%%writefile app.py
import streamlit as st
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stImage {
        text-align: center;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
    }
    .stButton > button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 10px 20px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        transition: 0.3s;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    .stTextInput {
        text-align: center;
    }
    .stWrite {
        text-align: center;
        font-size: 18px;
        font-weight: bold;
    }
    h1 {
        text-align: center;
        color: #388E3C;  /* Darker Green */
    }
    .answer-box {
        width: 100%;
        max-width: 600px;
        margin: 20px auto;
        padding: 15px;
        border: 2px solid #388E3C;
        border-radius: 10px;
        text-align: center;
        background-color: #f4f4f4;
    }
    .answer {
        font-weight: bold;
        color: black;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

st.markdown("<h1>Visual Query Response APP</h1>", unsafe_allow_html=True)

# File uploader for the image
uploaded_file = st.file_uploader("Upload Image file:", type=["jpg", "jpeg", "png"])

# Text input for the question
question = st.text_input("Enter your question:")

if st.button("Submit"):
    if uploaded_file is not None and question:
        # Open and convert the image to RGB format
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Display the question
        st.write("**Question:**", question)

        # Process the input
        inputs = processor(image, question, return_tensors="pt")

        # Use generate() for inference
        with torch.no_grad():
            output_ids = model.generate(**inputs)
            answer = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Display the answer inside a styled box
        st.markdown(f"""
        <div class="answer-box">
            <p class="answer">Answer: {answer}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.write("Please upload an image and enter a question.")

