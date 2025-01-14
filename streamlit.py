import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
import os
import base64
import openai
import dotenv

dotenv.load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="AI Image Classifier", layout="centered")

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def process_image(image):
    try:
        # Create the array of the right shape
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize and crop
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        return data
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def encode_image(image):
    import io
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

# Streamlit interface
st.title("AI Image Classifier")
st.markdown("Upload an image, and the app will classify it as a shop/store or other.")

uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for file in uploaded_files:
        if file is not None and allowed_file(file.name):
            try:
                # Process the image
                image = Image.open(file)
                processed_data = process_image(image)
                encoded_image = encode_image(image)

                image_prompts = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]

                prompt_content = {
                    "Given is an image. Check whether it resembles an image of any type of shop or store. If it resembles a shop or a store, return 1; otherwise, for any other image, return 0. Do not return any other text."
                }

                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": prompt_content},
                        {"role": "user", "content": str(image_prompts)},
                    ],
                    max_tokens=300,
                )
                response_text = response.choices[0].message.content

                if "0" in response_text:
                    st.warning(f"Image '{file.name}' classified as: Not a shop/store")
                    continue

                # Get prediction from the model
                prediction = model.predict(processed_data)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = float(prediction[0][index])

                result = f"Class: {class_name[2:].strip()} (Confidence: {confidence_score:.2f})"
                results.append(result)

                st.success(f"Image '{file.name}' classified as: {result}")
            except Exception as e:
                st.error(f"Error processing '{file.name}': {str(e)}")
        else:
            st.error(f"Invalid file type: {file.name}")

    if results:
        st.subheader("Classification Results")
        for res in results:
            st.write(res)
