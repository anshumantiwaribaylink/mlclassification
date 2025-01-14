from flask import Flask, render_template, request , jsonify
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
import os
import base64
import openai
import dotenv

dotenv.load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24) # Required for flashing messages

openai.api_key = os.getenv("OPENAI_API_KEY")
# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
with open("labels.txt", "r") as f:
    class_names = f.readlines()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    # Create a bytes buffer
    import io
    buffer = io.BytesIO()
    # Save the image to the buffer in JPEG format
    image.save(buffer, format='JPEG')
    # Get the bytes from the buffer
    img_bytes = buffer.getvalue()
    # Encode to base64
    return base64.b64encode(img_bytes).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/classify-store-new', methods=['POST'])
def api_classify_store_new():
    if 'images' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('images')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400

    results = []
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Read and process the image
                image = Image.open(file)
                processed_data = process_image(image)
                encoded_image = encode_image(image)
                image_prompts = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]

                prompt_content = {
                    "Given is an image. Chek whether it resembles a image of any type of shop or store.If it resembles like a shop or s store return 1 otherwise for any other image return 0. Dont return any other text."
                }

                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": prompt_content},
                        {"role": "user", "content": image_prompts},
                    ],
                    max_tokens=300,
                )
                print("Chat GPT response:")
                print(response.choices[0].message.content)
                response_text = response.choices[0].message.content
                # Check if the response text contains "1"
                if "0" in response_text:
                    result = f"Class: NA"
                    return jsonify({'results': [result]})
                # Get prediction
                prediction = model.predict(processed_data)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = float(prediction[0][index])
                 
                result = f"Class: {class_name[2:].strip()}"
                results.append(result)
            except Exception as e:
                results.append(f"Error: {str(e)}")
        else:
            results.append(f"Invalid file type")
    
    return jsonify({'results': results})

if __name__ == '__main__':
    PORT = int(os.getenv('PORT', 5000))
    app.run(debug=True,port=PORT)
