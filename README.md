# Baylink ML Store Classification

A Flask-based API service that uses machine learning to classify whether an image contains a store/shop or not.

## Overview

This service provides an API endpoint that accepts image uploads and uses a combination of machine learning models and OpenAI's vision capabilities to determine if the uploaded image represents a store or shop front. The system returns a binary classification (1 for store, 0 for non-store). A non-store response from OpenAI automatically indicates that the image does not contain a store or shop front, hence the class NA is returned. Beyond this, where a store is detected, the model classifies the store into one of the 3 categories: A, B, and C.

Rationale: We'd rather have false negatives than false positives, since a false negative can be manually fixed, and will tend to be less in number, i.e a store-front of a shop, classified a not-a-store, can be changed, but a lot of non-stores going into the store images section, with classified classes will be a bigger challenge to fix. 

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`:
  - TensorFlow 2.15.0
  - Flask 3.0.0
  - Pillow 10.1.0
  - NumPy 1.24.3
  - OpenAI
  - python-dotenv
  - Additional dependencies

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## API Endpoints

### POST /api/classify-store-new

Classifies whether uploaded images contain stores/shops.

#### Request
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body parameter: `images` (accepts multiple image files)
- Supported formats: PNG, JPG, JPEG

#### Response
- Success Response (200 OK):
  ```json
  {
    "results": [
      {
        "classification": 1,  // 1 for store, 0 for non-store
        "filename": "example.jpg"
      }
    ]
  }
  ```
- Error Response (400 Bad Request):
  ```json
  {
    "error": "Error message"
  }
  ```

## Key Functions

- `process_image(image)`: Preprocesses uploaded images for classification
- `encode_image(image)`: Encodes images to base64 for API processing
- `allowed_file(filename)`: Validates file extensions for uploaded images
- `api_classify_store_new()`: Main API endpoint handler for store classification

## Security

- Implements file extension validation
- Uses secure filename handling
- Environment variables for sensitive data
- Upload directory protection

## Local Development

The application runs in debug mode on the local development server:
```bash
python app.py
```
Access the application at `http://localhost:5000`
