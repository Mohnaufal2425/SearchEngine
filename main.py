from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import numpy as np
import os
import open_clip
import faiss
import logging
from typing import List, Tuple, Optional
from torchvision import transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# File paths
text_data_path = 'models/styles.csv'
image_data_path = 'models/image.csv'
text_embeddings_path = 'models/text_embeddings.npy'
image_embeddings_path = 'models/image_embeddings.npy'
batch_size = 32

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Load Data ###
def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame to prevent further errors
    try:
        df = pd.read_csv(file_path, sep=';', quotechar='"', skipinitialspace=True)
        if 'Price' in df.columns:
            df['Price'] = pd.to_numeric(df['Price'].str.replace('$', ''), errors='coerce')
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        raise

### Preprocess Text and Images ###
def preprocess_text(text: str) -> str:
    return ' '.join(text.lower().strip().split())

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    return image_transforms(image)

def download_image(url: str, save_path: str) -> Optional[Image.Image]:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img.save(save_path)  # Save the image locally
            logging.info(f"Image downloaded and saved to {save_path}")
            return img
        else:
            logging.warning(f"Failed to fetch image at {url}. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading image from {url}: {str(e)}")
        return None

### Initialize DataFrames ###
text_df = load_data(text_data_path)
image_df = load_data(image_data_path)

if text_df.empty or image_df.empty:
    logging.error("DataFrames are empty. Exiting.")
    exit()

# Data Preprocessing
text_df['id'] = text_df['id'].astype(str)
image_df['filename'] = image_df['filename'].str.replace('.jpg', '')

merged_df = pd.merge(text_df, image_df, left_on='id', right_on='filename')
product_descriptions = [preprocess_text(desc) for desc in merged_df['productDisplayName'].tolist()]
image_urls = merged_df['link'].tolist()

logging.info(f"Loaded {len(product_descriptions)} product descriptions and {len(image_urls)} image URLs.")

# Download and Preprocess Images
image_cache = {}

def get_image(url: str) -> Optional[torch.Tensor]:
    img_dir = os.path.join('models', 'img')
    os.makedirs(img_dir, exist_ok=True)  # Ensure the directory exists

    img_filename = os.path.join(img_dir, os.path.basename(url))
    try:
        if os.path.exists(img_filename):  # Check if the image already exists locally
            logging.info(f"Image already exists locally: {img_filename}")
            img = Image.open(img_filename).convert('RGB')
        else:
            img = download_image(url, img_filename)

        if img is not None:
            preprocessed_img = preprocess_image(img)
            image_cache[url] = preprocessed_img
            return preprocessed_img
        else:
            logging.warning(f"Image at {url} could not be processed.")
            return None
    except Exception as e:
        logging.error(f"Error processing image {url}: {e}")
        return None

image_list = [get_image(url) for url in image_urls]
image_list = [img for img in image_list if img is not None]

# Initialize CLIP
try:
    model_name = "hf-hub:Marqo/marqo-fashionCLIP"
    cache_dir = "models/cache"
    os.makedirs(cache_dir, exist_ok=True)  # Ensure cache directory exists

    model, _, processor = open_clip.create_model_and_transforms(model_name, cache_dir=cache_dir)
    model = model.to(device)  # Move model to the correct device
except Exception as e:
    logging.error(f"Error initializing CLIP model {model_name}: {e}")
    raise

### Extract Features ###
def extract_text_features(texts: List[str], batch_size: int) -> np.ndarray:
    all_features = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = open_clip.tokenize(batch_texts).to(device)
        with torch.no_grad():
            outputs = model.encode_text(inputs)
        all_features.append(outputs.cpu().numpy())
    return np.vstack(all_features)

def extract_image_features(images: List[torch.Tensor], batch_size: int) -> np.ndarray:
    all_features = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        inputs = torch.stack(batch_images).to(device)
        with torch.no_grad():
            outputs = model.encode_image(inputs)
        all_features.append(outputs.cpu().numpy())
    return np.vstack(all_features)

# Save or Load Embeddings
if os.path.exists(text_embeddings_path) and os.path.exists(image_embeddings_path):
    text_features = np.load(text_embeddings_path)
    image_features = np.load(image_embeddings_path)
    logging.info("Loaded embeddings from file.")
else:
    text_features = extract_text_features(product_descriptions, batch_size)
    image_features = extract_image_features(image_list, batch_size)
    np.save(text_embeddings_path, text_features)
    np.save(image_embeddings_path, image_features)
    logging.info("Extracted and saved embeddings.")

### Initialize FAISS Index ###
def init_faiss_index(d: int) -> faiss.IndexFlatL2:
    return faiss.IndexFlatL2(d)

faiss_index = init_faiss_index(image_features.shape[1])
faiss_index.add(image_features)

### Define Search Index ###
def search_index(index: faiss.IndexFlatL2, query_features: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
    distances, indices = index.search(query_features, k)
    return list(zip(indices[0], distances[0]))

### Routes ###
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search/text")
async def text_search(query: str = Form(...)):
    try:
        query_embedding = extract_text_features([preprocess_text(query)], batch_size)
        results = search_index(faiss_index, query_embedding, k=5)

        output = []
        for idx, similarity in results:
            row = merged_df.iloc[idx]
            output.append({
                'name': row['productDisplayName'],
                'category': f"{row['masterCategory']} - {row['subCategory']}",
                'color': row['baseColour'],
                'gender': row['gender'],
                'price': float(row['Price']),
                'image_url': row['link'],
                'similarity': float(similarity)
            })

        return JSONResponse(content={"results": output})
    except Exception as e:
        logging.error(f"Error in text search: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/search/image")
async def image_search(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            return JSONResponse(content={"error": "Invalid file type. Only JPEG and PNG are supported."}, status_code=400)

        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        processed_image = preprocess_image(image)
        query_embedding = extract_image_features([processed_image], batch_size)
        results = search_index(faiss_index, query_embedding, k=5)

        output = []
        for idx, similarity in results:
            row = merged_df.iloc[idx]
            output.append({
                'name': row['productDisplayName'],
                'category': f"{row['masterCategory']} - {row['subCategory']}",
                'color': row['baseColour'],
                'gender': row['gender'],
                'price': float(row['Price']),
                'image_url': row['link'],
                'similarity': float(similarity)
            })

        return JSONResponse(content={"results": output})
    except Exception as e:
        logging.error(f"Error in image search: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
