import pandas as pd
import numpy as np
import re
import os
from PIL import Image
from tqdm import tqdm

# Deep Learning Libraries
from transformers import DistilBertTokenizer, DistilBertModel
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- CONFIGURATION (STABILITY FIX) ---
DEVICE = torch.device("cpu") 

# --- STEP 1: INITIALIZATION FUNCTION (RUN ONCE) ---
@torch.no_grad()
def initialize_models():
    """Initializes and returns pre-trained text and image models."""
    
    print(f"Initializing deep learning models on device: {DEVICE}")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    text_model.to(DEVICE).eval()
    
    weights = ResNet50_Weights.DEFAULT
    image_transform = weights.transforms()
    image_model = resnet50(weights=weights)
    image_model.fc = nn.Identity() 
    image_model.to(DEVICE).eval()
    
    return tokenizer, text_model, image_transform, image_model

# NOTE: The models are NO LONGER loaded globally here. They will be loaded in main.py.


# --- FEATURE ENGINEERING FUNCTIONS (NOW ACCEPT MODELS AS ARGS) ---

def extract_structured_features(df, target_col='price', le_brand=None, le_ipq=None):
    """
    Extracts Brand, IPQ, handles target price normalization.
    (This function remains the same logically, just ensures internal stability.)
    """
    df_copy = df.copy().reset_index(drop=True) 
    ipq_pattern = re.compile(r'(\d+)\s*[- ]*(pack|count|oz|ounces|liters|g|grams|ml|milliliters|lb|pounds)', re.IGNORECASE)
    
    def parse_ipq(content):
        match = ipq_pattern.search(content)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            return value, unit
        return 1.0, 'other' 
    
    ipq_data = df_copy['catalog_content'].apply(parse_ipq).apply(pd.Series)
    ipq_data.columns = ['IPQ_Value', 'IPQ_Unit']
    df_copy = pd.concat([df_copy, ipq_data], axis=1)
    df_copy.reset_index(drop=True, inplace=True) 

    df_copy['Brand'] = df_copy['catalog_content'].apply(lambda x: x.split('\n')[0].split(': ')[-1].split(' - ')[0].strip())
    
    # Encoding logic (fixed for stability)
    if le_brand is None:
        le_brand = LabelEncoder()
        df_copy['Brand_Encoded'] = le_brand.fit_transform([str(x) for x in df_copy['Brand'].values]) 
    else:
        classes_list = list(le_brand.classes_)
        unknown_label = len(classes_list)
        df_copy['Brand_Encoded'] = [le_brand.transform([str(x)])[0] if str(x) in classes_list else unknown_label for x in df_copy['Brand'].values]
        
    if le_ipq is None:
        le_ipq = LabelEncoder()
        df_copy['IPQ_Unit_Encoded'] = le_ipq.fit_transform([str(x) for x in df_copy['IPQ_Unit'].values]) 
    else:
        classes_list = list(le_ipq.classes_)
        unknown_label = len(classes_list)
        df_copy['IPQ_Unit_Encoded'] = [le_ipq.transform([str(x)])[0] if str(x) in classes_list else unknown_label for x in df_copy['IPQ_Unit'].values]

    y_train = None
    if target_col in df_copy.columns:
        price_values = df_copy[target_col].to_numpy()
        ipq_values = df_copy['IPQ_Value'].to_numpy()
        
        if ipq_values.ndim > 1 and ipq_values.shape[1] == 1:
            ipq_values = ipq_values.flatten()
        elif ipq_values.ndim > 1 and ipq_values.shape[1] == 2:
            ipq_values = ipq_values[:, 0] 

        ipq_values = np.clip(ipq_values, a_min=1e-6, a_max=None)
        
        df_copy['Price_Per_Unit'] = price_values / ipq_values
        y_train = np.log1p(df_copy['Price_Per_Unit'])

    return df_copy, y_train, le_brand, le_ipq

@torch.no_grad()
def generate_text_embeddings(texts, tokenizer, model):
    """Generates DistilBERT embeddings using CPU."""
    embeddings = []
    for i in tqdm(range(0, len(texts), 32), desc="Text Embedding"):
        batch_texts = texts[i:i+32]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, 
                            max_length=512, return_tensors='pt').to(DEVICE)
        
        output = model(**encoded)
        batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
        
    return np.concatenate(embeddings, axis=0)

@torch.no_grad()
def generate_image_embeddings(df, image_folder, model, transform):
    """Generates ResNet-50 embeddings using CPU."""
    embeddings = []
    ZERO_VECTOR = np.zeros(2048) 
    
    for row in tqdm(df.itertuples(), total=len(df), desc="Image Embedding"):
        image_link = row.image_link
        filename = os.path.basename(image_link)
        image_path = os.path.join(image_folder, filename)
        
        try:
            img = Image.open(image_path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(DEVICE)
            embedding = model(img_t).squeeze(0).cpu().numpy()
            embeddings.append(embedding)
            
        except Exception:
            embeddings.append(ZERO_VECTOR)
            
    return np.array(embeddings)


# --- STEP 2: FEATURE MATRIX CREATOR (ACCEPTS MODELS) ---
def create_feature_matrix(df, image_folder, mode='train', metadata=None, models=None):
    """Master function to orchestrate feature generation and return the final matrix."""
    
    tokenizer, text_model, image_transform, image_model = models
    metadata = metadata if metadata is not None else {}
    
    le_brand = metadata.get('le_brand')
    le_ipq = metadata.get('le_ipq')
    
    df_processed, y, le_brand, le_ipq = extract_structured_features(
        df, le_brand=le_brand, le_ipq=le_ipq
    )
    
    text_embeddings = generate_text_embeddings(
        df_processed['catalog_content'].tolist(), tokenizer, text_model
    )
    image_embeddings = generate_image_embeddings(
        df_processed, image_folder, image_model, image_transform
    )
    
    structured_cols = ['IPQ_Value', 'Brand_Encoded', 'IPQ_Unit_Encoded']
    X_struct = df_processed[structured_cols].values
    
    scaler = metadata.get('scaler')
    if scaler is None:
        scaler = StandardScaler()
        X_struct_scaled = scaler.fit_transform(X_struct)
    else:
        X_struct_scaled = scaler.transform(X_struct)
        
    X = np.concatenate([text_embeddings, image_embeddings, X_struct_scaled], axis=1)

    if mode == 'train':
        metadata = {'le_brand': le_brand, 'le_ipq': le_ipq, 'scaler': scaler}
        return pd.DataFrame(X), y, metadata 
    else:
        return pd.DataFrame(X), df_processed['IPQ_Value']