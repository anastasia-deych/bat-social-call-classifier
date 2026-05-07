import numpy as np
import torch
from tqdm import tqdm

#effnet and beats import
from avex import load_model
#perch 2.0 import
import tensorflow as tf
import tensorflow_hub as hub


def extract_feature(window, encoder, model_name, device='cpu'):
    # Convert Torch tensor to NumPy for cross-framework compatibility
    if isinstance(window, torch.Tensor):
        window_np = window.cpu().numpy()
    else:
        window_np = window

    if model_name == 'perch2':
        # Perch expects a [Batch, Samples] float32 TF tensor
        # We must use tf.constant and ensure it's on the right device
        with tf.device('/GPU:0' if 'cuda' in str(device) else '/CPU:0'):
            # window_np shape is likely [Num_Windows, Samples]
            feats_tf = encoder(tf.constant(window_np, dtype=tf.float32))
            return torch.from_numpy(feats_tf['spatial_embedding'].numpy())
            
    else:
        # PyTorch models (EffNet/BEATs)
        encoder.eval()
        with torch.no_grad():
            # Ensure input is [Batch, Samples] or [Batch, 1, Samples]
            # avex usually wants [Batch, Samples]
            t_window = torch.from_numpy(window_np).to(device)
            feats = encoder(t_window)
            if isinstance(feats, dict): 
                feats = feats['x']
            return feats.cpu() # Return to CPU to avoid filling GPU RAM

def build_feature_bank(batdata, encoder, model_name, device='cpu'):
    feature_list = []
    label_list = []
    print(f"Dataset type: {type(batdata)}")
    
    for i in tqdm(range(len(batdata)), desc=f"Extracting {model_name}"):
        windows, labels = batdata[i]
        
        # Extract features for this specific file
        feats = extract_feature(windows, encoder, model_name, device)
        
        # feats is [Num_Windows, Embedding_Dim]
        feature_list.append(feats.numpy())
        label_list.append(labels.numpy())
            
    # Concatenate everything into two giant matrices
    return np.concatenate(feature_list, axis=0), np.concatenate(label_list, axis=0)


def extract_encoder(model_name, device='cpu'):
    if model_name in ['effnetb0', 'NLM_BEATs']:
        # PyTorch logic
        model_key = "esp_aves2_effnetb0_all" if model_name == 'effnetb0' else "esp_aves2_naturelm_audio_v1_beats"
        encoder = load_model(model_key, device=device, return_features_only=True)
        encoder.to(device)
        return encoder
        
    elif model_name == 'perch2':
        # TensorFlow logic
        suffix = "perch_v2_cpu" if device == 'cpu' else "perch_v2"
        perch_url = f"https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/{suffix}/versions/1"
        perch_model = hub.load(perch_url)
        return perch_model.signatures['serving_default'] 
    

def pool_features(features, windows : bool = False, method : str ='mean',encoder : str = 'perch2'):
    features = np.array(features) # Ensure it's a NumPy array for pooling operations

    if windows :
        if method == 'mean':
            return features.mean(axis=1)
        elif method == 'max':
            return features.max(axis=1)
        else:
            raise ValueError(f"Unsupported pooling method: {method}")
    else :
        if encoder == 'effnetb0' :
            ax = (2,3)
            if features.ndim > 4 : ax = (3,4)
        elif encoder == 'NLM_BEATs' :
            ax = 1
            if features.ndim > 3 : ax = 2
        elif encoder == 'perch2' :
            ax = (1,2)
            if features.ndim > 4 : ax = (2,3)

        
        if method == 'mean':
            return features.mean(axis=ax)
        elif method == 'max':
            return features.max(axis=ax)
        else:
            raise ValueError(f"Unsupported pooling method: {method}")