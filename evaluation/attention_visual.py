import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import librosa 

def plot_flashy_abmil_spectrograms(abmil_results, X_bags, y_true, target_indices, class_names, 
                                   predict_abmil_fn, root_dir, data_input_csv):
    """
    Loads raw audio files, computes Mel Spectrograms, and dynamically illuminates
    the spectrogram frequencies based on the model's attention weights.
    """
    if len(target_indices) != 4:
        raise ValueError("Please provide exactly 4 indices for the target pure recordings.")

    # Load tracking metadata CSV
    df_meta = pd.read_csv(data_input_csv)

    # Extract model variables
    best_wrapper = abmil_results[0]['best_models'][0]
    pt_model = best_wrapper.model_
    scaler = best_wrapper.scaler_
    device = best_wrapper.device
    
    if hasattr(pt_model, 'eval'):
        pt_model.eval()

    # Set up the dark studio canvas
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=False)
    
    for i, (idx, ax) in enumerate(zip(target_indices, axes)):
        # --- Step 1: Resolve Paths & Load Audio Track ---
        rel_path = df_meta['relative_path'].values[idx]
        file_path = os.path.join(root_dir, rel_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not locate audio file at: {file_path}")
        y_audio, sr = librosa.load(file_path, sr=None)
        
        duration = len(y_audio) / sr

        # --- Step 2: Compute Mel Spectrogram ---
        S = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        n_mels, n_frames = S_db.shape
        
        # Normalize spectrogram matrix to [0, 1] for colormap assignment
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        
        # Map the normalized values into an RGBA image matrix
        spec_rgba = plt.cm.inferno(S_norm)

        # --- Step 3: Extract and Align Attention Profile ---
        bag_data = X_bags[idx]
        true_labels = y_true[idx]
        pure_class_idx = np.argmax(true_labels)
        target_class_name = class_names[pure_class_idx]
        
        _, attention_out = predict_abmil_fn(pt_model, [bag_data], scaler, device=device)
        
        # Safe dictionary/list unpacking from your custom ABMIL tracking system
        if isinstance(attention_out, dict) and 0 in attention_out:
            bag_attention_list = attention_out[0]
            if isinstance(bag_attention_list, (list, tuple)) and len(bag_attention_list) > pure_class_idx:
                attention_profile = bag_attention_list[pure_class_idx]
            else:
                attention_profile = bag_attention_list
        else:
            attention_profile = attention_out
            
        attention_profile = np.asarray(attention_profile, dtype=np.float64).flatten()

        # Interpolate low-res attention tokens to align perfectly with the spectrogram's time frames
        time_axis_spec = np.linspace(0, duration, n_frames)
        attention_upsampled = np.interp(
            time_axis_spec, 
            np.linspace(0, duration, len(attention_profile)), 
            attention_profile
        )
        
        # Normalize the attention array for intensity mapping
        att_min, att_max = attention_upsampled.min(), attention_upsampled.max()
        if (att_max - att_min) > 1e-8:
            att_norm = (attention_upsampled - att_min) / (att_max - att_min)
        else:
            att_norm = np.ones_like(attention_upsampled)

        # --- Step 4: Perform the Intensity Flash Processing ---
        # Baseline brightness is 15% (shadowy backdrop), scaling up to 100% burst illumination
        intensity_profile = 0.15 + 0.85 * att_norm
        
        # Multiply RGB values across all Mel rows by the time-varying intensity vector
        spec_rgba[:, :, :3] *= intensity_profile[np.newaxis, :, np.newaxis]

        # --- Step 5: Render Visual Elements ---
        ax.set_facecolor('#0d0e15')
        ax.grid(False)
        
        # Render the illuminated spectrogram
        ax.imshow(
            spec_rgba, 
            origin='lower', 
            aspect='auto', 
            extent=[0, duration, 0, n_mels],
            zorder=1
        )
        
        # Polish subplot elements
        ax.set_title(f"Track [{idx}] Spectrogram  |  Verified Target: {target_class_name.upper()}", 
                     color='#e0e0e6', fontsize=11, fontweight='bold', loc='left', pad=6)
        ax.set_ylabel("Mel Frequency Bands", color='#a0a0ab', fontsize=9)
        ax.tick_params(colors='#a0a0ab', labelsize=9)
        ax.set_xlim(0, duration)

        if i == 3:
            ax.set_xlabel("Time (Seconds)", color='#e0e0e6', fontsize=10, fontweight='bold')

    # Global canvas styling
    fig.patch.set_facecolor('#0d0e15')
    plt.suptitle("ABMIL SPECTROGRAM FLASH-ILLUMINATION (BURSTS HIGHLIGHT CLASSIFICATION CRITICAL BIOACOUSTIC TOKENS)", 
                 color='#ffffff', fontsize=12, fontweight='bold', y=0.98)
                 
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()