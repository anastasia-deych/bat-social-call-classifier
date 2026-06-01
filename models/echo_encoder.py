import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ============================================================================
# 1. THE NORMAL-SPACE AUTOENCODER
# ============================================================================
class EchoAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32):
        super().__init__()
        # Compress the acoustic features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim * 2),
            nn.ReLU(),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.ReLU()
        )
        # Reconstruct the acoustic features
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.ReLU(),
            nn.Linear(bottleneck_dim * 2, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# ============================================================================
# 2. TRAINING FUNCTION (STRICTLY ON ECHO-ONLY RECORDINGS)
# ============================================================================
def train_echo_anomaly_detector(X_bags, y_labels, epochs=30, batch_size=256, lr=1e-3):
    """
    Filters out any recording containing Social Calls. Trains ONLY on pure Echos.
    """
    # Identify bags that are PURE echo (assuming Echo is index 4 based on your order)
    # Target structure: [Type A, Type B, Type C, Type D, Echo]
    # Pure echo means columns 0, 1, 2, 3 are all 0, and column 4 is 1.
    pure_echo_indices = np.where((y_labels[:, :4].sum(axis=1) == 0) & (y_labels[:, 4] == 1))[0]
    
    # Flatten all windows from pure echo bags into a single training matrix
    echo_windows = np.vstack([X_bags[i] for i in pure_echo_indices])
    
    input_dim = echo_windows.shape[1]
    ae = EchoAutoencoder(input_dim=input_dim).cuda() if torch.cuda.is_available() else EchoAutoencoder(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    
    # Convert to PyTorch Tensor Loader
    tensor_x = torch.FloatTensor(echo_windows)
    dataset = torch.utils.data.TensorDataset(tensor_x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    ae.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch_x,) in loader:
            if torch.cuda.is_available(): batch_x = batch_x.cuda()
            
            optimizer.zero_grad()
            outputs = ae(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    return ae

# ============================================================================
# 3. FEATURE ENHANCEMENT PIPELINE
# ============================================================================
def transform_bags_with_anomaly_scores(X_bags, trained_ae):
    """
    Transforms your raw window features by appending the reconstruction error.
    New shape per window: (n_features + 1)
    """
    trained_ae.eval()
    enhanced_bags = []
    
    with torch.no_grad():
        for bag in X_bags:
            # Convert single bag to tensor
            bag_tensor = torch.FloatTensor(bag)
            if torch.cuda.is_available(): bag_tensor = bag_tensor.cuda()
            
            # Predict reconstructions
            reconstructed = trained_ae(bag_tensor)
            
            # Calculate MSE per window: Mean over the feature dimension (dim=1)
            # Shape: (n_windows, 1)
            error_per_window = torch.mean((bag_tensor - reconstructed) ** 2, dim=1, keepdim=True)
            
            # Move back to numpy
            error_np = error_per_window.cpu().numpy()
            
            # Combine raw acoustic features with the new anomaly score feature
            enhanced_bag = np.hstack([bag, error_np])
            enhanced_bags.append(enhanced_bag)
            
    return enhanced_bags