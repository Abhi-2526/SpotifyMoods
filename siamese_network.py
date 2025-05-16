import os
import math
import json
from datetime import datetime

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
import secrets

# For reproducibility
def set_seed(seed=42):
    secrets.SystemRandom().seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ---------------------------
# Dataset Preparation
# ---------------------------
class SongDataset(Dataset):
    def __init__(self, df, scaler=None, train=True):
        """
        Initialize dataset with proper handling of features and mood hierarchies.

        Args:
            df (DataFrame): DataFrame containing song data
            scaler (StandardScaler, optional): Scaler for feature normalization
            train (bool): Whether this is training data (for fitting scaler)
        """
        self.df = df.reset_index(drop=True)

        # Process features
        features_list = []
        for _, row in df.iterrows():
            # Normalize continuous features
            tempo_normalized = row['tempo'] / 250  # Assuming max tempo is 250 BPM
            loudness_normalized = (row['loudness'] + 60) / 60  # Normalize from [-60,0] to [0,1]

            # Process key using circular encoding
            key_angle = 2 * math.pi * row['key'] / 12  # Convert key to angle
            key_sin = math.sin(key_angle)
            key_cos = math.cos(key_angle)

            # Create feature vector in consistent order
            feature_vector = [
                row['danceability'],          # Already 0-1
                row['energy'],                # Already 0-1
                row['valence'],               # Already 0-1
                tempo_normalized,             # Normalized tempo
                row['speechiness'],           # Already 0-1
                row['acousticness'],          # Already 0-1
                row['instrumentalness'],      # Already 0-1
                row['liveness'],              # Already 0-1
                loudness_normalized,          # Normalized loudness
                row['mode'],                  # Binary (0 or 1)
                key_sin,                      # Circular encoding for key
                key_cos                       # Circular encoding for key
            ]
            features_list.append(feature_vector)

        self.features = np.array(features_list, dtype=np.float32)

        # Scale features if needed
        if scaler is None and train:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        elif scaler is not None:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)

        # Handle mood information
        # Primary moods with confidence
        self.primary_moods = df['primary_mood'].fillna('Unknown').values
        self.primary_confidences = df['primary_mood_confidence'].fillna(0.0).values

        # Secondary moods with confidence
        self.secondary_moods = df['secondary_mood'].fillna('None').values
        self.secondary_confidences = df['secondary_mood_confidence'].fillna(0.0).values

        # Tertiary moods with confidence
        self.tertiary_moods = df['third_mood'].fillna('None').values
        self.tertiary_confidences = df['third_mood_confidence'].fillna(0.0).values

        # Store mood hierarchies for each song as a list
        self.mood_hierarchies = []
        for i, row in df.iterrows():
            moods = []
            # Add primary mood if valid
            if pd.notna(row['primary_mood']) and row['primary_mood'] != 'Unknown':
                moods.append({
                    'mood': row['primary_mood'],
                    'confidence': float(row['primary_mood_confidence']) if pd.notna(
                        row['primary_mood_confidence']) else 0.0,
                    'level': 0  # Primary level
                })

            # Add secondary mood if valid
            if pd.notna(row['secondary_mood']) and row['secondary_mood'] != 'None':
                moods.append({
                    'mood': row['secondary_mood'],
                    'confidence': float(row['secondary_mood_confidence']) if pd.notna(
                        row['secondary_mood_confidence']) else 0.0,
                    'level': 1  # Secondary level
                })

            # Add tertiary mood if valid
            if pd.notna(row['third_mood']) and row['third_mood'] != 'None':
                moods.append({
                    'mood': row['third_mood'],
                    'confidence': float(row['third_mood_confidence']) if pd.notna(
                        row['third_mood_confidence']) else 0.0,
                    'level': 2  # Tertiary level
                })

            self.mood_hierarchies.append(moods)

        # Create mood mapping excluding invalid moods
        valid_moods = set()
        for moods in self.mood_hierarchies:
            for mood_info in moods:
                valid_moods.add(mood_info['mood'])

        self.unique_moods = sorted(valid_moods - {'Unknown', 'None'})
        self.mood_to_idx = {mood: idx for idx, mood in enumerate(self.unique_moods)}

    def get_all_moods(self, idx):
        return self.mood_hierarchies[idx]

    def get_mood_vector(self, idx):
        mood_vector = torch.zeros(len(self.unique_moods))
        for mood_info in self.mood_hierarchies[idx]:
            if mood_info['mood'] in self.mood_to_idx:
                mood_idx = self.mood_to_idx[mood_info['mood']]
                # Weight confidence by hierarchy level
                hierarchy_weight = 1.0 if mood_info['level'] == 0 else 0.7 if mood_info['level'] == 1 else 0.4
                mood_vector[mood_idx] = mood_info['confidence'] * hierarchy_weight
        return mood_vector

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get item from dataset.

        Args:
            idx (int): Index of the item

        Returns:
            dict: Dictionary containing:
                - features: Tensor of audio features
                - moods: List of mood information
                - mood_vector: Weighted mood vector
        """
        return {
            'features': torch.FloatTensor(self.features[idx]),
            'moods': self.get_all_moods(idx),
            'mood_vector': self.get_mood_vector(idx)
        }

    def get_feature_names(self):
        """Return list of feature names in order."""
        return [
            'danceability',
            'energy',
            'valence',
            'tempo_normalized',
            'speechiness',
            'acousticness',
            'instrumentalness',
            'liveness',
            'loudness_normalized',
            'mode',
            'key_sin',
            'key_cos'
        ]

    def get_mood_names(self):
        """Return list of all unique valid moods."""
        return self.unique_moods

    def get_mood_statistics(self):
        """Return statistics about mood distribution."""
        stats = {
            'total_songs': len(self.df),
            'unique_moods': len(self.unique_moods),
            'mood_counts': {},
            'hierarchy_distribution': {
                'primary_only': 0,
                'primary_secondary': 0,
                'all_three': 0
            }
        }

        # Count mood occurrences
        for moods in self.mood_hierarchies:
            mood_levels = [m['level'] for m in moods]

            # Update hierarchy distribution
            if len(mood_levels) == 1:
                stats['hierarchy_distribution']['primary_only'] += 1
            elif len(mood_levels) == 2:
                stats['hierarchy_distribution']['primary_secondary'] += 1
            elif len(mood_levels) == 3:
                stats['hierarchy_distribution']['all_three'] += 1

            # Update mood counts
            for mood_info in moods:
                mood = mood_info['mood']
                if mood not in stats['mood_counts']:
                    stats['mood_counts'][mood] = {
                        'total': 0,
                        'primary': 0,
                        'secondary': 0,
                        'tertiary': 0
                    }
                stats['mood_counts'][mood]['total'] += 1
                if mood_info['level'] == 0:
                    stats['mood_counts'][mood]['primary'] += 1
                elif mood_info['level'] == 1:
                    stats['mood_counts'][mood]['secondary'] += 1
                else:
                    stats['mood_counts'][mood]['tertiary'] += 1

        return stats

# ---------------------------
# Siamese Network Definition
# ---------------------------
class SiameseSongEmbedder(nn.Module):
    def __init__(self, input_dim=12, embedding_dim=32):
        super(SiameseSongEmbedder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),  # Increased dropout

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),  # Increased dropout

            nn.Linear(64, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# ---------------------------
# Triplet Creation
# ---------------------------
def create_smart_triplets(dataset, num_triplets=50000):
    """Create triplets considering mood hierarchy and confidence"""
    triplets = []

    # Create mood-based indexing
    mood_to_indices = {}
    for i in range(len(dataset)):
        moods = dataset.get_all_moods(i)
        for mood_info in moods:
            mood = mood_info['mood']
            confidence = mood_info['confidence']
            level = mood_info['level']

            if mood not in mood_to_indices:
                mood_to_indices[mood] = []
            mood_to_indices[mood].append((i, confidence, level))

    if not mood_to_indices:
        raise ValueError("No moods available to create triplets.")

    for _ in tqdm(range(num_triplets), desc="Generating Triplets"):
        # Select anchor mood
        anchor_mood = np.random.choice(list(mood_to_indices.keys()))

        # Select anchor with preference for primary moods
        anchor_candidates = mood_to_indices[anchor_mood]
        # Weight by confidence and hierarchy level
        anchor_weights = [conf * (1.0 if level == 0 else 0.7 if level == 1 else 0.4)
                          for _, conf, level in anchor_candidates]

        if len(anchor_weights) == 0:
            continue

        anchor_weights = np.array(anchor_weights)
        if anchor_weights.sum() == 0:
            anchor_weights = np.ones_like(anchor_weights) / len(anchor_weights)
        else:
            anchor_weights = anchor_weights / anchor_weights.sum()

        anchor_choice = np.random.choice(len(anchor_candidates), p=anchor_weights)
        anchor_idx, anchor_conf, anchor_level = anchor_candidates[anchor_choice]

        # Get all moods for anchor
        anchor_moods = {mood_info['mood'] for mood_info in dataset.get_all_moods(anchor_idx)}

        # Select positive (shares at least one mood)
        positive_candidates = []
        positive_weights = []

        for mood in anchor_moods:
            if mood in mood_to_indices:
                for idx, conf, level in mood_to_indices[mood]:
                    if idx != anchor_idx:
                        # Calculate weight based on confidence and hierarchy
                        weight = conf * (1.0 if level == 0 else 0.7 if level == 1 else 0.4)
                        positive_candidates.append(idx)
                        positive_weights.append(weight)

        if not positive_candidates:
            continue

        positive_weights = np.array(positive_weights)
        if positive_weights.sum() == 0:
            positive_weights = np.ones_like(positive_weights) / len(positive_weights)
        else:
            positive_weights = positive_weights / positive_weights.sum()

        positive_idx = positive_candidates[np.random.choice(len(positive_candidates), p=positive_weights)]

        # Select negative (no shared moods)
        negative_candidates = []
        negative_weights = []

        for mood, indices in mood_to_indices.items():
            if mood not in anchor_moods:
                for idx, conf, level in indices:
                    weight = conf * (1.0 if level == anchor_level else 0.7)
                    negative_candidates.append(idx)
                    negative_weights.append(weight)

        if not negative_candidates:
            continue

        negative_weights = np.array(negative_weights)
        if negative_weights.sum() == 0:
            negative_weights = np.ones_like(negative_weights) / len(negative_weights)
        else:
            negative_weights = negative_weights / negative_weights.sum()

        negative_idx = negative_candidates[np.random.choice(len(negative_candidates), p=negative_weights)]

        triplets.append((anchor_idx, positive_idx, negative_idx))

    # Ensure we have at least some triplets
    if len(triplets) == 0:
        raise ValueError("Could not create any valid triplets. Check your mood data.")

    return np.array(triplets)

# ---------------------------
# Early Stopping Implementation
# ---------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True, save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

# ---------------------------
# Training and Visualization
# ---------------------------
def plot_training_progress(train_losses, val_losses, epoch, lr_history):
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.axvline(x=epoch, color='gray', linestyle='--', alpha=0.5)
    plt.title('Training vs. Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Learning Rate
    plt.subplot(1, 2, 2)
    plt.plot(lr_history, label='Learning Rate', color='green')
    plt.axvline(x=epoch, color='gray', linestyle='--', alpha=0.5)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.close()

def plot_mood_distribution(df):
    plt.figure(figsize=(15, 5))
    sns.countplot(data=df, x='primary_mood', order=df['primary_mood'].value_counts().index)
    plt.title('Distribution of Primary Moods')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.close()

def save_training_artifacts(model, scaler, train_losses, val_losses, unique_moods, timestamp=None, lr_history=None):
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs('models', exist_ok=True)

    model_path = f'models/siamese_model_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': model.encoder[-1].out_features,
        'input_dim': model.encoder[0].in_features
    }, model_path)

    scaler_path = f'models/feature_scaler_{timestamp}.joblib'
    joblib.dump(scaler, scaler_path)

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'best_train_loss': min(train_losses),
        'best_val_loss': min(val_losses),
        'num_epochs': len(train_losses),
        'unique_moods': unique_moods,
        'lr_history': lr_history,
        'timestamp': timestamp
    }

    history_path = f'models/training_history_{timestamp}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"\nSaved training artifacts:")
    print(f"Model: {model_path}")
    print(f"Scaler: {scaler_path}")
    print(f"History: {history_path}")

# ---------------------------
# Training Function
# ---------------------------
def train_model(df, embedding_dim=32, epochs=200, patience=10, num_triplets=50000):
    """
    Train the Siamese Network with enhanced strategies to prevent overfitting.

    Args:
        df (DataFrame): DataFrame containing song data with moods.
        embedding_dim (int): Dimension of the embedding space.
        epochs (int): Maximum number of training epochs.
        patience (int): Patience for early stopping.
        num_triplets (int): Number of triplets to generate per epoch.

    Returns:
        model (nn.Module): Trained Siamese Network model.
        scaler (StandardScaler): Scaler used for feature normalization.
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        best_model_path (str): Path to the best saved model.
    """
    # Data preparation
    df['primary_mood'] = df['primary_mood'].fillna('Unknown')
    df['primary_mood_confidence'] = df['primary_mood_confidence'].fillna(0.0)
    df['secondary_mood'] = df['secondary_mood'].fillna('None')
    df['secondary_mood_confidence'] = df['secondary_mood_confidence'].fillna(0.0)
    df['third_mood'] = df['third_mood'].fillna('None')
    df['third_mood_confidence'] = df['third_mood_confidence'].fillna(0.0)

    # Remove rows with no valid moods
    valid_rows = (
            (df['primary_mood'] != 'Unknown') |
            (df['secondary_mood'] != 'None') |
            (df['third_mood'] != 'None')
    )
    df = df[valid_rows].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid songs found after filtering")

    # Print mood statistics
    print("\nMood Statistics:")
    print(f"Primary moods: {df['primary_mood'].nunique()} unique values")
    print(f"Secondary moods: {df['secondary_mood'].nunique()} unique values")
    print(f"Third moods: {df['third_mood'].nunique()} unique values")

    # Plot mood distribution
    plot_mood_distribution(df)

    # Split and create datasets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    train_dataset = SongDataset(train_df, train=True)
    val_dataset = SongDataset(val_df, scaler=train_dataset.scaler)

    print(f"\nTotal songs: {len(df)}")
    print(f"Training songs: {len(train_df)}")
    print(f"Validation songs: {len(val_df)}")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseSongEmbedder(input_dim=12, embedding_dim=embedding_dim).to(device)

    # Define loss function and optimizer with weight decay for L2 regularization
    criterion = nn.TripletMarginLoss(margin=1.2)  # Increased margin
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight_decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    # Early stopping
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_model_path = f'models/best_model_{timestamp}.pth'
    early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=best_model_path)

    # Training parameters
    batch_size = 32
    train_losses = []
    val_losses = []
    lr_history = []

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        model.train()
        epoch_losses = []

        # Create triplets
        triplets = create_smart_triplets(train_dataset, num_triplets=num_triplets)

        for i in tqdm(range(0, len(triplets), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"):
            batch_triplets = triplets[i:i + batch_size]

            anchors = torch.stack([train_dataset[idx]['features']
                                   for idx in batch_triplets[:, 0]]).to(device)
            positives = torch.stack([train_dataset[idx]['features']
                                     for idx in batch_triplets[:, 1]]).to(device)
            negatives = torch.stack([train_dataset[idx]['features']
                                     for idx in batch_triplets[:, 2]]).to(device)

            optimizer.zero_grad()
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            negative_embeddings = model(negatives)

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_triplets = create_smart_triplets(val_dataset, num_triplets=min(2000, num_triplets // 5))
        val_losses_epoch = []

        with torch.no_grad():
            for i in range(0, len(val_triplets), batch_size):
                batch_triplets = val_triplets[i:i + batch_size]

                anchors = torch.stack([val_dataset[idx]['features']
                                       for idx in batch_triplets[:, 0]]).to(device)
                positives = torch.stack([val_dataset[idx]['features']
                                         for idx in batch_triplets[:, 1]]).to(device)
                negatives = torch.stack([val_dataset[idx]['features']
                                         for idx in batch_triplets[:, 2]]).to(device)

                anchor_embeddings = model(anchors)
                positive_embeddings = model(positives)
                negative_embeddings = model(negatives)

                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                val_losses_epoch.append(loss.item())

        avg_val_loss = np.mean(val_losses_epoch)
        val_losses.append(avg_val_loss)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Early stopping check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # Plot progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_training_progress(train_losses, val_losses, epoch, lr_history)

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Load best model
    model.load_state_dict(torch.load(best_model_path))

    # Save final artifacts
    save_training_artifacts(
        model=model,
        scaler=train_dataset.scaler,
        train_losses=train_losses,
        val_losses=val_losses,
        unique_moods=train_dataset.unique_moods,
        timestamp=timestamp,
        lr_history=lr_history
    )

    return model, train_dataset.scaler, train_losses, val_losses, best_model_path

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # Load DataFrame
    # Replace 'songs_with_moods_20241030_154936.csv' with your actual CSV file path
    df = pd.read_csv('songs_with_moods_20241030_154936.csv')

    # Train the model
    try:
        model, scaler, train_losses, val_losses, best_model_path = train_model(
            df,
            embedding_dim=32,
            epochs=200,
            patience=10,
            num_triplets=50000
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")

    # Final training curve
    plot_training_progress(train_losses, val_losses, len(train_losses) - 1, lr_history=[])

    print(f"\nTraining completed. Best model saved at: {best_model_path}")
