import torch
import joblib


def load_model_for_inference():
    # Load model
    checkpoint = torch.load('song_siamese_model.pth')
    model = SiameseSongEmbedder(
        input_dim=checkpoint['input_dim'],
        embedding_dim=checkpoint['embedding_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load scaler
    scaler = joblib.load('song_feature_scaler.joblib')

    return model, scaler


def get_embedding(model, scaler, features):
    # Scale features
    scaled_features = scaler.transform(features.reshape(1, -1))

    # Get embedding
    with torch.no_grad():
        embedding = model(torch.FloatTensor(scaled_features))

    return embedding.numpy()