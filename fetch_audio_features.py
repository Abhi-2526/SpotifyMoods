# fetch_audio_features.py

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

def fetch_audio_features(track_id):
    try:
        # Fetch audio features
        audio_features = spotify.audio_features([track_id])[0]
        if audio_features is None:
            print(f"No audio features found for track ID: {track_id}")
            return

        # Print audio features
        print(f"Audio Features for Track ID {track_id}:")
        for feature, value in audio_features.items():
            print(f"{feature}: {value}")

    except spotipy.SpotifyException as e:
        print(f"Spotify API error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    SPOTIFY_CLIENT_ID = "e90f1b66779d476fb11f86b325778c45"
    SPOTIFY_CLIENT_SECRET = "0836c55b44f74807bb779c35adf9a392"

    # Initialize Spotify client
    client_credentials_manager = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Example track ID (replace with any valid Spotify track ID)
    example_track_id = '5TqzSPA4swEzMKBQtkvcU0'

    # Fetch and print audio features
    fetch_audio_features(example_track_id)
