import math
import pandas as pd
from elasticsearch import Elasticsearch
from tqdm import tqdm
import time
from mood_classifier import OptimizedMoodDetector
import json
from datetime import datetime


def read_track_data_from_csv(filename):
    """Read track data from CSV file"""
    try:
        df = pd.read_csv(filename)
        return df, df.to_dict('records')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, []


def create_feature_vector(features):
    """Create normalized feature vector for similarity search"""
    tempo_normalized = features['tempo'] / 250
    loudness_normalized = (features['loudness'] + 60) / 60
    key = features['key']
    key_angle = 2 * math.pi * key / 12
    key_sin = math.sin(key_angle)
    key_cos = math.cos(key_angle)

    feature_vector = [
        features['danceability'],
        features['energy'],
        features['valence'],
        tempo_normalized,
        features['speechiness'],
        features['acousticness'],
        features['instrumentalness'],
        features['liveness'],
        loudness_normalized,
        features['mode'],
        key_sin,
        key_cos
    ]
    return feature_vector


def index_song(es, index_name, track_data, detected_moods, feature_vector):
    """Index song data into Elasticsearch"""
    doc = {
        'track_id': track_data['track_id'],
        'title': track_data['track_name'],
        'artist': track_data['artists'],
        'album': track_data['album_name'],
        'duration_ms': track_data['duration_ms'],
        'popularity': track_data['popularity'],
        'audio_features': {
            'danceability': track_data['danceability'],
            'energy': track_data['energy'],
            'key': track_data['key'],
            'loudness': track_data['loudness'],
            'mode': track_data['mode'],
            'speechiness': track_data['speechiness'],
            'acousticness': track_data['acousticness'],
            'instrumentalness': track_data['instrumentalness'],
            'liveness': track_data['liveness'],
            'valence': track_data['valence'],
            'tempo': track_data['tempo'],
            'time_signature': track_data['time_signature']
        },
        'moods': detected_moods,
        'feature_vector': feature_vector,
        'genre': track_data.get('track_genre', None)
    }

    try:
        es.index(index=index_name, id=track_data['track_id'], body=doc)
        return True
    except Exception as e:
        tqdm.write(f"Error indexing track {track_data['track_id']}: {e}")
        return False


def create_moods_dataframe(original_df):
    """Create a new DataFrame with mood information"""
    # Create copies of necessary columns
    moods_df = original_df.copy()

    # Add columns for moods
    moods_df['primary_mood'] = None
    moods_df['primary_mood_confidence'] = None
    moods_df['secondary_mood'] = None
    moods_df['secondary_mood_confidence'] = None
    moods_df['third_mood'] = None
    moods_df['third_mood_confidence'] = None
    moods_df['all_moods'] = None

    return moods_df


def update_moods_df(moods_df, index, detected_moods):
    """Update the moods DataFrame with detected moods"""
    # Convert detected moods to strings for storage
    all_moods = json.dumps([{
        'mood': m['mood'],
        'confidence': m['confidence'],
        'type': m['type']
    } for m in detected_moods])

    # Update the DataFrame
    moods_df.at[index, 'all_moods'] = all_moods

    # Add individual moods
    for i, mood in enumerate(detected_moods):
        if i == 0:
            moods_df.at[index, 'primary_mood'] = mood['mood']
            moods_df.at[index, 'primary_mood_confidence'] = mood['confidence']
        elif i == 1:
            moods_df.at[index, 'secondary_mood'] = mood['mood']
            moods_df.at[index, 'secondary_mood_confidence'] = mood['confidence']
        elif i == 2:
            moods_df.at[index, 'third_mood'] = mood['mood']
            moods_df.at[index, 'third_mood_confidence'] = mood['confidence']


def main():
    # Initialize Elasticsearch
    try:
        es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
        if not es.ping():
            raise Exception("Could not connect to Elasticsearch")
        print("Connected to Elasticsearch")
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
        return

    index_name = 'songs'

    # Create index with mapping if it doesn't exist
    if not es.indices.exists(index=index_name):
        mapping = {
            'mappings': {
                'properties': {
                    'track_id': {'type': 'keyword'},
                    'title': {'type': 'text'},
                    'artist': {'type': 'text'},
                    'album': {'type': 'text'},
                    'duration_ms': {'type': 'integer'},
                    'popularity': {'type': 'integer'},
                    'genre': {'type': 'keyword'},
                    'audio_features': {
                        'properties': {
                            'danceability': {'type': 'float'},
                            'energy': {'type': 'float'},
                            'key': {'type': 'integer'},
                            'loudness': {'type': 'float'},
                            'mode': {'type': 'integer'},
                            'speechiness': {'type': 'float'},
                            'acousticness': {'type': 'float'},
                            'instrumentalness': {'type': 'float'},
                            'liveness': {'type': 'float'},
                            'valence': {'type': 'float'},
                            'tempo': {'type': 'float'},
                            'time_signature': {'type': 'integer'}
                        }
                    },
                    'moods': {
                        'type': 'nested',
                        'properties': {
                            'mood': {'type': 'keyword'},
                            'confidence': {'type': 'float'},
                            'type': {'type': 'keyword'}
                        }
                    },
                    'feature_vector': {
                        'type': 'dense_vector',
                        'dims': 12
                    }
                }
            }
        }
        es.indices.create(index=index_name, body=mapping)
        print(f"Created index '{index_name}'")

    # Read track data
    original_df, tracks_data = read_track_data_from_csv('dataset.csv')
    if not tracks_data:
        print("No tracks data found. Exiting.")
        return

    # Create new DataFrame for moods
    moods_df = create_moods_dataframe(original_df)

    # Initialize mood detector
    detector = OptimizedMoodDetector()

    # Process tracks
    successful_indexes = 0
    failed_indexes = 0

    with tqdm(total=len(tracks_data), desc="Processing songs") as pbar:
        for index, track_data in enumerate(tracks_data):
            try:
                features = {
                    'danceability': track_data['danceability'],
                    'energy': track_data['energy'],
                    'loudness': track_data['loudness'],
                    'speechiness': track_data['speechiness'],
                    'acousticness': track_data['acousticness'],
                    'instrumentalness': track_data['instrumentalness'],
                    'liveness': track_data['liveness'],
                    'valence': track_data['valence'],
                    'tempo': track_data['tempo']
                }

                # Detect moods
                detected_moods = detector.detect_moods(features)

                # Update moods DataFrame
                update_moods_df(moods_df, index, detected_moods)

                # Create feature vector and index to Elasticsearch
                feature_vector = create_feature_vector(track_data)
                if index_song(es, index_name, track_data, detected_moods, feature_vector):
                    successful_indexes += 1
                else:
                    failed_indexes += 1

            except Exception as e:
                tqdm.write(f"Error processing track {track_data.get('track_id', 'unknown')}: {e}")
                failed_indexes += 1

            pbar.update(1)
            time.sleep(0.01)

    # Save the moods DataFrame
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'songs_with_moods_{timestamp}.csv'
    moods_df.to_csv(csv_filename, index=False)
    print(f"\nSaved moods data to: {csv_filename}")

    # Save mood statistics
    mood_stats = moods_df['primary_mood'].value_counts().to_dict()
    stats_filename = f'mood_statistics_{timestamp}.json'
    with open(stats_filename, 'w') as f:
        json.dump({
            'total_songs': len(moods_df),
            'mood_distribution': mood_stats,
            'successful_indexes': successful_indexes,
            'failed_indexes': failed_indexes
        }, f, indent=2)
    print(f"Saved mood statistics to: {stats_filename}")

    print("\nIndexing Summary:")
    print(f"Successfully indexed: {successful_indexes} tracks")
    print(f"Failed to index: {failed_indexes} tracks")
    print("Processing complete.")


if __name__ == '__main__':
    main()