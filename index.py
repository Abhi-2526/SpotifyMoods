import math
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
import time
from mood_classifier import OptimizedMoodDetector
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
elasticsearch_logger = logging.getLogger('elasticsearch')
elasticsearch_logger.setLevel(logging.DEBUG)

# Constants
BATCH_SIZE = 100
ES_INDEX_NAME = 'songs'
ES_SETTINGS = {
    'hosts': [{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],
    'request_timeout': 60,
    'retry_on_timeout': True,
    'max_retries': 3
}


def read_track_data_from_csv(filename):
    """Read track data from CSV file"""
    try:
        logging.info(f"Reading CSV file: {filename}")
        df = pd.read_csv(filename)
        logging.info(f"Successfully read {len(df)} records from CSV")
        return df, df.to_dict('records')
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return None, []


def create_feature_vector(features):
    """Create normalized feature vector for similarity search"""
    try:
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
    except Exception as e:
        logging.error(f"Error creating feature vector: {e}")
        raise


def prepare_document(track_data, detected_moods, feature_vector):
    """Prepare document for indexing"""
    try:
        return {
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
    except Exception as e:
        logging.error(f"Error preparing document: {e}")
        raise


def create_es_index(es, index_name):
    """Create Elasticsearch index with mapping"""
    try:
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
            },
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                    'refresh_interval': '30s'
                }
            }
        }

        exists = es.indices.exists(index=index_name)
        if exists:
            logging.info(f"Index '{index_name}' already exists")
            return

        es.indices.create(index=index_name, body=mapping)
        logging.info(f"Created index '{index_name}' successfully")
    except Exception as e:
        logging.error(f"Error creating index: {e}")
        raise


def create_moods_dataframe(original_df):
    """Create a new DataFrame with mood information"""
    try:
        moods_df = original_df.copy()
        moods_df['primary_mood'] = None
        moods_df['primary_mood_confidence'] = None
        moods_df['secondary_mood'] = None
        moods_df['secondary_mood_confidence'] = None
        moods_df['third_mood'] = None
        moods_df['third_mood_confidence'] = None
        moods_df['all_moods'] = None
        return moods_df
    except Exception as e:
        logging.error(f"Error creating moods DataFrame: {e}")
        raise


def update_moods_df(moods_df, index, detected_moods):
    """Update the moods DataFrame with detected moods"""
    try:
        all_moods = json.dumps([{
            'mood': m['mood'],
            'confidence': m['confidence'],
            'type': m['type']
        } for m in detected_moods])

        moods_df.at[index, 'all_moods'] = all_moods

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
    except Exception as e:
        logging.error(f"Error updating moods DataFrame: {e}")
        raise


def documents_generator(tracks_data, detector, moods_df):
    """Generator function for bulk indexing"""
    for index, track_data in enumerate(tracks_data):
        try:
            logging.debug(f"Processing track {index}: {track_data.get('track_id', 'unknown')}")

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

            detected_moods = detector.detect_moods(features)
            logging.debug(f"Detected moods for track {track_data.get('track_id', 'unknown')}: {detected_moods}")

            update_moods_df(moods_df, index, detected_moods)
            feature_vector = create_feature_vector(track_data)
            doc = prepare_document(track_data, detected_moods, feature_vector)

            yield {
                '_index': ES_INDEX_NAME,
                '_id': track_data['track_id'],
                '_source': doc
            }
        except Exception as e:
            logging.error(f"Error processing track {track_data.get('track_id', 'unknown')}: {e}")
            continue


def save_statistics(moods_df, successful_indexes, failed_indexes):
    """Save mood statistics and processing results"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save moods DataFrame
        csv_filename = f'songs_with_moods_{timestamp}.csv'
        moods_df.to_csv(csv_filename, index=False)
        logging.info(f"Saved moods data to: {csv_filename}")

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
        logging.info(f"Saved mood statistics to: {stats_filename}")
    except Exception as e:
        logging.error(f"Error saving statistics: {e}")
        raise


def main():
    try:
        # Initialize Elasticsearch with detailed logging
        logging.info("Attempting to connect to Elasticsearch...")
        es = Elasticsearch(**ES_SETTINGS)

        # Test connection and get cluster info
        info = es.info()
        logging.info(f"Elasticsearch info: {info}")

        if not es.ping():
            raise Exception("Could not ping Elasticsearch")
        logging.info("Successfully connected to Elasticsearch")

        # Create index with mapping
        create_es_index(es, ES_INDEX_NAME)

        # Read track data
        logging.info("Reading track data from CSV...")
        original_df, tracks_data = read_track_data_from_csv('dataset.csv')
        if not tracks_data:
            logging.error("No tracks data found. Exiting.")
            return

        logging.info(f"Successfully loaded {len(tracks_data)} tracks")
        logging.debug(f"Sample track data: {tracks_data[0] if tracks_data else 'No data'}")

        # Create new DataFrame for moods
        moods_df = create_moods_dataframe(original_df)
        logging.info("Created moods DataFrame")

        # Initialize mood detector
        detector = OptimizedMoodDetector()
        logging.info("Initialized mood detector")

        # Process and index tracks
        successful_indexes = 0
        failed_indexes = 0

        logging.info("Starting bulk indexing...")
        try:
            with tqdm(total=len(tracks_data), desc="Processing songs") as pbar:
                # Create the generator for documents
                actions = documents_generator(tracks_data, detector, moods_df)

                # Perform bulk indexing
                success_count, errors = bulk(
                    es,
                    actions,
                    chunk_size=BATCH_SIZE,
                    refresh=True,
                    raise_on_error=False,
                    stats_only=True
                )

                successful_indexes = success_count
                failed_indexes = len(tracks_data) - successful_indexes
                pbar.update(len(tracks_data))

        except Exception as e:
            logging.error(f"Error during bulk indexing: {e}")
            raise

        # Save statistics
        save_statistics(moods_df, successful_indexes, failed_indexes)

        logging.info("\nIndexing Summary:")
        logging.info(f"Successfully indexed: {successful_indexes} tracks")
        logging.info(f"Failed to index: {failed_indexes} tracks")
        logging.info("Processing complete.")

    except Exception as e:
        logging.error(f"Fatal error in main: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()