# collect_random_tracks.py

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import string
import secrets


def collect_random_track_ids(target_count=10000):
    SPOTIFY_CLIENT_ID = "e90f1b66779d476fb11f86b325778c45"
    SPOTIFY_CLIENT_SECRET = "0836c55b44f74807bb779c35adf9a392"

    # Initialize Spotify client
    client_credentials_manager = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    collected_track_ids = set()
    attempts = 0
    max_attempts = target_count * 10

    while len(collected_track_ids) < target_count and attempts < max_attempts:
        attempts += 1

        query = generate_random_query()

        try:
            # Search for tracks using the random query
            results = spotify.search(q=query, type='track', limit=50)
            tracks = results['tracks']['items']

            if tracks:
                for track in tracks:
                    track_id = track['id']
                    collected_track_ids.add(track_id)
                    if len(collected_track_ids) >= target_count:
                        break
            else:
                continue  # No tracks found, generate a new query

            # Small delay to avoid hitting rate limits
            time.sleep(0.1)

        except spotipy.SpotifyException as e:
            if e.http_status == 429:
                retry_after = int(e.headers.get('Retry-After', 5))
                print(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
                continue
            else:
                print(f"Spotify API error: {e}")
                continue
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

    print(f"Collected {len(collected_track_ids)} unique track IDs.")
    return list(collected_track_ids)


def generate_random_query():
    # Generate a random string of 1 or 2 characters
    length = secrets.choice([1, 2])
    letters = string.ascii_lowercase + string.digits
    random_query = ''.join(secrets.choice(letters) for _ in range(length))
    return random_query


if __name__ == '__main__':
    # Set the target number of tracks to collect
    TARGET_TRACK_COUNT = 100000

    # Collect random track IDs
    track_ids = collect_random_track_ids(target_count=TARGET_TRACK_COUNT)

    # Save the track IDs to a file for later use
    with open('track_ids.txt', 'w') as f:
        for track_id in track_ids:
            f.write(f"{track_id}\n")

    print("Track IDs have been saved to 'track_ids.txt'.")
