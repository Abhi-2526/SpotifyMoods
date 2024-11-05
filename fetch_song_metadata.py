# fetch_song_metadata.py

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def fetch_song_metadata(track_id):
    SPOTIFY_CLIENT_ID = "e90f1b66779d476fb11f86b325778c45"
    SPOTIFY_CLIENT_SECRET = "0836c55b44f74807bb779c35adf9a392"

    # Initialize Spotify client
    client_credentials_manager = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    try:
        # Fetch track metadata
        track = spotify.track(track_id)

        # Print track metadata
        print(f"Track ID: {track['id']}")
        print(f"Title: {track['name']}")
        print(f"Artist(s): {', '.join(artist['name'] for artist in track['artists'])}")
        print(f"Album: {track['album']['name']}")
        print(f"Release Date: {track['album']['release_date']}")
        print(f"Duration: {track['duration_ms'] // 1000} seconds")
        print(f"Popularity: {track['popularity']}")
        print(f"Preview URL: {track['preview_url']}")
        print(f"External URL: {track['external_urls']['spotify']}")

    except spotipy.SpotifyException as e:
        print(f"Spotify API error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    example_track_id = '5TqzSPA4swEzMKBQtkvcU0'
    fetch_song_metadata(example_track_id)
