# Music Discovery Application

A full-stack application for music discovery that uses audio features and mood detection to help users find songs they might enjoy. The application includes sophisticated search capabilities, mood-based filtering, and song similarity matching.

## Features

- **Advanced Search**: Search for songs by title, artist, or album with fuzzy matching
- **Mood-Based Discovery**: Filter songs by multiple moods (up to 3 simultaneous moods)
- **Similar Song Discovery**: Find songs similar to ones you like based on audio features
- **Spotify Integration**: Direct links to play songs on Spotify
- **Pagination**: Efficient deep pagination using search_after for large result sets
- **Featured Songs**: Curated list of popular songs
- **Multiple View Modes**: Featured, filtered, and similarity-based views
- **Responsive Design**: Works on desktop and mobile devices

## Tech Stack

### Backend
- FastAPI (Python web framework)
- Elasticsearch (Search engine and vector database)
- Spotipy (Spotify Web API client)
- Pandas (Data processing)

### Frontend
- React
- Tailwind CSS
- Lucide Icons

## Prerequisites

1. Python 3.8+
2. Node.js 14+
3. Elasticsearch 8.x
4. Spotify Developer Account

## Installation

### 1. Elasticsearch Setup

1. Download and install Elasticsearch 8.x from the official website or use docker container (make sure that docker daemon is running)
2. Start Elasticsearch service
3. Verify it's running:
```bash
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.security.http.ssl.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.0 (elasticsearch in docker container)

curl http://localhost:9200 (check docker is running)
```

### 2. Backend Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required Python packages:
```bash
pip install fastapi uvicorn elasticsearch pandas spotipy tqdm
```

3. Set up Spotify API credentials:
   - Create an application in the Spotify Developer Dashboard
   - Get your Client ID and Client Secret
   - Update the credentials in the backend code

4. Prepare your dataset:
   - Ensure you have a CSV file (dataset.csv) with the required columns:
     - track_id
     - track_name
     - artists
     - album_name
     - duration_ms
     - popularity
     - danceability
     - energy
     - key
     - loudness
     - mode
     - speechiness
     - acousticness
     - instrumentalness
     - liveness
     - valence
     - tempo
     - time_signature
     - track_genre (optional)

5. Index your data:
```bash
python index.py
```

6. Start the backend server:
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup

1. In another terminal, go to UI code
```bash
cd SpotifyMood/my-music-app/src/
```

3. Install required dependencies:
```bash
npm install lucide-react tailwindcss @tailwindcss/forms
```

3. Set up Tailwind CSS:
```bash
npx tailwindcss init
```

4. Configure Tailwind CSS by updating `tailwind.config.js`

5. Update the API URL in the frontend code to match your backend server

6. Start the development server:
```bash
npm run dev
```

## API Endpoints

- `POST /songs/search_with_filters`: Search songs with text and mood filters
- `POST /songs/featured`: Get featured songs
- `POST /songs/similar/{track_id}`: Find similar songs
- `POST /songs/by_mood/{mood_slug}`: Get songs by specific mood
- `POST /songs/by_multiple_moods`: Get songs matching multiple moods
- `POST /songs/by_artist`: Search songs by artist
- `POST /songs/search`: General search endpoint

## Available Moods

- Energetic
- Danceable
- Electronic
- Upbeat
- Dark
- Acoustic
- Melancholic
- Dark Dance
- Calm
- Moderate Energy

Demo Video: 

https://github.com/user-attachments/assets/30750533-9094-400a-ac41-7568fc184183




## License

MIT License
