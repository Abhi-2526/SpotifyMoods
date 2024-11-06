from fastapi import FastAPI, HTTPException, Body, Query
from elasticsearch import Elasticsearch, NotFoundError
from typing import List, Optional
import math
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging

app = FastAPI(title="Music Search API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

# Define mood slug mappings
SLUG_TO_MOOD = {
    "energetic": "Energetic",
    "danceable": "Danceable",
    "electronic": "Electronic",
    "upbeat": "Upbeat",
    "dark": "Dark",
    "acoustic": "Acoustic",
    "melancholic": "Melancholic",
    "dark-dance": "Dark Dance",
    "calm": "Calm",
    "moderate-energy": "Moderate Energy"
}

MOOD_TO_SLUG = {v: k for k, v in SLUG_TO_MOOD.items()}

class Mood(BaseModel):
    mood: str
    confidence: float
    type: str

class SongResponse(BaseModel):
    track_id: str
    title: str
    artist: str
    album: str
    moods: List[Mood]
    audio_features: dict
    popularity: int
    genre: Optional[str] = None

class PaginatedResponse(BaseModel):
    items: List[SongResponse]
    total: int
    size: int
    search_after: Optional[List] = None  # Used for `search_after` pagination
    sort_field: str
    sort_order: str

def create_feature_vector(features):
    """Create normalized feature vector for similarity search"""
    tempo_normalized = features['tempo'] / 250
    loudness_normalized = (features['loudness'] + 60) / 60
    key = features['key']
    key_angle = 2 * math.pi * key / 12
    key_sin = math.sin(key_angle)
    key_cos = math.cos(key_angle)

    return [
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

def map_slug_to_mood(slug: str) -> str:
    """Map a slug to its corresponding mood name."""
    mood = SLUG_TO_MOOD.get(slug.lower())
    if not mood:
        logger.warning(f"Invalid mood slug received: {slug}")
        raise HTTPException(status_code=400, detail=f"Invalid mood slug: {slug}")
    return mood

@app.post("/songs/search_with_filters", response_model=PaginatedResponse)
async def search_with_filters(
        body: dict = Body(...)
):
    """
    Search songs with optional text query and mood filters, using search_after for deep pagination.
    The text search is fuzzy, while mood filters are exact matches.
    """
    try:
        query = body.get('query')
        moods = body.get('moods')
        size = body.get('size', 12)
        sort_field = body.get('sort_field', 'popularity')
        sort_order = body.get('sort_order', 'desc')
        search_after = body.get('search_after')

        # Build the query
        must_conditions = []

        # Add text search if query provided
        if query:
            must_conditions.append({
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "artist^2", "album"],
                    "fuzziness": "AUTO",
                    "operator": "or"
                }
            })

        # Add mood filters if provided
        if moods:
            mood_names = [map_slug_to_mood(slug) for slug in moods]
            # Build a nested query to match all selected moods
            nested_must_conditions = []
            for mood_name in mood_names:
                nested_must_conditions.append({
                    "nested": {
                        "path": "moods",
                        "query": {
                            "term": {"moods.mood": mood_name}
                        }
                    }
                })
            # Use a bool must to require all moods
            must_conditions.append({
                "bool": {
                    "must": nested_must_conditions
                }
            })

        # If no conditions, match all
        if not must_conditions:
            must_conditions.append({"match_all": {}})

        # Build the final query
        search_query = {
            "query": {
                "bool": {
                    "must": must_conditions
                }
            },
            "size": size,
            "sort": [
                {sort_field: sort_order},
                {"track_id": "asc"}  # Ensure uniqueness
            ],
            "track_total_hits": True
        }

        if search_after:
            search_query["search_after"] = search_after

        # Execute search
        response = es.search(
            index="songs",
            body=search_query
        )

        # Extract sort values for the last hit
        hits = response["hits"]["hits"]
        if hits:
            last_sort = hits[-1]["sort"]
        else:
            last_sort = None

        # Format response
        return {
            "items": [hit["_source"] for hit in hits],
            "total": response["hits"]["total"]["value"],
            "size": size,
            "sort_field": sort_field,
            "sort_order": sort_order,
            "search_after": last_sort
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in search_with_filters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/songs/featured", response_model=PaginatedResponse)
async def get_featured_songs(
        body: dict = Body(...)
):
    """
    Get featured songs based on popularity, using search_after for pagination
    """
    try:
        size = body.get('size', 12)
        sort_field = body.get('sort_field', 'popularity')
        sort_order = body.get('sort_order', 'desc')
        search_after = body.get('search_after')
        min_popularity = body.get('min_popularity', 70)

        query = {
            "range": {
                "popularity": {
                    "gte": min_popularity
                }
            }
        }

        search_query = {
            "query": query,
            "size": size,
            "sort": [
                {sort_field: sort_order},
                {"track_id": "asc"}  # Ensure uniqueness
            ],
            "track_total_hits": True
        }

        if search_after:
            search_query["search_after"] = search_after

        response = es.search(
            index="songs",
            body=search_query
        )

        hits = response["hits"]["hits"]
        if hits:
            last_sort = hits[-1]["sort"]
        else:
            last_sort = None

        return {
            "items": [hit["_source"] for hit in hits],
            "total": response["hits"]["total"]["value"],
            "size": size,
            "sort_field": sort_field,
            "sort_order": sort_order,
            "search_after": last_sort
        }
    except Exception as e:
        logger.error(f"Error in get_featured_songs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/songs/similar/{track_id}", response_model=PaginatedResponse)
async def get_similar_songs(
        track_id: str,
        body: dict = Body(...)
):
    """
    Get similar songs based on audio features using cosine similarity, using search_after for pagination
    """
    try:
        size = body.get('size', 12)
        search_after = body.get('search_after')
        min_score = body.get('min_score', 0.7)

        # First, get the source song
        try:
            source_song = es.get(index="songs", id=track_id)
        except NotFoundError:
            raise HTTPException(status_code=404, detail="Song not found")

        # Get the feature vector of the source song
        source_vector = source_song["_source"]["feature_vector"]

        # Search for similar songs using cosine similarity
        query = {
            "script_score": {
                "query": {"bool": {"must_not": {"term": {"track_id": track_id}}}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'feature_vector') + 1.0",
                    "params": {"query_vector": source_vector}
                }
            }
        }

        search_query = {
            "query": query,
            "size": size,
            "min_score": min_score,
            "sort": [
                {"_score": "desc"},
                {"track_id": "asc"}  # Ensure uniqueness
            ],
            "track_total_hits": True
        }

        if search_after:
            search_query["search_after"] = search_after

        response = es.search(
            index="songs",
            body=search_query
        )

        hits = response["hits"]["hits"]
        if hits:
            last_sort = hits[-1]["sort"]
        else:
            last_sort = None

        similar_songs = [hit["_source"] for hit in hits]

        return {
            "items": similar_songs,
            "total": response["hits"]["total"]["value"],
            "size": size,
            "sort_field": "_score",
            "sort_order": "desc",
            "search_after": last_sort
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get_similar_songs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/songs/by_mood/{mood_slug}", response_model=PaginatedResponse)
async def get_songs_by_mood(
        mood_slug: str,
        body: dict = Body(...)
):
    """
    Get songs by mood using mood slug with minimum confidence threshold, using search_after for pagination
    """
    try:
        size = body.get('size', 12)
        sort_field = body.get('sort_field', 'popularity')
        sort_order = body.get('sort_order', 'desc')
        search_after = body.get('search_after')
        min_confidence = body.get('min_confidence', 0.5)

        # Map slug to mood name
        mood = map_slug_to_mood(mood_slug)
        logger.info(f"Searching for mood: {mood} (slug: {mood_slug})")

        query = {
            "nested": {
                "path": "moods",
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"moods.mood": mood}},
                            {"range": {"moods.confidence": {"gte": min_confidence}}}
                        ]
                    }
                }
            }
        }

        search_query = {
            "query": query,
            "size": size,
            "sort": [
                {sort_field: sort_order},
                {"track_id": "asc"}  # Ensure uniqueness
            ],
            "track_total_hits": True
        }

        if search_after:
            search_query["search_after"] = search_after

        response = es.search(
            index="songs",
            body=search_query
        )

        hits = response["hits"]["hits"]
        if hits:
            last_sort = hits[-1]["sort"]
        else:
            last_sort = None

        return {
            "items": [hit["_source"] for hit in hits],
            "total": response["hits"]["total"]["value"],
            "size": size,
            "sort_field": sort_field,
            "sort_order": sort_order,
            "search_after": last_sort
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get_songs_by_mood: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/songs/by_multiple_moods", response_model=PaginatedResponse)
async def get_songs_by_multiple_moods(
        body: dict = Body(...)
):
    """
    Get songs that match multiple moods using mood slugs, using search_after for pagination
    """
    try:
        moods_slugs = body.get('moods', [])
        if not moods_slugs:
            raise HTTPException(status_code=400, detail="At least one mood slug must be provided.")
        if len(moods_slugs) > 3:
            raise HTTPException(status_code=400, detail="Maximum of 3 moods allowed.")

        size = body.get('size', 12)
        sort_field = body.get('sort_field', 'popularity')
        sort_order = body.get('sort_order', 'desc')
        search_after = body.get('search_after')
        min_confidence = body.get('min_confidence', 0.5)

        # Map slugs to mood names
        moods = [map_slug_to_mood(slug) for slug in moods_slugs]
        logger.info(f"Searching for moods: {moods} (slugs: {moods_slugs})")

        mood_conditions = []
        for mood in moods:
            mood_conditions.append({
                "nested": {
                    "path": "moods",
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"moods.mood": mood}},
                                {"range": {"moods.confidence": {"gte": min_confidence}}}
                            ]
                        }
                    }
                }
            })

        query = {
            "bool": {
                "should": mood_conditions,
                "minimum_should_match": 1
            }
        }

        search_query = {
            "query": query,
            "size": size,
            "sort": [
                {sort_field: sort_order},
                {"track_id": "asc"}  # Ensure uniqueness
            ],
            "track_total_hits": True
        }

        if search_after:
            search_query["search_after"] = search_after

        response = es.search(
            index="songs",
            body=search_query
        )

        hits = response["hits"]["hits"]
        if hits:
            last_sort = hits[-1]["sort"]
        else:
            last_sort = None

        return {
            "items": [hit["_source"] for hit in hits],
            "total": response["hits"]["total"]["value"],
            "size": size,
            "sort_field": sort_field,
            "sort_order": sort_order,
            "search_after": last_sort
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get_songs_by_multiple_moods: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/songs/by_artist", response_model=PaginatedResponse)
async def get_songs_by_artist(
        body: dict = Body(...)
):
    """
    Get songs by artist name (supports partial matches), using search_after for pagination
    """
    try:
        artist = body.get('artist')
        if not artist:
            raise HTTPException(status_code=400, detail="Artist name must be provided.")

        size = body.get('size', 12)
        sort_field = body.get('sort_field', 'popularity')
        sort_order = body.get('sort_order', 'desc')
        search_after = body.get('search_after')

        query = {
            "match": {
                "artist": {
                    "query": artist,
                    "fuzziness": "AUTO"
                }
            }
        }

        search_query = {
            "query": query,
            "size": size,
            "sort": [
                {sort_field: sort_order},
                {"track_id": "asc"}
            ],
            "track_total_hits": True
        }

        if search_after:
            search_query["search_after"] = search_after

        response = es.search(
            index="songs",
            body=search_query
        )

        hits = response["hits"]["hits"]
        if hits:
            last_sort = hits[-1]["sort"]
        else:
            last_sort = None

        return {
            "items": [hit["_source"] for hit in hits],
            "total": response["hits"]["total"]["value"],
            "size": size,
            "sort_field": sort_field,
            "sort_order": sort_order,
            "search_after": last_sort
        }
    except Exception as e:
        logger.error(f"Error in get_songs_by_artist: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/songs/search", response_model=PaginatedResponse)
async def search_songs(
        body: dict = Body(...)
):
    """
    Search songs by title, artist, or album, using search_after for pagination
    """
    try:
        query_text = body.get('query')
        if not query_text:
            raise HTTPException(status_code=400, detail="Query text must be provided.")

        size = body.get('size', 12)
        sort_field = body.get('sort_field', '_score')
        sort_order = body.get('sort_order', 'desc')
        search_after = body.get('search_after')

        search_query = {
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["title^3", "artist^2", "album"],
                    "fuzziness": "AUTO",
                    "operator": "or"
                }
            },
            "size": size,
            "sort": [
                {sort_field: sort_order},
                {"track_id": "asc"}
            ],
            "track_total_hits": True
        }

        if search_after:
            search_query["search_after"] = search_after

        response = es.search(
            index="songs",
            body=search_query
        )

        hits = response["hits"]["hits"]
        if hits:
            last_sort = hits[-1]["sort"]
        else:
            last_sort = None

        return {
            "items": [hit["_source"] for hit in hits],
            "total": response["hits"]["total"]["value"],
            "size": size,
            "sort_field": sort_field,
            "sort_order": sort_order,
            "search_after": last_sort
        }
    except Exception as e:
        logger.error(f"Error in search_songs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup Event: Check Elasticsearch Connection and Index
@app.on_event("startup")
def startup_event():
    try:
        if not es.ping():
            logger.error("Cannot connect to Elasticsearch")
            raise Exception("Elasticsearch connection failed")
        logger.info("Successfully connected to Elasticsearch")

        # Check if 'songs' index exists
        if not es.indices.exists(index="songs"):
            logger.error("Index 'songs' does not exist in Elasticsearch")
            raise Exception("Index 'songs' not found")
        else:
            doc_count = es.count(index="songs")["count"]
            logger.info(f"Index 'songs' exists with {doc_count} documents")
    except Exception as e:
        logger.error(f"Startup Error: {e}")
        raise

# Root Endpoint
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Music Search API is running",
        "endpoints": [
            "/songs/featured",
            "/songs/search_with_filters",
            "/songs/similar/{track_id}",
            "/songs/by_mood/{mood_slug}",
            "/songs/by_multiple_moods",
            "/songs/by_artist",
            "/songs/search"
            # ... other endpoints ...
        ]
    }

if __name__ == "__main__":
    # Start the server
    uvicorn.run(
        "backend:app",  # Assuming your file is named backend.py
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
