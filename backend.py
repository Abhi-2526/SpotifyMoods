from fastapi import FastAPI, HTTPException, Query
from elasticsearch import Elasticsearch
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

class SongResponse(BaseModel):
    track_id: str
    title: str
    artist: str
    album: str
    moods: List[dict]
    audio_features: dict
    popularity: int
    genre: Optional[str] = None

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

@app.get("/songs/by_mood/{mood_slug}", response_model=List[SongResponse])
async def get_songs_by_mood(
        mood_slug: str,
        limit: int = Query(default=10, gt=0, le=50),
        min_confidence: float = Query(default=0.5, gt=0, le=1)
):
    """
    Get songs by mood using mood slug with minimum confidence threshold
    """
    try:
        # Map slug to mood name
        mood = map_slug_to_mood(mood_slug)
        logger.info(f"Searching for mood: {mood} (slug: {mood_slug})")

        query = {
            "query": {
                "nested": {
                    "path": "moods",
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"moods.mood": mood}},  # Corrected field reference
                                {"range": {"moods.confidence": {"gte": min_confidence}}}
                            ]
                        }
                    }
                }
            },
            "size": limit,
            "sort": [
                {"popularity": "desc"}
            ]
        }

        logger.debug(f"Executing query: {query}")

        response = es.search(index="songs", body=query)
        hits = response["hits"]["hits"]

        logger.info(f"Found {len(hits)} matches for mood: {mood}")

        return [hit["_source"] for hit in hits]

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get_songs_by_mood: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/songs/by_multiple_moods", response_model=List[SongResponse])
async def get_songs_by_multiple_moods(
        moods_slugs: List[str] = Query(..., alias="moods", min_items=1, max_items=3),
        limit: int = Query(default=10, gt=0, le=50),
        min_confidence: float = Query(default=0.5, gt=0, le=1)
):
    """
    Get songs that match multiple moods using mood slugs
    """
    try:
        # Map slugs to mood names
        moods = [map_slug_to_mood(slug) for slug in moods_slugs]
        logger.info(f"Searching for moods: {moods} (slugs: {moods_slugs})")

        must_conditions = []
        for mood in moods:
            must_conditions.append({
                "nested": {
                    "path": "moods",
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"moods.mood": mood}},  # Corrected field reference
                                {"range": {"moods.confidence": {"gte": min_confidence}}}
                            ]
                        }
                    }
                }
            })

        query = {
            "query": {
                "bool": {
                    "must": must_conditions
                }
            },
            "size": limit,
            "sort": [
                {"popularity": "desc"}
            ]
        }

        logger.debug(f"Executing query: {query}")

        response = es.search(index="songs", body=query)
        hits = response["hits"]["hits"]

        logger.info(f"Found {len(hits)} matches for moods: {moods}")

        return [hit["_source"] for hit in hits]

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get_songs_by_multiple_moods: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/songs/similar/{track_id}", response_model=List[SongResponse])
async def get_similar_songs(
        track_id: str,
        limit: int = Query(default=10, gt=0, le=50),
        min_score: float = Query(default=0.7, gt=0, le=1)
):
    """
    Get similar songs based on audio features using cosine similarity
    """
    try:
        # First, get the source song
        source_song = es.get(index="songs", id=track_id)
        if not source_song["found"]:
            raise HTTPException(status_code=404, detail="Song not found")

        # Get the feature vector of the source song
        source_vector = source_song["_source"]["feature_vector"]

        # Search for similar songs using cosine similarity
        query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'feature_vector') + 1.0",
                    "params": {"query_vector": source_vector}
                }
            }
        }

        response = es.search(
            index="songs",
            body={
                "query": query,
                "size": limit + 1,  # +1 to account for the source song
                "min_score": min_score
            }
        )

        # Filter out the source song and return similar songs
        similar_songs = [
                            hit["_source"] for hit in response["hits"]["hits"]
                            if hit["_source"]["track_id"] != track_id
                        ][:limit]

        return similar_songs
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get_similar_songs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/songs/by_artist/{artist}", response_model=List[SongResponse])
async def get_songs_by_artist(
        artist: str,
        limit: int = Query(default=10, gt=0, le=50)
):
    """
    Get songs by artist name (supports partial matches)
    """
    try:
        query = {
            "match": {
                "artist": {
                    "query": artist,
                    "fuzziness": "AUTO"
                }
            }
        }

        response = es.search(
            index="songs",
            body={
                "query": query,
                "size": limit,
                "sort": [{"popularity": "desc"}]
            }
        )

        return [hit["_source"] for hit in response["hits"]["hits"]]
    except Exception as e:
        logger.error(f"Error in get_songs_by_artist: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/songs/search", response_model=List[SongResponse])
async def search_songs(
        query: str = Query(..., min_length=2),
        limit: int = Query(default=10, gt=0, le=50)
):
    """
    Search songs by title, artist, or album
    """
    try:
        search_query = {
            "multi_match": {
                "query": query,
                "fields": ["title^3", "artist^2", "album"],
                "fuzziness": "AUTO",
                "operator": "or"
            }
        }

        response = es.search(
            index="songs",
            body={
                "query": search_query,
                "size": limit,
                "sort": [
                    "_score",
                    {"popularity": "desc"}
                ]
            }
        )

        return [hit["_source"] for hit in response["hits"]["hits"]]
    except Exception as e:
        logger.error(f"Error in search_songs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/songs/featured", response_model=List[SongResponse])
async def get_featured_songs(
        limit: int = Query(default=10, gt=0, le=50),
        min_popularity: int = Query(default=70, ge=0, le=100)
):
    """
    Get featured songs based on popularity
    """
    try:
        query = {
            "range": {
                "popularity": {
                    "gte": min_popularity
                }
            }
        }

        response = es.search(
            index="songs",
            body={
                "query": query,
                "size": limit,
                "sort": [
                    {"popularity": "desc"},
                    "_score"
                ]
            }
        )

        return [hit["_source"] for hit in response["hits"]["hits"]]
    except Exception as e:
        logger.error(f"Error in get_featured_songs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class PaginatedResponse(BaseModel):
    items: List[SongResponse]
    total: int
    page: int
    size: int
    total_pages: int


@app.get("/songs/search_with_filters", response_model=PaginatedResponse)
async def search_with_filters(
        query: Optional[str] = None,
        moods: Optional[List[str]] = Query(None),
        page: int = Query(default=1, gt=0),
        size: int = Query(default=10, gt=0, le=50),
        min_confidence: float = Query(default=0.5, gt=0, le=1)
):
    """
    Search songs with optional text query and mood filters, with pagination
    """
    try:
        # Calculate offset
        from_value = (page - 1) * size

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
            mood_conditions = []
            for mood_slug in moods:
                mood = map_slug_to_mood(mood_slug)
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
            if mood_conditions:
                must_conditions.append({
                    "bool": {
                        "should": mood_conditions,
                        "minimum_should_match": 1
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
            "from": from_value,
            "size": size,
            "sort": [
                "_score",
                {"popularity": "desc"}
            ]
        }

        # Execute search
        response = es.search(
            index="songs",
            body=search_query
        )

        # Calculate pagination info
        total_hits = response["hits"]["total"]["value"]
        total_pages = (total_hits + size - 1) // size

        # Format response
        return {
            "items": [hit["_source"] for hit in response["hits"]["hits"]],
            "total": total_hits,
            "page": page,
            "size": size,
            "total_pages": total_pages
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in search_with_filters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/mapping")
async def get_index_mapping():
    """Get the current index mapping for debugging"""
    try:
        mapping = es.indices.get_mapping(index="songs")
        return mapping
    except Exception as e:
        logger.error(f"Error in get_index_mapping: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/sample")
async def get_sample_data():
    """Get a sample document for debugging"""
    try:
        response = es.search(
            index="songs",
            body={
                "size": 1,
                "query": {"match_all": {}}
            }
        )
        if response["hits"]["hits"]:
            return response["hits"]["hits"][0]["_source"]
        return {"message": "No documents found"}
    except Exception as e:
        logger.error(f"Error in get_sample_data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/available_moods")
async def get_available_moods():
    """Get all available moods with slugs in the index"""
    try:
        query = {
            "size": 0,
            "aggs": {
                "unique_moods": {
                    "nested": {
                        "path": "moods"
                    },
                    "aggs": {
                        "mood_values": {
                            "terms": {
                                "field": "moods.mood",
                                "size": 100
                            }
                        }
                    }
                }
            }
        }

        response = es.search(
            index="songs",
            body=query
        )

        moods = []
        if 'aggregations' in response:
            buckets = response['aggregations']['unique_moods']['mood_values']['buckets']
            for bucket in buckets:
                mood_name = bucket['key']
                slug = MOOD_TO_SLUG.get(mood_name, None)
                if slug:
                    moods.append({"name": mood_name, "slug": slug})
                else:
                    logger.warning(f"No slug found for mood: {mood_name}")

        return {
            "total_moods": len(moods),
            "moods": moods
        }
    except Exception as e:
        logger.error(f"Error in get_available_moods: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/index_stats")
async def get_index_stats():
    """Get index statistics for debugging"""
    try:
        # Check if index exists
        if not es.indices.exists(index="songs"):
            return {
                "status": "error",
                "message": "Index 'songs' does not exist"
            }

        # Get document count
        count = es.count(index="songs")

        # Get a sample document if any exist
        sample = None
        if count["count"] > 0:
            response = es.search(
                index="songs",
                body={
                    "size": 1,
                    "query": {"match_all": {}}
                }
            )
            if response["hits"]["hits"]:
                sample = response["hits"]["hits"][0]["_source"]

        return {
            "status": "success",
            "index_exists": True,
            "document_count": count["count"],
            "sample_document": sample
        }
    except Exception as e:
        logger.error(f"Error in get_index_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Music Search API is running",
        "endpoints": [
            "/songs/featured",
            "/songs/by_mood/{mood_slug}",
            "/songs/by_multiple_moods",
            "/songs/similar/{track_id}",
            "/songs/by_artist/{artist}",
            "/songs/search",
            "/debug/mapping",
            "/debug/sample",
            "/debug/available_moods",
            "/debug/index_stats",
            "/debug/check_mood/{mood_slug}",
            "/debug/mood_stats",
            "/debug/mood_query_test/{mood_slug}"
        ]
    }

@app.get("/debug/check_mood/{mood_slug}")
async def check_mood(mood_slug: str):
    """Simple endpoint to check if a mood slug exists in the index"""
    try:
        # Map slug to mood name
        mood = map_slug_to_mood(mood_slug)

        query = {
            "size": 0,
            "query": {
                "nested": {
                    "path": "moods",
                    "query": {
                        "term": {
                            "moods.mood": mood
                        }
                    }
                }
            },
            "aggs": {
                "mood_count": {
                    "nested": {
                        "path": "moods"
                    },
                    "aggs": {
                        "matching_moods": {
                            "filter": {
                                "term": {
                                    "moods.mood": mood
                                }
                            }
                        }
                    }
                }
            }
        }

        result = es.search(index="songs", body=query)

        total_matches = result["aggregations"]["mood_count"]["matching_moods"]["doc_count"]

        # Get a sample song with this mood
        sample_query = {
            "query": {
                "nested": {
                    "path": "moods",
                    "query": {
                        "term": {
                            "moods.mood": mood
                        }
                    }
                }
            },
            "size": 1
        }

        sample_result = es.search(index="songs", body=sample_query)
        sample_song = sample_result["hits"]["hits"][0]["_source"] if sample_result["hits"]["hits"] else None

        return {
            "mood": mood,
            "slug": mood_slug,
            "exists": total_matches > 0,
            "total_songs": total_matches,
            "sample_song": {
                "title": sample_song["title"],
                "artist": sample_song["artist"],
                "moods": sample_song["moods"]
            } if sample_song else None,
            "elasticsearch_status": "connected"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in check_mood: {str(e)}")
        return {
            "mood": mood_slug,
            "exists": False,
            "error": str(e),
            "elasticsearch_status": "error"
        }

@app.get("/debug/mood_stats")
async def get_mood_statistics():
    """
    Get statistics about moods in the index
    """
    try:
        # Get mood distribution
        mood_agg_query = {
            "size": 0,
            "aggs": {
                "all_moods": {
                    "nested": {
                        "path": "moods"
                    },
                    "aggs": {
                        "unique_moods": {
                            "terms": {
                                "field": "moods.mood",
                                "size": 100
                            }
                        },
                        "mood_types": {
                            "terms": {
                                "field": "moods.type",
                                "size": 10
                            }
                        },
                        "confidence_stats": {
                            "stats": {
                                "field": "moods.confidence"
                            }
                        }
                    }
                }
            }
        }

        response = es.search(index="songs", body=mood_agg_query)

        # Get sample documents
        sample_query = {
            "query": {"match_all": {}},
            "size": 5,
            "_source": ["track_id", "title", "artist", "moods"]
        }

        samples = es.search(index="songs", body=sample_query)

        stats_results = {
            "total_documents": es.count(index="songs")["count"],
            "mood_statistics": {
                "unique_moods": [
                    {
                        "mood": bucket["key"],
                        "count": bucket["doc_count"],
                        "slug": MOOD_TO_SLUG.get(bucket["key"], None)
                    }
                    for bucket in response["aggregations"]["all_moods"]["unique_moods"]["buckets"]
                ],
                "mood_types": [
                    {
                        "type": bucket["key"],
                        "count": bucket["doc_count"]
                    }
                    for bucket in response["aggregations"]["all_moods"]["mood_types"]["buckets"]
                ],
                "confidence_stats": response["aggregations"]["all_moods"]["confidence_stats"]
            },
            "sample_documents": [
                {
                    "track_id": hit["_source"]["track_id"],
                    "title": hit["_source"]["title"],
                    "artist": hit["_source"]["artist"],
                    "moods": hit["_source"]["moods"]
                }
                for hit in samples["hits"]["hits"]
            ]
        }

        return stats_results

    except Exception as e:
        logger.error(f"Error in get_mood_statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/mood_query_test/{mood_slug}")
async def test_mood_query(
        mood_slug: str,
        min_confidence: float = Query(default=0.6, gt=0, le=1)
):
    """
    Debug endpoint to test mood query using mood slug
    """
    try:
        # Map slug to mood name
        mood = map_slug_to_mood(mood_slug)

        query = {
            "query": {
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
            },
            "size": 5
        }

        response = es.search(index="songs", body=query)

        return {
            "mood_tested": mood,
            "slug_tested": mood_slug,
            "min_confidence": min_confidence,
            "total_matches": response["hits"]["total"]["value"],
            "sample_matches": [
                {
                    "title": hit["_source"]["title"],
                    "artist": hit["_source"]["artist"],
                    "moods": hit["_source"]["moods"],
                    "score": hit["_score"]
                }
                for hit in response["hits"]["hits"]
            ]
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in test_mood_query: {str(e)}")
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
            "/songs/by_mood/{mood_slug}",
            "/songs/by_multiple_moods",
            "/songs/similar/{track_id}",
            "/songs/by_artist/{artist}",
            "/songs/search",
            "/debug/mapping",
            "/debug/sample",
            "/debug/available_moods",
            "/debug/index_stats",
            "/debug/check_mood/{mood_slug}",
            "/debug/mood_stats",
            "/debug/mood_query_test/{mood_slug}"
        ]
    }

if __name__ == "__main__":
    # Start the server
    uvicorn.run(
        "backend:app",  # Assuming your file is named main.py
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
