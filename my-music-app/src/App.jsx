import React, { useState, useEffect, memo, useCallback } from 'react';
import { Search, Music, Star, Heart, ExternalLink } from 'lucide-react';

// API URL constant
const API_URL = 'http://localhost:8000';

// Define moods with slugs and colors
const MOODS = [
  { name: 'Energetic', slug: 'energetic', color: 'bg-red-200 text-red-800' },
  { name: 'Danceable', slug: 'danceable', color: 'bg-pink-200 text-pink-800' },
  { name: 'Electronic', slug: 'electronic', color: 'bg-purple-200 text-purple-800' },
  { name: 'Upbeat', slug: 'upbeat', color: 'bg-yellow-200 text-yellow-800' },
  { name: 'Dark', slug: 'dark', color: 'bg-gray-800 text-gray-100' },
  { name: 'Acoustic', slug: 'acoustic', color: 'bg-green-200 text-green-800' },
  { name: 'Melancholic', slug: 'melancholic', color: 'bg-blue-200 text-blue-800' },
  { name: 'Dark Dance', slug: 'dark-dance', color: 'bg-indigo-200 text-indigo-800' },
  { name: 'Calm', slug: 'calm', color: 'bg-teal-200 text-teal-800' },
  { name: 'Moderate Energy', slug: 'moderate-energy', color: 'bg-orange-200 text-orange-800' },
];

// Memoized SearchBar component
const SearchBar = memo(({ searchTerm, setSearchTerm, handleSearch }) => (
  <div className="relative w-full max-w-2xl mx-auto">
    <form onSubmit={handleSearch} className="relative">
      <input
        type="text"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder="Search songs, artists, or albums..."
        className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
      />
      <button
        type="submit"
        className="absolute right-3 top-3 text-gray-400 hover:text-gray-600"
      >
        <Search className="w-5 h-5" />
      </button>
    </form>
  </div>
));

// SongCard Component
const SongCard = ({ song, isSelected, onSelect }) => {
  const getMoodColor = (type) => {
    switch (type) {
      case 'primary':
        return 'bg-indigo-100 text-indigo-800';
      case 'secondary':
        return 'bg-purple-100 text-purple-800';
      case 'weak':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-blue-100 text-blue-800';
    }
  };

  return (
    <div
      onClick={() => onSelect(song)}
      className={`bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow p-4 cursor-pointer ${
        isSelected ? 'ring-2 ring-indigo-500' : ''
      }`}
    >
      <div className="flex justify-between items-start mb-2">
        <div className="flex-1">
          {/* Fixed the song title overflow issue by allowing text wrapping */}
          <h3 className="font-semibold text-gray-900 whitespace-normal break-words">
            {song.title}
          </h3>
          <p className="text-sm text-gray-600 truncate">{song.artist}</p>
        </div>
        {song.popularity >= 80 && (
          <Star className="w-5 h-5 text-yellow-400 flex-shrink-0" />
        )}
      </div>

      <div className="flex flex-wrap gap-2 mb-3">
        {song.moods?.map((mood, index) => (
          <span
            key={index}
            className={`text-xs px-2 py-1 rounded-full ${getMoodColor(
              mood.type
            )} flex items-center`}
            title={`Confidence: ${Math.round(mood.confidence * 100)}%`}
          >
            {mood.mood}
            <span className="ml-1 opacity-75">
              {Math.round(mood.confidence * 100)}%
            </span>
          </span>
        ))}
      </div>

      <div className="flex items-center justify-between text-sm text-gray-500">
        <span className="truncate flex-1">{song.album}</span>
        <div className="flex items-center ml-2 flex-shrink-0">
          <Heart
            className={`w-4 h-4 ${
              song.popularity > 75 ? 'text-red-500 fill-current' : 'text-gray-400'
            }`}
          />
          <span className="ml-1">{song.popularity}%</span>
        </div>
      </div>

      {/* Add Spotify Button */}
      {song.spotify_url && (
        <a
          href={song.spotify_url}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-3 inline-flex items-center text-sm text-green-600 hover:text-green-800"
          onClick={(e) => e.stopPropagation()}
        >
          <ExternalLink className="w-4 h-4 mr-1" />
          Play on Spotify
        </a>
      )}
    </div>
  );
};

function MusicApp() {
  // State management
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedMoodSlugs, setSelectedMoodSlugs] = useState([]);
  const [songs, setSongs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedSong, setSelectedSong] = useState(null);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('featured'); // 'featured', 'filtered', or 'similar'

  // Pagination state
  const [searchAfterStack, setSearchAfterStack] = useState([]);
  const [currentSearchAfter, setCurrentSearchAfter] = useState(null);
  const [totalItems, setTotalItems] = useState(0);
  const [resultsCount, setResultsCount] = useState(0); // Number of results shown so far
  const pageSize = 12;

  // Fetch songs function
  const fetchSongs = async (endpoint, body = {}) => {
    setLoading(true);
    setError(null);

    const url = `${API_URL}${endpoint}`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.detail || `HTTP error! status: ${response.status}`
        );
      }

      const data = await response.json();

      setSongs(data.items || []);
      setTotalItems(data.total || 0);
      setCurrentSearchAfter(data.search_after || null);

      // Update results count
      if (body.search_after) {
        setResultsCount((prev) => prev + data.items.length);
      } else {
        setResultsCount(data.items.length);
      }

      // Update searchAfterStack for previous pages
      if (body.search_after) {
        setSearchAfterStack((prev) => [...prev, body.search_after]);
      } else {
        setSearchAfterStack([]);
      }
    } catch (err) {
      console.error('Error fetching songs:', err);
      setError(`Failed to fetch songs: ${err.message}`);
      setSongs([]);
    } finally {
      setLoading(false);
    }
  };

  // Memoized fetch function for filtered songs
  const fetchFilteredSongs = useCallback(
    (searchAfter = null, moods = selectedMoodSlugs) => {
      const body = {
        size: pageSize,
        query: searchTerm.trim() || undefined,
        moods: moods.length > 0 ? moods : undefined,
        search_after: searchAfter,
      };

      fetchSongs('/songs/search_with_filters', body);
    },
    [pageSize, searchTerm, selectedMoodSlugs]
  );

  // Event handlers
  const handleSearch = (e) => {
    e?.preventDefault();
    setViewMode('filtered');
    setCurrentSearchAfter(null);
    setResultsCount(0);
    fetchFilteredSongs(null, selectedMoodSlugs);
  };

  const handleMoodSelection = (moodSlug) => {
    // Compute new moods before updating state
    let newSelectedMoods;
    if (selectedMoodSlugs.includes(moodSlug)) {
      newSelectedMoods = selectedMoodSlugs.filter((m) => m !== moodSlug);
    } else if (selectedMoodSlugs.length < 3) {
      newSelectedMoods = [...selectedMoodSlugs, moodSlug];
    } else {
      newSelectedMoods = selectedMoodSlugs;
    }

    setSelectedMoodSlugs(newSelectedMoods);
    setViewMode('filtered');
    setCurrentSearchAfter(null);
    setResultsCount(0);

    // Fetch songs with the updated moods
    fetchFilteredSongs(null, newSelectedMoods);
  };

  const handleNextPage = () => {
    if (viewMode === 'similar' && selectedSong) {
      fetchSongs(`/songs/similar/${selectedSong.track_id}`, {
        size: pageSize,
        search_after: currentSearchAfter,
      });
    } else if (viewMode === 'featured') {
      fetchSongs('/songs/featured', {
        size: pageSize,
        search_after: currentSearchAfter,
      });
    } else if (viewMode === 'filtered') {
      fetchFilteredSongs(currentSearchAfter, selectedMoodSlugs);
    }
  };

  const handlePreviousPage = () => {
    const previousSearchAfterStack = [...searchAfterStack];
    previousSearchAfterStack.pop(); // Remove current search_after
    const prevSearchAfter = previousSearchAfterStack.pop() || null; // Get previous search_after

    setSearchAfterStack(previousSearchAfterStack);
    setCurrentSearchAfter(prevSearchAfter);

    // Update results count
    setResultsCount((prev) => prev - songs.length);

    if (viewMode === 'similar' && selectedSong) {
      fetchSongs(`/songs/similar/${selectedSong.track_id}`, {
        size: pageSize,
        search_after: prevSearchAfter,
      });
    } else if (viewMode === 'featured') {
      fetchSongs('/songs/featured', {
        size: pageSize,
        search_after: prevSearchAfter,
      });
    } else if (viewMode === 'filtered') {
      fetchFilteredSongs(prevSearchAfter, selectedMoodSlugs);
    }
  };

  const handleSongSelect = (song) => {
    setSelectedSong(song);
    setViewMode('similar');
    setCurrentSearchAfter(null);
    setSearchAfterStack([]);
    setResultsCount(0);
    fetchSongs(`/songs/similar/${song.track_id}`, {
      size: pageSize,
    });
  };

  const handleBackToMain = () => {
    setSelectedSong(null);
    setViewMode('featured');
    setCurrentSearchAfter(null);
    setSearchAfterStack([]);
    setSelectedMoodSlugs([]);
    setSearchTerm('');
    setResultsCount(0);
    fetchSongs('/songs/featured', {
      size: pageSize,
    });
  };

  // Initial load effect
  useEffect(() => {
    // On initial load, set viewMode to 'featured' and fetch featured songs
    setViewMode('featured');
    setCurrentSearchAfter(null);
    setSearchAfterStack([]);
    setResultsCount(0);
    fetchSongs('/songs/featured', {
      size: pageSize,
    });
  }, []);

  // Pagination component
  const Pagination = () => {
    const hasNextPage = currentSearchAfter !== null;
    const hasPreviousPage = searchAfterStack.length > 0;

    return (
      <div className="flex flex-col items-center justify-between px-4 py-3 sm:px-6">
        <div className="text-sm text-gray-700 mb-2">
          Showing{' '}
          <span className="font-medium">
            {resultsCount - songs.length + 1}
          </span>{' '}
          -{' '}
          <span className="font-medium">{resultsCount}</span> of{' '}
          <span className="font-medium">{totalItems}</span> results
        </div>
        <div>
          <button
            onClick={handlePreviousPage}
            disabled={!hasPreviousPage}
            className="relative inline-flex items-center px-4 py-2 mr-2 text-sm font-medium rounded-md text-gray-700 bg-white border border-gray-300 disabled:opacity-50"
          >
            Previous
          </button>
          <button
            onClick={handleNextPage}
            disabled={!hasNextPage}
            className="relative inline-flex items-center px-4 py-2 text-sm font-medium rounded-md text-gray-700 bg-white border border-gray-300 disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-blue-100">
      {/* Header */}
      <nav className="bg-white shadow-sm sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Music className="h-8 w-8 text-indigo-600" />
              <span className="ml-2 text-xl font-semibold text-gray-900">
                Music Discovery
              </span>
            </div>
            {viewMode === 'similar' && (
              <button
                onClick={handleBackToMain}
                className="px-4 py-2 text-sm font-medium text-indigo-600 hover:text-indigo-800"
              >
                ← Back to all songs
              </button>
            )}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Only show search and filters when not in similar songs view */}
        {viewMode !== 'similar' && (
          <>
            {/* Search Section */}
            <div className="mb-8">
              <SearchBar
                searchTerm={searchTerm}
                setSearchTerm={setSearchTerm}
                handleSearch={handleSearch}
              />
            </div>

            {/* Mood Selection */}
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-4 text-gray-900">
                Mood Filters
              </h3>
              <div className="flex flex-wrap gap-2">
                {MOODS.map((mood) => (
                  <button
                    key={mood.slug}
                    onClick={() => handleMoodSelection(mood.slug)}
                    className={`px-4 py-2 rounded-full text-sm font-medium transition-colors relative ${
                      selectedMoodSlugs.includes(mood.slug)
                        ? `${mood.color} shadow-md`
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                    disabled={
                      !selectedMoodSlugs.includes(mood.slug) &&
                      selectedMoodSlugs.length >= 3
                    }
                    title={
                      !selectedMoodSlugs.includes(mood.slug) &&
                      selectedMoodSlugs.length >= 3
                        ? 'You can select up to 3 moods'
                        : ''
                    }
                  >
                    {mood.name}
                    {selectedMoodSlugs.includes(mood.slug) && (
                      <span className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                        {selectedMoodSlugs.indexOf(mood.slug) + 1}
                      </span>
                    )}
                  </button>
                ))}
              </div>
              <div className="mt-2 text-sm text-gray-500">
                Select up to 3 moods to filter songs. Results will show songs
                matching all of the selected moods.
              </div>
            </div>
          </>
        )}

        {/* Show heading for similar songs view */}
        {viewMode === 'similar' && selectedSong && (
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900">
              Similar songs to "{selectedSong.title}" by {selectedSong.artist}
            </h2>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="mb-4 p-4 bg-red-100 text-red-700 rounded">
            {error}
          </div>
        )}

        {/* Songs Grid */}
        {loading ? (
          <div className="flex justify-center items-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-indigo-500 border-t-transparent"></div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
              {songs.map((song) => (
                <SongCard
                  key={song.track_id}
                  song={song}
                  isSelected={selectedSong?.track_id === song.track_id}
                  onSelect={handleSongSelect}
                />
              ))}
            </div>

            {/* Pagination */}
            {songs.length > 0 && <Pagination />}

            {/* No Results Message */}
            {!loading && songs.length === 0 && (
              <div className="text-center py-12">
                <p className="text-gray-500">
                  {viewMode === 'similar'
                    ? 'No similar songs found.'
                    : 'No songs found. Try adjusting your search or mood filters.'}
                </p>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

export default MusicApp;
