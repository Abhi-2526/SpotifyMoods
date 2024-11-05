import React, { useState, useEffect, memo, useCallback } from 'react';
import { Search, Music, Star, Heart, ChevronLeft, ChevronRight } from 'lucide-react';

// API URL constant
const API_URL = 'http://localhost:8000';

// Define moods with slugs
const MOODS = [
  { name: 'Energetic', slug: 'energetic' },
  { name: 'Danceable', slug: 'danceable' },
  { name: 'Electronic', slug: 'electronic' },
  { name: 'Upbeat', slug: 'upbeat' },
  { name: 'Dark', slug: 'dark' },
  { name: 'Acoustic', slug: 'acoustic' },
  { name: 'Melancholic', slug: 'melancholic' },
  { name: 'Dark Dance', slug: 'dark-dance' },
  { name: 'Calm', slug: 'calm' },
  { name: 'Moderate Energy', slug: 'moderate-energy' }
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
    switch(type) {
      case 'primary': return 'bg-indigo-100 text-indigo-800';
      case 'secondary': return 'bg-purple-100 text-purple-800';
      case 'weak': return 'bg-gray-100 text-gray-800';
      default: return 'bg-blue-100 text-blue-800';
    }
  };

  return (
    <div
      className={`bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow p-4 cursor-pointer ${
        isSelected ? 'ring-2 ring-indigo-500' : ''
      }`}
      onClick={() => onSelect(song)}
    >
      <div className="flex justify-between items-start mb-2">
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900 truncate">{song.title}</h3>
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
            className={`text-xs px-2 py-1 rounded-full ${getMoodColor(mood.type)} flex items-center`}
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
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalItems, setTotalItems] = useState(0);
  const pageSize = 12;

  // Fetch songs function
  // Fetch songs function
const fetchSongs = async (endpoint, params = {}) => {
  setLoading(true);
  setError(null);

  const baseUrl = `${API_URL}${endpoint}`;
  const searchParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      value.forEach(v => searchParams.append(key, v));
    } else if (value !== undefined && value !== null) {
      searchParams.append(key, value);
    }
  });

  const url = `${baseUrl}${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
  console.log('Fetching from:', url);

  try {
    const response = await fetch(url);
    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      throw new Error(errorData?.detail || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    // Handle different response formats
    if (endpoint === '/songs/similar' || endpoint.startsWith('/songs/similar/')) {
      // Similar songs endpoint returns array directly
      setSongs(data || []);
      setTotalPages(1);
      setTotalItems(data?.length || 0);
      setCurrentPage(1);
    } else if (endpoint === '/songs/featured') {
      // Featured songs endpoint returns array directly
      setSongs(data || []);
      setTotalPages(Math.ceil((data?.length || 0) / pageSize));
      setTotalItems(data?.length || 0);
      setCurrentPage(params.page || 1);
    } else {
      // Search and filtered endpoints return paginated response
      setSongs(data.items || data || []);
      setTotalPages(data.total_pages || Math.ceil((data?.length || 0) / pageSize));
      setTotalItems(data.total || data?.length || 0);
      setCurrentPage(data.page || params.page || 1);
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
  const fetchFilteredSongs = useCallback(() => {
    const params = {
      page: currentPage,
      size: pageSize,
    };

    if (searchTerm.trim()) {
      params.query = searchTerm.trim();
    }

    if (selectedMoodSlugs.length > 0) {
      params.moods = selectedMoodSlugs;
    }

    fetchSongs('/songs/search_with_filters', params);
  }, [currentPage, pageSize, searchTerm, selectedMoodSlugs]);

  // Event handlers
  const handleSearch = (e) => {
    e?.preventDefault();
    setCurrentPage(1);
  };

  const handleMoodSelection = (moodSlug) => {
    setSelectedMoodSlugs(prev => {
      if (prev.includes(moodSlug)) {
        return prev.filter(m => m !== moodSlug);
      }
      if (prev.length < 3) {
        return [...prev, moodSlug];
      }
      return prev;
    });
    setCurrentPage(1);
  };

  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
  };

  const handleSongSelect = async (song) => {
  setSelectedSong(song);
  setViewMode('similar');
  setCurrentPage(1);

  try {
    await fetchSongs(`/songs/similar/${song.track_id}`);
  } catch (err) {
    console.error('Error fetching similar songs:', err);
    setError('Failed to fetch similar songs');
  }
};

const handleBackToMain = () => {
  setSelectedSong(null);
  setViewMode('featured');
  setCurrentPage(1);
  fetchSongs('/songs/featured');
};

// Initial load effect
useEffect(() => {
  fetchSongs('/songs/featured');
}, []);

// Main effect for search and filters
useEffect(() => {
  if (viewMode === 'similar') return;

  const timeoutId = setTimeout(() => {
    if (searchTerm || selectedMoodSlugs.length > 0) {
      setViewMode('filtered');
      fetchFilteredSongs();
    } else {
      setViewMode('featured');
      fetchSongs('/songs/featured', { page: currentPage });
    }
  }, 300);

  return () => clearTimeout(timeoutId);
}, [fetchFilteredSongs, searchTerm, selectedMoodSlugs.length, currentPage, viewMode]);
  // Pagination component
  const Pagination = () => (
    <div className="flex items-center justify-between px-4 py-3 sm:px-6">
      <div className="flex justify-between sm:hidden">
        <button
          onClick={() => handlePageChange(currentPage - 1)}
          disabled={currentPage === 1}
          className="relative inline-flex items-center px-4 py-2 text-sm font-medium rounded-md text-gray-700 bg-white border border-gray-300 disabled:opacity-50"
        >
          Previous
        </button>
        <button
          onClick={() => handlePageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
          className="ml-3 relative inline-flex items-center px-4 py-2 text-sm font-medium rounded-md text-gray-700 bg-white border border-gray-300 disabled:opacity-50"
        >
          Next
        </button>
      </div>
      <div className="hidden sm:flex sm:flex-1 sm:items-center sm:justify-between">
        <div>
          <p className="text-sm text-gray-700">
            Showing{' '}
            <span className="font-medium">
              {Math.min((currentPage - 1) * pageSize + 1, totalItems)}
            </span>
            {' '}-{' '}
            <span className="font-medium">
              {Math.min(currentPage * pageSize, totalItems)}
            </span>
            {' '}of{' '}
            <span className="font-medium">{totalItems}</span> results
          </p>
        </div>
        <div>
          <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px">
            <button
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 1}
              className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
            {[...Array(Math.min(5, totalPages))].map((_, idx) => {
              const pageNumber = currentPage - 2 + idx;
              if (pageNumber > 0 && pageNumber <= totalPages) {
                return (
                  <button
                    key={pageNumber}
                    onClick={() => handlePageChange(pageNumber)}
                    className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                      currentPage === pageNumber
                        ? 'z-10 bg-indigo-50 border-indigo-500 text-indigo-600'
                        : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50'
                    }`}
                  >
                    {pageNumber}
                  </button>
                );
              }
              return null;
            })}
            <button
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage === totalPages}
              className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
            >
              <ChevronRight className="h-5 w-5" />
            </button>
          </nav>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
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
              <h3 className="text-lg font-semibold mb-4 text-gray-900">Mood Filters</h3>
              <div className="flex flex-wrap gap-2">{MOODS.map((mood) => (
                  <button
                    key={mood.slug}
                    onClick={() => handleMoodSelection(mood.slug)}
                    className={`px-4 py-2 rounded-full text-sm font-medium transition-colors relative ${
                      selectedMoodSlugs.includes(mood.slug)
                        ? 'bg-indigo-600 text-white shadow-md'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                    disabled={!selectedMoodSlugs.includes(mood.slug) && selectedMoodSlugs.length >= 3}
                    title={
                      !selectedMoodSlugs.includes(mood.slug) && selectedMoodSlugs.length >= 3
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
                Select up to 3 moods to filter songs. Results will show songs matching any of the selected moods.
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