import React, { useState, useEffect, memo } from 'react';
import { Search, Music, Radio, Star, Heart } from 'lucide-react';

// API URL constant
const API_URL = 'http://localhost:8000';

// Define moods with slugs to handle spaces and special characters
const MOODS = [
  { name: 'Energetic', slug: 'energetic' },          // 37,904
  { name: 'Danceable', slug: 'danceable' },          // 26,560
  { name: 'Electronic', slug: 'electronic' },        // 23,553
  { name: 'Upbeat', slug: 'upbeat' },                // 22,078
  { name: 'Dark', slug: 'dark' },                    // 20,619
  { name: 'Acoustic', slug: 'acoustic' },            // 18,848
  { name: 'Melancholic', slug: 'melancholic' },      // 6,732
  { name: 'Dark Dance', slug: 'dark-dance' },        // 5,292
  { name: 'Calm', slug: 'calm' },                    // 384
  { name: 'Moderate Energy', slug: 'moderate-energy' } // 279
];

// Memoized SearchBar to prevent unnecessary re-renders
const SearchBar = memo(({ searchTerm, setSearchTerm, handleSearch }) => (
  <div className="relative w-full max-w-2xl mx-auto">
    <form onSubmit={handleSearch} className="relative">
      <input
        type="text"
        value={searchTerm}
        onChange={(e) => {
          setSearchTerm(e.target.value);
        }}
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

function MusicApp() {
  // State management
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedMoodSlugs, setSelectedMoodSlugs] = useState([]);
  const [songs, setSongs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('discover');
  const [selectedSong, setSelectedSong] = useState(null);
  const [error, setError] = useState(null);
  const [lastRequestUrl, setLastRequestUrl] = useState(null);

  // Helper: Map slugs to mood names
  const slugToMood = (slug) => {
    const mood = MOODS.find(m => m.slug === slug);
    return mood ? mood.name : slug;
  };

  // Updated fetchSongs function to properly handle array parameters
  const fetchSongs = async (endpoint, params = {}) => {
    setLoading(true);
    setError(null);

    const baseUrl = `${API_URL}${endpoint}`;

    // Create URLSearchParams instance
    const searchParams = new URLSearchParams();

    // Handle arrays properly by adding multiple entries with the same key
    Object.entries(params).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        value.forEach(v => searchParams.append(key, v));
      } else {
        searchParams.append(key, value);
      }
    });

    const url = `${baseUrl}${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
    setLastRequestUrl(url);
    console.log('Fetching from:', url);

    try {
      const response = await fetch(url);
      console.log('Response status:', response.status);

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Received data:', data);
      setSongs(data);
    } catch (err) {
      console.error('Error fetching songs:', err);
      setError(`Failed to fetch songs: ${err.message}`);
      setSongs([]);
    } finally {
      setLoading(false);
    }
  };

  // Event handlers
  const handleSearch = async (e) => {
    e?.preventDefault();
    if (!searchTerm.trim()) return;

    console.log('Performing search for:', searchTerm);
    fetchSongs('/songs/search', { query: searchTerm });
  };

  const handleMoodSelection = (moodSlug) => {
    console.log('Handling mood selection:', moodSlug);
    if (selectedMoodSlugs.includes(moodSlug)) {
      setSelectedMoodSlugs(selectedMoodSlugs.filter(m => m !== moodSlug));
    } else if (selectedMoodSlugs.length < 3) {
      setSelectedMoodSlugs([...selectedMoodSlugs, moodSlug]);
    }
  };

  const handleSongSelect = async (song) => {
    console.log('Selected song:', song);
    setSelectedSong(song);
    if (song) {
      fetchSongs(`/songs/similar/${song.track_id}`);
    }
  };

  // Effects
  useEffect(() => {
    if (selectedMoodSlugs.length > 0) {
      console.log('Selected moods changed:', selectedMoodSlugs);
      // Pass mood slugs as an array, fetchSongs will handle it properly
      fetchSongs('/songs/by_multiple_moods', { moods: selectedMoodSlugs });
    } else {
      fetchSongs('/songs/featured');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedMoodSlugs]);

  useEffect(() => {
    console.log('Active tab changed:', activeTab);
    if (activeTab === 'discover') {
      fetchSongs('/songs/featured');
    } else if (activeTab === 'search') {
      // Optionally fetch featured songs or handle differently
      // For now, do nothing
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  // Component definitions
  const SongCard = ({ song }) => {
    const getMoodColor = (type) => {
      switch(type) {
        case 'primary':
          return 'bg-indigo-100 text-indigo-800';
        case 'secondary':
          return 'bg-purple-100 text-purple-800';
        case 'weak':
          return 'bg-gray-100 text-gray-800';
        case 'fallback':
          return 'bg-blue-100 text-blue-800';
        default:
          return 'bg-gray-100 text-gray-800';
      }
    };

    return (
      <div
        className={`bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow p-4 cursor-pointer ${
          selectedSong?.track_id === song.track_id ? 'ring-2 ring-indigo-500' : ''
        }`}
        onClick={() => handleSongSelect(song)}
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

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Music className="h-8 w-8 text-indigo-600" />
              <span className="ml-2 text-xl font-semibold text-gray-900">
                Music Discovery
              </span>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setActiveTab('discover')}
                className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'discover'
                    ? 'bg-indigo-100 text-indigo-700'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                }`}
              >
                <Radio className="inline-block w-4 h-4 mr-2" />
                Discover
              </button>
              <button
                onClick={() => setActiveTab('search')}
                className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'search'
                    ? 'bg-indigo-100 text-indigo-700'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                }`}
              >
                <Search className="inline-block w-4 h-4 mr-2" />
                Search
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Search Section */}
        {activeTab === 'search' && (
          <div className="mb-8">
            <SearchBar searchTerm={searchTerm} setSearchTerm={setSearchTerm} handleSearch={handleSearch} />
          </div>
        )}

        {/* Mood Selection */}
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900">Mood Filters</h3>
          <div className="flex flex-wrap gap-2">
            {MOODS.map((mood) => (
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
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {songs.map((song) => (
              <SongCard key={song.track_id} song={song} />
            ))}
          </div>
        )}

        {/* No Results Message */}
        {!loading && songs.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-500">
              No songs found. Try adjusting your search or mood filters.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

export default MusicApp;
