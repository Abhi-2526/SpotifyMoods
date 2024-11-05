class OptimizedMoodDetector:
    def __init__(self):
        # Slightly lower thresholds
        self.PRIMARY_MOOD_THRESHOLD = 0.70
        self.SECONDARY_MOOD_THRESHOLD = 0.60
        self.WEAK_MOOD_THRESHOLD = 0.50
        self.MAX_MOODS = 3

        # Adjusted mood rules with broader ranges
        self.mood_rules = {
            'Energetic': {
                'conditions': {
                    'energy': (0.60, 1.0),     # Lowered from 0.67
                    'valence': (0.35, 1.0)     # Lowered from 0.4
                },
                'weight': 0.9
            },
            'Dark': {
                'conditions': {
                    'valence': (0, 0.35),      # Increased from 0.3
                    'energy': (0.45, 1.0)      # Lowered from 0.55
                },
                'weight': 0.85
            },
            'Danceable': {
                'conditions': {
                    'danceability': (0.65, 1.0),  # Lowered from 0.75
                    'energy': (0.45, 1.0)         # Lowered from 0.5
                },
                'weight': 0.9
            },
            'Dark Dance': {
                'conditions': {
                    'danceability': (0.65, 1.0),  # Lowered from 0.7
                    'valence': (0, 0.4),          # Increased from 0.35
                    'energy': (0.45, 1.0)         # Lowered from 0.5
                },
                'weight': 0.85
            },
            'Electronic': {
                'conditions': {
                    'acousticness': (0, 0.3),     # Increased from 0.2
                    'instrumentalness': (0, 0.9)   # Increased from 0.8
                },
                'weight': 0.8
            },
            'Acoustic': {
                'conditions': {
                    'acousticness': (0.6, 1.0),   # Lowered from 0.8
                    'instrumentalness': (0, 0.7)   # Increased from 0.6
                },
                'weight': 0.8
            },
            'Melancholic': {
                'conditions': {
                    'valence': (0, 0.35),         # Increased from 0.3
                    'energy': (0, 0.45),          # Increased from 0.4
                    'tempo': (0, 110)             # Increased from 100
                },
                'weight': 0.85
            },
            'Upbeat': {
                'conditions': {
                    'valence': (0.55, 1.0),       # Lowered from 0.6
                    'energy': (0.55, 1.0),        # Lowered from 0.6
                    'tempo': (90, 180)            # Lowered from 100
                },
                'weight': 0.85
            }
        }

    def get_fallback_mood(self, features):
        """Determine fallback mood based on dominant features"""
        # High energy fallbacks
        if features['energy'] >= 0.6:
            if features['valence'] >= 0.5:
                return {'mood': 'Energetic', 'confidence': 0.55, 'type': 'fallback'}
            else:
                return {'mood': 'Dark', 'confidence': 0.55, 'type': 'fallback'}

        # Dance fallbacks
        if features['danceability'] >= 0.6:
            if features['valence'] >= 0.5:
                return {'mood': 'Danceable', 'confidence': 0.55, 'type': 'fallback'}
            else:
                return {'mood': 'Dark Dance', 'confidence': 0.55, 'type': 'fallback'}

        # Production character fallbacks
        if features['acousticness'] >= 0.5:
            return {'mood': 'Acoustic', 'confidence': 0.55, 'type': 'fallback'}
        if features['acousticness'] <= 0.3:
            return {'mood': 'Electronic', 'confidence': 0.55, 'type': 'fallback'}

        # Emotional fallbacks
        if features['valence'] <= 0.3:
            return {'mood': 'Melancholic', 'confidence': 0.55, 'type': 'fallback'}
        if features['valence'] >= 0.6:
            return {'mood': 'Upbeat', 'confidence': 0.55, 'type': 'fallback'}

        # Final fallback based on energy level
        if features['energy'] >= 0.5:
            return {'mood': 'Moderate Energy', 'confidence': 0.5, 'type': 'fallback'}
        else:
            return {'mood': 'Calm', 'confidence': 0.5, 'type': 'fallback'}

    def detect_moods(self, features):
        """Detect moods with fallback system"""
        mood_scores = []

        for mood, rules in self.mood_rules.items():
            confidence = 1.0
            conditions_met = True

            for feature, (min_val, max_val) in rules['conditions'].items():
                if feature not in features:
                    conditions_met = False
                    break

                value = features[feature]
                if value < min_val or value > max_val:
                    conditions_met = False
                    break

                # More lenient confidence calculation
                range_size = max_val - min_val
                midpoint = (min_val + max_val) / 2
                distance_from_mid = abs(value - midpoint)
                centrality = 1 - (distance_from_mid / (range_size / 2))
                confidence *= (0.85 + (0.15 * centrality))  # Higher base confidence

            if conditions_met:
                final_confidence = confidence * rules['weight']
                mood_scores.append({
                    'mood': mood,
                    'confidence': round(final_confidence, 3),
                    'type': 'primary' if final_confidence >= self.PRIMARY_MOOD_THRESHOLD else
                           'secondary' if final_confidence >= self.SECONDARY_MOOD_THRESHOLD else 'weak'
                })

        # Sort by confidence
        mood_scores.sort(key=lambda x: x['confidence'], reverse=True)

        final_moods = []

        # Add highest confidence mood if available
        if mood_scores and mood_scores[0]['confidence'] >= self.WEAK_MOOD_THRESHOLD:
            final_moods.append(mood_scores[0])

            # Add additional moods if they meet thresholds
            for mood in mood_scores[1:]:
                if len(final_moods) >= self.MAX_MOODS:
                    break
                if mood['confidence'] >= self.SECONDARY_MOOD_THRESHOLD:
                    final_moods.append(mood)

        # If no moods detected, use fallback system
        if not final_moods:
            final_moods = [self.get_fallback_mood(features)]

        return final_moods