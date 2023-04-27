
"""
blinking_analyzer.py - Module for analyzing blinking data

This module provides a BlinkingAnalyzer class, which can be used to analyze blinking data.
It contains methods for analyzing the frequency, duration, and intensity of blinks, as well as
performing basic statistical analysis on the data.

Classes:
- BlinkingAnalyzer: Class for analyzing blinking data.

Usage:
from blinking_analyzer import BlinkingAnalyzer

# Load blinking data
blinking_data = load_blinking_data("blinking.csv")

# Analyze blinking data
analyzer = BlinkingAnalyzer(blinking_data)
analysis_results = analyzer.analyze()

"""

class BlinkingAnalyzer:
    """
    Analyzer class for analyzing blinking data.
    """
    
    def __init__(self, data, threshold=0.5):
        """
        Constructor for BlinkingAnalyzer class.

        Args:
            data (numpy.ndarray): 1D array of blinking data.
            threshold (float): threshold value for determining whether a blink occurred (default=0.5).
        """
        self.data = data
        self.threshold = threshold

    def analyze(self):
        """
        Analyze blinking data.

        Returns:
            dict: A dictionary with the analysis results.
        """
        num_blinks = 0
        blink_durations = []
        is_blinking = False
        for i in range(len(self.data)):
            if self.data[i] > self.threshold:
                if not is_blinking:
                    is_blinking = True
                    num_blinks += 1
                    start_time = i
            else:
                if is_blinking:
                    is_blinking = False
                    end_time = i
                    duration = end_time - start_time
                    blink_durations.append(duration)
        if is_blinking:
            end_time = len(self.data)
            duration = end_time - start_time
            blink_durations.append(duration)

        avg_blink_duration = sum(blink_durations) / len(blink_durations)

        analysis_results = {
            'num_blinks': num_blinks,
            'blink_durations': blink_durations,
            'avg_blink_duration': avg_blink_duration
        }

        return analysis_results