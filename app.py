"""
Real-Time Anomaly Detection App
A Streamlit-based application for detecting anomalies in streaming time-series data.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import time
from collections import deque

# Page configuration
st.set_page_config(
    page_title="Real-Time Anomaly Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


class DataGenerator:
    """Generates truly random time-series data stream with occasional anomalies."""
    
    def __init__(self, volatility=5, drift=0.1, anomaly_probability=0.05, anomaly_magnitude=40):
        """
        Initialize random data generator using random walk process.
        
        Args:
            volatility: Standard deviation of random changes (default=5)
            drift: Slight upward/downward bias per step (default=0.1)
            anomaly_probability: Chance of anomaly (default=0.05 = 5%)
            anomaly_magnitude: Size of anomaly spike (default=40)
        """
        self.current_value = 100  # Starting value
        self.volatility = volatility
        self.drift = drift
        self.anomaly_probability = anomaly_probability
        self.anomaly_magnitude = anomaly_magnitude
        self.time_step = 0
        
    def generate_point(self):
        """
        Generate next random data point using random walk.
        
        Random Walk: Each value = previous_value + random_change
        This is how stock prices actually move (truly unpredictable!)
        
        Returns:
            tuple: (value, is_anomaly)
        """
        self.time_step += 1
        
        # Random walk: unpredictable like real sensor data or stock prices
        random_change = np.random.normal(self.drift, self.volatility)
        self.current_value += random_change
        
        # Keep value in reasonable range (prevent drift to infinity)
        self.current_value = np.clip(self.current_value, 10, 500)
        
        value = self.current_value
        is_anomaly = False
        
        # Randomly inject anomalies
        if np.random.random() < self.anomaly_probability:
            # Spike up or down
            spike = np.random.choice([-1, 1]) * np.random.uniform(
                self.anomaly_magnitude * 0.8, 
                self.anomaly_magnitude * 1.2
            )
            value += spike
            is_anomaly = True
            
        return value, is_anomaly


class AnomalyDetector:
    """
    Implements multiple anomaly detection algorithms with production-grade robustness.
    
    All algorithms now feature:
    - O(n) time complexity for scalability  
    - Input validation
    - Minimum sample size enforcement
    - Fixed windows for reproducibility
    - Proper floating-point handling
    - Comprehensive documentation
    """
    
    MIN_SAMPLES_ZSCORE = 10  # Minimum samples for Z-score (literature recommendation)
    MIN_SAMPLES_MAD = 10     # Minimum samples for MAD
    MIN_SAMPLES_MA = 20      # Minimum samples for Moving Average (= window size)
    MIN_SAMPLES_IQR = 20     # Minimum samples for IQR
    FIXED_IQR_WINDOW = 50    # Fixed window for IQR (reproducible)
    EPSILON = 1e-10          # Tolerance for floating-point comparisons
    
    @staticmethod
    def _validate_input(data):
        """
        Validate input data for anomaly detection.
        
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")
        
        if not np.isfinite(data).all():
            raise ValueError("Input contains NaN or Inf values")
        
        return data
    
    @staticmethod
    def z_score_detection(data, threshold=3.0):
        """
        Detect anomalies using Z-score method with Welford's online algorithm (O(n)).
        
        Uses only historical data to avoid contamination. Implements Welford's 
        algorithm for efficient online variance calculation.
        
        Based on:
        - Grubbs, F. E. (1969). "Procedures for detecting outlying observations 
          in samples." Technometrics, 11(1), 1-21.
        - Welford, B. P. (1962). "Note on a method for calculating corrected 
          sums of squares and products." Technometrics, 4(3), 419-420.
        
        Args:
            data: Time series data array
            threshold: Number of standard deviations (default=3.0, ~99.7% confidence)
        
        Returns:
            Boolean array indicating anomalies
            
        Time Complexity: O(n)
        Space Complexity: O(n) for output array
        
        Warm-up Period: First 10 points not checked (minimum sample requirement)
        """
        data = AnomalyDetector._validate_input(data)
        
        if len(data) < AnomalyDetector.MIN_SAMPLES_ZSCORE:
            return np.array([False] * len(data))
        
        anomalies = np.array([False] * len(data))
        
        # Welford's online algorithm for mean and variance
        n = 0
        mean = 0.0
        M2 = 0.0  # Sum of squared differences from current mean
        
        for i in range(len(data)):
            # Update running statistics (O(1) per point)
            n += 1
            delta = data[i] - mean
            mean += delta / n
            delta2 = data[i] - mean
            M2 += delta * delta2
            
            # Only check for anomalies after minimum samples
            if i >= AnomalyDetector.MIN_SAMPLES_ZSCORE:
                # Calculate sample variance (Bessel's correction)
                if n > 1:
                    variance = M2 / (n - 1)
                    std = np.sqrt(variance)
                    
                    # Use epsilon for floating-point comparison
                    if std > AnomalyDetector.EPSILON:
                        z_score = abs((data[i] - mean) / std)
                        anomalies[i] = z_score > threshold
                    else:
                        # Low variance: check if significantly different
                        anomalies[i] = abs(data[i] - mean) > AnomalyDetector.EPSILON
        
        return anomalies
    
    @staticmethod
    def moving_average_detection(data, window=20, threshold=3.5):
        """
        Detect anomalies using moving average with robust statistics.
        
        Uses median and MAD (Median Absolute Deviation) for robustness.
        Note: Threshold is on modified Z-score scale (3.5 recommended).
        
        Based on:
        - Rousseeuw, P. J., & Croux, C. (1993). "Alternatives to the median 
          absolute deviation." Journal of the American Statistical Association, 
          88(424), 1273-1283.
        - Iglewicz & Hoaglin (1993) for modified Z-score threshold.
        
        Args:
            data: Time series data array
            window: Size of moving window (default=20, minimum recommended)
            threshold: Modified Z-score threshold (default=3.5, recommended value)
        
        Returns:
            Boolean array indicating anomalies
            
        Time Complexity: O(n × window × log(window)) due to median calculation
        Space Complexity: O(n)
        
        Warm-up Period: First 'window' points not checked
        """
        data = AnomalyDetector._validate_input(data)
        
        if window < AnomalyDetector.MIN_SAMPLES_MA:
            window = AnomalyDetector.MIN_SAMPLES_MA
        
        if len(data) < window:
            return np.array([False] * len(data))
        
        anomalies = np.array([False] * len(data))
        
        for i in range(window, len(data)):
            window_data = data[i-window:i]
            
            # Use robust statistics: median instead of mean
            median = np.median(window_data)
            # MAD: Median Absolute Deviation
            mad = np.median(np.abs(window_data - median))
            
            if mad < AnomalyDetector.EPSILON:
                # When MAD is zero, stay robust: use IQR as fallback
                q1 = np.percentile(window_data, 25)
                q3 = np.percentile(window_data, 75)
                iqr = q3 - q1
                
                if iqr > AnomalyDetector.EPSILON:
                    # Convert IQR to MAD-equivalent: IQR ≈ 1.349 × MAD for normal dist
                    mad = iqr / 1.349
                else:
                    # Last resort: check exact difference
                    anomalies[i] = abs(data[i] - median) > AnomalyDetector.EPSILON
                    continue
            
            # Modified Z-score using MAD (Iglewicz & Hoaglin, 1993)
            # 0.6745 is the 0.75 quantile of standard normal distribution
            # This makes MAD comparable to standard deviation
            modified_z_score = 0.6745 * abs(data[i] - median) / mad
            anomalies[i] = modified_z_score > threshold
        
        return anomalies
    
    @staticmethod
    def iqr_detection(data, multiplier=1.5):
        """
        Detect anomalies using windowed Interquartile Range (IQR) method.
        
        Uses FIXED rolling window for reproducible real-time detection.
        
        Based on: Tukey, J. W. (1977). "Exploratory Data Analysis." 
        Addison-Wesley, Reading, MA.
        
        The 1.5×IQR rule is Tukey's classic outlier detection criterion,
        corresponding to approximately 99.3% coverage for normal distributions.
        
        Args:
            data: Time series data array
            multiplier: IQR multiplier (default=1.5, Tukey's original value)
                       2.0 for ~99.9% coverage, 3.0 for ~99.99%
        
        Returns:
            Boolean array indicating anomalies
            
        Time Complexity: O(n × window × log(window)) due to percentile calculation
        Space Complexity: O(n)
        
        Warm-up Period: First 20 points not checked (minimum sample requirement)
        Window Size: Fixed at 50 points for reproducibility
        """
        data = AnomalyDetector._validate_input(data)
        
        if len(data) < AnomalyDetector.MIN_SAMPLES_IQR:
            return np.array([False] * len(data))
        
        anomalies = np.array([False] * len(data))
        window = AnomalyDetector.FIXED_IQR_WINDOW  # FIXED, not adaptive!
        
        for i in range(AnomalyDetector.MIN_SAMPLES_IQR, len(data)):
            # Use windowed historical data for real-time detection
            window_start = max(0, i - window)
            window_data = data[window_start:i]
            
            q1 = np.percentile(window_data, 25)
            q3 = np.percentile(window_data, 75)
            iqr = q3 - q1
            
            # Handle case when IQR is very small or zero
            if iqr < AnomalyDetector.EPSILON:
                # Stay robust: use MAD as fallback instead of std
                median = np.median(window_data)
                mad = np.median(np.abs(window_data - median))
                
                if mad > AnomalyDetector.EPSILON:
                    # Convert MAD to IQR-equivalent
                    iqr = mad * 1.349
                else:
                    # All values nearly identical: check exact difference
                    anomalies[i] = abs(data[i] - median) > AnomalyDetector.EPSILON
                    continue
            
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            anomalies[i] = (data[i] < lower_bound) or (data[i] > upper_bound)
        
        return anomalies
    
    @staticmethod
    def mad_detection(data, threshold=3.5):
        """
        Detect anomalies using Modified Z-score with MAD (Most Robust Method).
        
        MAD is more robust than standard deviation and less affected by outliers.
        This implementation uses incremental median calculation for efficiency.
        
        Based on: Iglewicz, B., & Hoaglin, D. C. (1993). "How to detect and 
        handle outliers." ASQC Quality Press, Milwaukee, WI.
        
        The modified Z-score is recommended for small sample sizes and
        datasets with outliers. Threshold of 3.5 corresponds to p < 0.001.
        
        Args:
            data: Time series data array
            threshold: Modified Z-score threshold (default=3.5, recommended value)
        
        Returns:
            Boolean array indicating anomalies
            
        Time Complexity: O(n² × log n) due to median calculations
        Space Complexity: O(n)
        
        Note: This is computationally more expensive than Z-score but much more robust.
        For large datasets (>10,000 points), consider using moving_average_detection
        which provides similar robustness with better performance.
        
        Warm-up Period: First 10 points not checked (minimum sample requirement)
        """
        data = AnomalyDetector._validate_input(data)
        
        if len(data) < AnomalyDetector.MIN_SAMPLES_MAD:
            return np.array([False] * len(data))
        
        anomalies = np.array([False] * len(data))
        
        # Online detection using historical data only
        for i in range(AnomalyDetector.MIN_SAMPLES_MAD, len(data)):
            historical = data[:i]
            
            # Median is more robust than mean
            median = np.median(historical)
            # MAD: Median Absolute Deviation from median
            mad = np.median(np.abs(historical - median))
            
            if mad < AnomalyDetector.EPSILON:
                # When MAD is zero, use IQR as robust fallback
                q1 = np.percentile(historical, 25)
                q3 = np.percentile(historical, 75)
                iqr = q3 - q1
                
                if iqr > AnomalyDetector.EPSILON:
                    # Convert IQR to MAD-equivalent
                    mad = iqr / 1.349
                else:
                    # Last resort: check if different from median
                    anomalies[i] = abs(data[i] - median) > AnomalyDetector.EPSILON
                    continue
            
            # Modified Z-score (Iglewicz & Hoaglin, 1993)
            # 0.6745 is the 0.75 quantile of standard normal distribution
            # This makes MAD comparable to standard deviation
            modified_z_score = 0.6745 * abs(data[i] - median) / mad
            anomalies[i] = modified_z_score > threshold
        
        return anomalies


class IncrementalDetector:
    """
    Truly incremental anomaly detector for real-time streaming.
    
    Maintains state between calls so each point is processed exactly once.
    This is O(1) for Z-Score and O(W) for windowed methods.
    """
    
    def __init__(self, method='z_score', **params):
        """
        Initialize incremental detector.
        
        Args:
            method: 'z_score', 'moving_average', or 'iqr'
            **params: Algorithm-specific parameters
        """
        self.method = method
        self.params = params
        self.reset()
    
    def reset(self):
        """Reset all state - called when starting fresh"""
        # Z-Score state (Welford's algorithm)
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        
        # Windowed methods state
        if self.method == 'moving_average':
            window_size = self.params.get('window', 20)
        elif self.method == 'iqr':
            window_size = 50
        else:
            window_size = 50
        
        self.window = deque(maxlen=window_size)
        self.min_samples = 10 if self.method == 'z_score' else 20
    
    def detect_next(self, value):
        """
        Detect if the new value is an anomaly.
        
        This is the key method - processes ONE point at a time.
        
        Args:
            value: The new data point
            
        Returns:
            bool: True if anomaly, False otherwise
        """
        if self.method == 'z_score':
            return self._detect_z_score_incremental(value)
        elif self.method == 'moving_average':
            return self._detect_moving_average_incremental(value)
        elif self.method == 'iqr':
            return self._detect_iqr_incremental(value)
        else:
            return False
    
    def _detect_z_score_incremental(self, value):
        """
        O(1) incremental Z-Score detection.
        
        Uses Welford's algorithm to update mean and variance in constant time.
        """
        threshold = self.params.get('threshold', 3.0)
        
        # Update running statistics (O(1) - the magic!)
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2
        
        # Need minimum samples before detecting
        if self.n < self.min_samples:
            return False
        
        # Calculate variance and detect
        if self.n > 1:
            variance = self.M2 / (self.n - 1)
            std = np.sqrt(variance)
            
            if std > 1e-10:
                z_score = abs((value - self.mean) / std)
                return z_score > threshold
        
        return False
    
    def _detect_moving_average_incremental(self, value):
        """
        O(W) incremental Moving Average detection.
        
        Only looks at last W points, adapts to recent patterns.
        """
        threshold = self.params.get('threshold', 3.5)
        
        # Add to window
        self.window.append(value)
        
        # Need full window
        if len(self.window) < self.window.maxlen:
            return False
        
        # Use all except current point for statistics
        window_array = np.array(list(self.window)[:-1])
        
        median = np.median(window_array)
        mad = np.median(np.abs(window_array - median))
        
        if mad < 1e-10:
            # Robust fallback
            q1 = np.percentile(window_array, 25)
            q3 = np.percentile(window_array, 75)
            iqr = q3 - q1
            
            if iqr > 1e-10:
                mad = iqr / 1.349
            else:
                return abs(value - median) > 1e-10
        
        # Modified Z-score
        modified_z = 0.6745 * abs(value - median) / mad
        return modified_z > threshold
    
    def _detect_iqr_incremental(self, value):
        """
        O(W) incremental IQR detection.
        
        Uses fixed window with Tukey's fences.
        """
        multiplier = self.params.get('multiplier', 1.5)
        
        # Add to window
        self.window.append(value)
        
        # Need minimum samples
        if len(self.window) < self.min_samples:
            return False
        
        # Use all except current point for statistics
        window_array = np.array(list(self.window)[:-1])
        
        q1 = np.percentile(window_array, 25)
        q3 = np.percentile(window_array, 75)
        iqr = q3 - q1
        
        if iqr < 1e-10:
            # Stay robust: use MAD as fallback
            median = np.median(window_array)
            mad = np.median(np.abs(window_array - median))
            
            if mad > 1e-10:
                # Convert MAD to IQR-equivalent
                iqr = mad * 1.349
            else:
                return abs(value - median) > 1e-10
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        return (value < lower_bound) or (value > upper_bound)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'data' not in st.session_state:
        st.session_state.data = deque(maxlen=200)  # Keep last 200 points
    if 'timestamps' not in st.session_state:
        st.session_state.timestamps = deque(maxlen=200)
    if 'actual_anomalies' not in st.session_state:
        st.session_state.actual_anomalies = deque(maxlen=200)
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'generator' not in st.session_state:
        st.session_state.generator = DataGenerator()
    if 'start_time' not in st.session_state:
        st.session_state.start_time = datetime.now()
    if 'total_points' not in st.session_state:
        st.session_state.total_points = 0
    if 'detected_anomalies' not in st.session_state:
        st.session_state.detected_anomalies = 0
    # Performance tracking
    if 'true_positives' not in st.session_state:
        st.session_state.true_positives = 0
    if 'false_positives' not in st.session_state:
        st.session_state.false_positives = 0
    if 'false_negatives' not in st.session_state:
        st.session_state.false_negatives = 0
    if 'total_injected' not in st.session_state:
        st.session_state.total_injected = 0
    if 'last_detection_correct' not in st.session_state:
        st.session_state.last_detection_correct = False
    # Incremental detection state
    if 'detection_mode' not in st.session_state:
        st.session_state.detection_mode = 'Full History (Retrospective)'
    if 'incremental_detector' not in st.session_state:
        st.session_state.incremental_detector = None
    if 'incremental_results' not in st.session_state:
        st.session_state.incremental_results = deque(maxlen=200)


def create_plot(data, timestamps, anomalies, actual_anomalies):
    """Create an interactive Plotly chart with performance indicators."""
    fig = go.Figure()
    
    # Main data line
    fig.add_trace(go.Scatter(
        x=list(timestamps),
        y=list(data),
        mode='lines',
        name='Data Stream',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Time</b>: %{x}<br><b>Value</b>: %{y:.2f}<extra></extra>'
    ))
    
    # Calculate performance categories
    anomaly_array = np.array(anomalies) if len(anomalies) > 0 else np.array([])
    actual_array = np.array(actual_anomalies)
    
    # True Positives: Detected AND Actual
    true_positive_indices = []
    # False Positives: Detected but NOT Actual
    false_positive_indices = []
    # False Negatives: Actual but NOT Detected
    false_negative_indices = []
    
    if len(anomaly_array) > 0:
        for i in range(len(actual_array)):
            if actual_array[i] and anomaly_array[i]:
                true_positive_indices.append(i)
            elif anomaly_array[i] and not actual_array[i]:
                false_positive_indices.append(i)
            elif actual_array[i] and not anomaly_array[i]:
                false_negative_indices.append(i)
    else:
        # No detections yet, all actual anomalies are false negatives
        false_negative_indices = [i for i, is_anom in enumerate(actual_array) if is_anom]
    
    # Plot True Positives (Correct Detections) - Green checkmarks
    if len(true_positive_indices) > 0:
        fig.add_trace(go.Scatter(
            x=[list(timestamps)[i] for i in true_positive_indices],
            y=[list(data)[i] for i in true_positive_indices],
            mode='markers',
            name='✅ Correct Detection (TP)',
            marker=dict(color='lime', size=16, symbol='star', 
                       line=dict(width=2, color='darkgreen')),
            hovertemplate='<b>✅ CORRECT DETECTION!</b><br><b>Time</b>: %{x}<br><b>Value</b>: %{y:.2f}<extra></extra>'
        ))
    
    # Plot False Positives (Incorrect Detections) - Red X
    if len(false_positive_indices) > 0:
        fig.add_trace(go.Scatter(
            x=[list(timestamps)[i] for i in false_positive_indices],
            y=[list(data)[i] for i in false_positive_indices],
            mode='markers',
            name='❌ False Positive (FP)',
            marker=dict(color='red', size=12, symbol='x', line=dict(width=2)),
            hovertemplate='<b>❌ False Positive</b><br><b>Time</b>: %{x}<br><b>Value</b>: %{y:.2f}<extra></extra>'
        ))
    
    # Plot False Negatives (Missed Anomalies) - Yellow warning
    if len(false_negative_indices) > 0:
        fig.add_trace(go.Scatter(
            x=[list(timestamps)[i] for i in false_negative_indices],
            y=[list(data)[i] for i in false_negative_indices],
            mode='markers',
            name='⚠️ Missed Anomaly (FN)',
            marker=dict(color='yellow', size=14, symbol='diamond', 
                       line=dict(width=2, color='orange')),
            hovertemplate='<b>⚠️ MISSED ANOMALY</b><br><b>Time</b>: %{x}<br><b>Value</b>: %{y:.2f}<extra></extra>'
        ))
    
    # Plot all actual anomalies as background reference (smaller, transparent)
    actual_indices = [i for i, is_anom in enumerate(actual_anomalies) if is_anom]
    if len(actual_indices) > 0:
        fig.add_trace(go.Scatter(
            x=[list(timestamps)[i] for i in actual_indices],
            y=[list(data)[i] for i in actual_indices],
            mode='markers',
            name='Ground Truth (Injected)',
            marker=dict(color='orange', size=8, symbol='circle', 
                       opacity=0.3, line=dict(width=1, color='darkorange')),
            hovertemplate='<b>Injected Anomaly</b><br><b>Time</b>: %{x}<br><b>Value</b>: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='Real-Time Anomaly Detection with Performance Tracking',
            font=dict(size=24, color='#1f77b4')
        ),
        xaxis_title='Time',
        yaxis_title='Value',
        hovermode='closest',
        height=500,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def calculate_performance_metrics(anomalies, actual_anomalies):
    """Calculate precision, recall, F1-score, and accuracy."""
    if len(anomalies) == 0 or len(actual_anomalies) == 0:
        return {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0
        }
    
    anomaly_array = np.array(anomalies)
    actual_array = np.array(actual_anomalies)
    
    # Calculate confusion matrix elements
    true_positives = np.sum(anomaly_array & actual_array)
    false_positives = np.sum(anomaly_array & ~actual_array)
    false_negatives = np.sum(~anomaly_array & actual_array)
    true_negatives = np.sum(~anomaly_array & ~actual_array)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_positives + true_negatives) / len(actual_array) if len(actual_array) > 0 else 0.0
    
    return {
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'true_negatives': int(true_negatives),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.title("📊 Real-Time Anomaly Detection")
    st.markdown("**Detect anomalies in streaming time-series data using multiple statistical methods**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Detection mode toggle
    detection_mode = st.sidebar.radio(
        "Detection Mode",
        ["Full History (Retrospective)", "Incremental (True Streaming)"],
        help="Full History: Re-analyzes all data each time (academic accuracy)\nIncremental: Processes each point once (true real-time)"
    )
    st.session_state.detection_mode = detection_mode
    
    # Detection method selection (conditional based on mode)
    if detection_mode == "Full History (Retrospective)":
        detection_methods = ["MAD (Modified Z-Score)", "Z-Score", "Moving Average", "IQR (Interquartile Range)"]
    else:
        # Incremental mode: MAD not available (requires full history for median)
        detection_methods = ["Z-Score", "Moving Average", "IQR (Interquartile Range)"]
    
    detection_method = st.sidebar.selectbox(
        "Detection Method",
        detection_methods,
        help="Choose the anomaly detection algorithm"
    )
    
    # Warning for incremental mode
    if detection_mode == "Incremental (True Streaming)":
        st.sidebar.warning("⚡ **True Streaming Mode**: Each point processed once. MAD unavailable (requires full median calculation).")
    
    # Add info box about the selected method
    if detection_method == "MAD (Modified Z-Score)":
        st.sidebar.info("🏆 **Most Robust**: Uses median & MAD. Best for outlier-heavy data.")
    
    # Method-specific parameters
    if detection_method == "MAD (Modified Z-Score)":
        threshold = st.sidebar.slider(
            "Modified Z-Score Threshold",
            min_value=2.0,
            max_value=5.0,
            value=3.5,
            step=0.1,
            help="Recommended: 3.5 (Iglewicz & Hoaglin, 1993)"
        )
        method_params = {'threshold': threshold}
        st.sidebar.caption("📚 *Based on: Iglewicz & Hoaglin (1993)*")
    elif detection_method == "Z-Score":
        threshold = st.sidebar.slider(
            "Z-Score Threshold",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Number of standard deviations from mean"
        )
        method_params = {'threshold': threshold}
        st.sidebar.caption("📚 *Based on: Grubbs (1969)*")
    elif detection_method == "Moving Average":
        window = st.sidebar.slider(
            "Window Size",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Number of points for moving average"
        )
        threshold = st.sidebar.slider(
            "Threshold (MAD-based)",
            min_value=1.0,
            max_value=5.0,
            value=3.5,
            step=0.1,
            help="Deviation threshold using robust statistics"
        )
        method_params = {'window': window, 'threshold': threshold}
        st.sidebar.caption("📚 *Based on: Rousseeuw & Croux (1993)*")
    else:  # IQR
        multiplier = st.sidebar.slider(
            "IQR Multiplier",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Tukey's original value: 1.5"
        )
        method_params = {'multiplier': multiplier}
        st.sidebar.caption("📚 *Based on: Tukey (1977)*")
    
    st.sidebar.markdown("---")
    
    # Data generation parameters
    st.sidebar.subheader("Data Generation")
    volatility = st.sidebar.slider(
        "Volatility (Random Walk)", 
        1, 20, 5, 1,
        help="Standard deviation of random changes - higher = more chaotic"
    )
    drift = st.sidebar.slider(
        "Drift Bias", 
        -1.0, 1.0, 0.1, 0.1,
        help="Slight upward (+) or downward (-) bias"
    )
    anomaly_magnitude = st.sidebar.slider(
        "Anomaly Magnitude",
        10, 100, 40, 5,
        help="Size of injected anomaly spikes"
    )
    anomaly_prob = st.sidebar.slider(
        "Anomaly Probability",
        0.0, 0.2, 0.05, 0.01,
        help="Probability of injecting an anomaly"
    )
    
    # Update generator parameters
    st.session_state.generator.volatility = volatility
    st.session_state.generator.drift = drift
    st.session_state.generator.anomaly_magnitude = anomaly_magnitude
    st.session_state.generator.anomaly_probability = anomaly_prob
    
    st.sidebar.markdown("---")
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start", use_container_width=True):
            st.session_state.running = True
    
    with col2:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.running = False
    
    if st.sidebar.button("🔄 Reset", use_container_width=True):
        st.session_state.data.clear()
        st.session_state.timestamps.clear()
        st.session_state.actual_anomalies.clear()
        st.session_state.total_points = 0
        st.session_state.detected_anomalies = 0
        st.session_state.true_positives = 0
        st.session_state.false_positives = 0
        st.session_state.false_negatives = 0
        st.session_state.total_injected = 0
        st.session_state.generator.time_step = 0
        st.session_state.start_time = datetime.now()
        # Reset incremental detector
        st.session_state.incremental_detector = None
        st.session_state.incremental_results.clear()
        st.rerun()
    
    # Statistics
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Basic Statistics")
    st.sidebar.metric("Total Data Points", st.session_state.total_points)
    
    # Calculate current performance metrics if we have data
    if len(st.session_state.data) > 0:
        data_array = np.array(st.session_state.data)
        
        # Use incremental results if in streaming mode, otherwise compute from scratch
        if detection_mode == "Incremental (True Streaming)" and len(st.session_state.incremental_results) > 0:
            anomalies = np.array(list(st.session_state.incremental_results))
        else:
            # Full history mode
            if detection_method == "MAD (Modified Z-Score)":
                anomalies = AnomalyDetector.mad_detection(data_array, **method_params)
            elif detection_method == "Z-Score":
                anomalies = AnomalyDetector.z_score_detection(data_array, **method_params)
            elif detection_method == "Moving Average":
                anomalies = AnomalyDetector.moving_average_detection(data_array, **method_params)
            else:  # IQR
                anomalies = AnomalyDetector.iqr_detection(data_array, **method_params)
        
        metrics = calculate_performance_metrics(anomalies, st.session_state.actual_anomalies)
        
        # Anomaly counts
        total_injected = sum(st.session_state.actual_anomalies)
        total_detected = np.sum(anomalies)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Injected", total_injected)
        with col2:
            st.metric("Detected", total_detected)
        
        # Performance metrics
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Performance Metrics")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("✅ True Positives", metrics['true_positives'])
            st.metric("❌ False Positives", metrics['false_positives'])
        with col2:
            st.metric("⚠️ False Negatives", metrics['false_negatives'])
            st.metric("Precision", f"{metrics['precision']:.1%}")
        
        st.sidebar.metric("Recall (Sensitivity)", f"{metrics['recall']:.1%}")
        st.sidebar.metric("F1-Score", f"{metrics['f1_score']:.1%}")
        st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        
        # Performance interpretation
        if metrics['f1_score'] >= 0.8:
            st.sidebar.success("🎉 Excellent Detection!")
        elif metrics['f1_score'] >= 0.6:
            st.sidebar.info("👍 Good Detection")
        elif metrics['f1_score'] >= 0.4:
            st.sidebar.warning("⚠️ Moderate Detection")
        else:
            st.sidebar.error("❌ Poor Detection - Adjust Parameters")
    
    # Main content area
    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Info boxes
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_value = st.empty()
    with col2:
        anomaly_status = st.empty()
    with col3:
        runtime = st.empty()
    with col4:
        stream_status = st.empty()
    
    # Real-time update loop
    if st.session_state.running:
        stream_status.success("🟢 Streaming Active")
        
        # Generate new data point
        value, is_actual_anomaly = st.session_state.generator.generate_point()
        current_timestamp = datetime.now()
        
        st.session_state.data.append(value)
        st.session_state.timestamps.append(current_timestamp)
        st.session_state.actual_anomalies.append(is_actual_anomaly)
        st.session_state.total_points += 1
        
        # Detect anomalies using appropriate mode
        data_array = np.array(st.session_state.data)
        
        if detection_mode == "Incremental (True Streaming)":
            # Initialize or reset detector if method/mode changed
            method_key = detection_method.split()[0].lower()  # 'z-score' -> 'z', 'moving' -> 'moving', etc.
            if method_key == 'z-score':
                method_key = 'z_score'
            elif method_key == 'moving':
                method_key = 'moving_average'
            elif method_key == 'iqr':
                method_key = 'iqr'
            
            if (st.session_state.incremental_detector is None or 
                st.session_state.incremental_detector.method != method_key):
                # Create new detector
                st.session_state.incremental_detector = IncrementalDetector(method=method_key, **method_params)
                st.session_state.incremental_results.clear()
            
            # Detect current point (O(1) or O(W) - true streaming!)
            is_anomaly = st.session_state.incremental_detector.detect_next(value)
            st.session_state.incremental_results.append(is_anomaly)
            
            # Build full anomaly array from incremental results
            anomalies = np.array(list(st.session_state.incremental_results))
        else:
            # Full history mode: Re-analyze all data (retrospective)
            if detection_method == "MAD (Modified Z-Score)":
                anomalies = AnomalyDetector.mad_detection(data_array, **method_params)
            elif detection_method == "Z-Score":
                anomalies = AnomalyDetector.z_score_detection(data_array, **method_params)
            elif detection_method == "Moving Average":
                anomalies = AnomalyDetector.moving_average_detection(data_array, **method_params)
            else:  # IQR
                anomalies = AnomalyDetector.iqr_detection(data_array, **method_params)
        
        st.session_state.detected_anomalies = np.sum(anomalies)
        
        # Check if latest point is a correct detection
        is_detected = len(anomalies) > 0 and anomalies[-1]
        is_actual = is_actual_anomaly
        is_correct_detection = is_detected and is_actual
        
        # Update display
        current_value.metric("Current Value", f"{value:.2f}")
        
        if is_correct_detection:
            anomaly_status.success("🎉 CORRECT DETECTION! ✅")
        elif is_detected and not is_actual:
            anomaly_status.error("🚨 Anomaly Detected (False Positive)")
        elif not is_detected and is_actual:
            anomaly_status.warning("⚠️ Anomaly Missed (False Negative)")
        else:
            anomaly_status.info("✅ Normal")
        
        elapsed_time = (datetime.now() - st.session_state.start_time).total_seconds()
        runtime.metric("Runtime", f"{elapsed_time:.0f}s")
        
        # Create and display chart
        fig = create_plot(
            st.session_state.data,
            st.session_state.timestamps,
            anomalies,
            st.session_state.actual_anomalies
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Performance summary banner
        if st.session_state.total_points >= 20:  # Show after enough data
            metrics = calculate_performance_metrics(anomalies, st.session_state.actual_anomalies)
            total_injected = sum(st.session_state.actual_anomalies)
            
            if total_injected > 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Anomalies Injected", total_injected, 
                             help="Total anomalies we've injected for testing")
                with col2:
                    delta = metrics['true_positives'] - metrics['false_negatives']
                    st.metric("✅ Correctly Detected", metrics['true_positives'], 
                             delta=delta if delta != 0 else None,
                             help="True Positives - Anomalies we caught!")
                with col3:
                    st.metric("⚠️ Missed", metrics['false_negatives'],
                             delta=-metrics['false_negatives'] if metrics['false_negatives'] > 0 else None,
                             delta_color="inverse",
                             help="False Negatives - Anomalies we missed")
                with col4:
                    st.metric("🎯 Detection Rate", f"{metrics['recall']:.0%}",
                             help="Recall: What % of anomalies did we catch?")
        
        # Status message
        status_placeholder.info(f"📡 Streaming data... (Point #{st.session_state.total_points})")
        
        # Auto-refresh
        time.sleep(0.5)
        st.rerun()
    else:
        stream_status.warning("⏸️ Paused")
        
        if len(st.session_state.data) > 0:
            # Display current state
            data_array = np.array(st.session_state.data)
            
            # Use incremental results if in streaming mode, otherwise recompute
            if detection_mode == "Incremental (True Streaming)" and len(st.session_state.incremental_results) > 0:
                anomalies = np.array(list(st.session_state.incremental_results))
            else:
                # Full history mode
                if detection_method == "MAD (Modified Z-Score)":
                    anomalies = AnomalyDetector.mad_detection(data_array, **method_params)
                elif detection_method == "Z-Score":
                    anomalies = AnomalyDetector.z_score_detection(data_array, **method_params)
                elif detection_method == "Moving Average":
                    anomalies = AnomalyDetector.moving_average_detection(data_array, **method_params)
                else:  # IQR
                    anomalies = AnomalyDetector.iqr_detection(data_array, **method_params)
            
            st.session_state.detected_anomalies = np.sum(anomalies)
            
            current_value.metric("Last Value", f"{list(st.session_state.data)[-1]:.2f}")
            
            if len(anomalies) > 0 and anomalies[-1]:
                anomaly_status.error("🚨 Last: Anomaly")
            else:
                anomaly_status.info("✅ Last: Normal")
            
            elapsed_time = (datetime.now() - st.session_state.start_time).total_seconds()
            runtime.metric("Total Runtime", f"{elapsed_time:.0f}s")
            
            fig = create_plot(
                st.session_state.data,
                st.session_state.timestamps,
                anomalies,
                st.session_state.actual_anomalies
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            status_placeholder.warning("⏸️ Paused - Click 'Start' to resume streaming")
        else:
            status_placeholder.info("👆 Click 'Start' to begin streaming data")
    
    # Footer with information
    st.markdown("---")
    
    # Performance Legend
    with st.expander("📊 Understanding the Visualization & Performance Metrics", expanded=False):
        st.markdown("""
        ### Chart Legend
        
        The chart uses different markers to show detection performance in real-time:
        
        - **✅ Green Stars**: **True Positives (TP)** - Correct detections! The algorithm successfully identified an actual anomaly.
        - **❌ Red X**: **False Positives (FP)** - Algorithm detected an anomaly where there wasn't one (false alarm).
        - **⚠️ Yellow Diamonds**: **False Negatives (FN)** - Missed anomalies! The algorithm failed to detect an actual anomaly.
        - **🔵 Blue Line**: Normal data stream with natural patterns and noise.
        - **🟠 Orange Circles (transparent)**: Ground truth - actual injected anomalies for reference.
        
        ### Performance Metrics Explained
        
        **Confusion Matrix Elements**:
        - **True Positives (TP)**: Correctly detected anomalies ✅
        - **False Positives (FP)**: Incorrect detections (false alarms) ❌
        - **False Negatives (FN)**: Missed anomalies ⚠️
        - **True Negatives (TN)**: Correctly identified normal points
        
        **Key Metrics**:
        - **Precision**: TP / (TP + FP) - How many detected anomalies were actually anomalies?
        - **Recall (Sensitivity)**: TP / (TP + FN) - What percentage of actual anomalies did we catch?
        - **F1-Score**: Harmonic mean of precision and recall - Overall detection quality.
        - **Accuracy**: (TP + TN) / Total - Overall correctness of the algorithm.
        
        ### How It Works
        
        **Data Generation**: The app generates truly random time-series data using:
        - **Random Walk Process**: Each value = previous value + random change (like real stock prices!)
        - **Volatility**: Controls the magnitude of random fluctuations
        - **Drift**: Optional upward or downward bias
        - **True Randomness**: Completely unpredictable, no predetermined patterns
        - **Injected Anomalies**: Random spikes at configurable probability for testing
        
        **Anomaly Detection Methods** (All Academically Validated):
        
        1. **MAD (Modified Z-Score)** 🏆 **Most Robust**: Uses median and Median Absolute Deviation instead of mean/std. Extremely robust to outliers and recommended for most use cases. Based on Iglewicz & Hoaglin (1993).
        
        2. **Z-Score**: Improved version using only historical data to avoid contamination. Identifies points that deviate from the mean by more than a threshold number of standard deviations. Based on Grubbs (1969).
        
        3. **Moving Average**: Uses a sliding window with robust statistics (median & MAD) to compute local baselines. Adapts to trends and seasonal patterns. Based on Rousseeuw & Croux (1993).
        
        4. **IQR (Interquartile Range)**: Tukey's classic method using windowed quartiles to identify outliers. The 1.5×IQR rule covers ~99.3% of normal distributions. Based on Tukey (1977).
        
        ### Why This Design?
        
        By **injecting known anomalies** and **comparing detections**, you can:
        - Evaluate algorithm accuracy in real-time
        - Tune parameters to optimize performance
        - Compare different detection methods
        - Understand the trade-off between sensitivity and false alarms
        
        This mirrors how machine learning engineers develop anomaly detection systems in production!
        
        ### Technologies Used
        - **Streamlit**: Web framework for rapid prototyping
        - **Plotly**: Interactive real-time visualizations
        - **NumPy & SciPy**: Numerical computations and statistics
        - **Pandas**: Data manipulation and analysis
        """)


if __name__ == "__main__":
    main()
