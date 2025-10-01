# 📊 Real-Time Anomaly Detection App

A production-grade web application for detecting anomalies in streaming time-series data using multiple academically validated statistical methods. Built with Streamlit for rapid deployment.

**Live Demo**: `[Coming Soon - Deployment in progress]`

---

## 🎯 What This App Does

This application generates **truly random time-series data** (simulating sensor readings or stock prices) and detects anomalies in **real-time** using four different statistical algorithms. It provides:

- ✅ **Random Data Generation**: True random walk process (unpredictable, like real financial data)
- ✅ **Real-Time Detection**: Both retrospective and truly incremental streaming modes
- ✅ **Interactive Visualization**: Live charts with performance tracking
- ✅ **Multiple Algorithms**: 4 academically validated detection methods
- ✅ **Performance Metrics**: Precision, Recall, F1-Score, Accuracy tracking
- ✅ **Ground Truth Validation**: Compare detected vs. actual anomalies

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation & Running

```bash
# 1. Clone or download this repository
git clone <repository-url>
cd T212\ Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py

# 4. Open your browser to http://localhost:8501
```

### Using Virtual Environment (Recommended)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install and run
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔬 How It Works

### 1. Random Data Generation

The app generates **truly random time-series data** using a **random walk process**:

```
value(t) = value(t-1) + random_change
```

This mimics real-world behavior (stock prices, sensor drift) with:

- **Volatility**: Controls magnitude of random fluctuations
- **Drift**: Optional upward/downward bias
- **No predetermined patterns**: Completely unpredictable
- **Controlled anomaly injection**: Random spikes for testing

**Why random walk?** Unlike deterministic sine waves, this meets the "random data" requirement and simulates realistic streaming data.

### 2. Anomaly Detection Modes

#### **Full History (Retrospective) Mode**

- Re-analyzes all historical data with each new point
- Maximum academic accuracy
- O(n) to O(n²) complexity depending on algorithm
- All 4 algorithms available

#### **Incremental (True Streaming) Mode** ⚡

- Processes each point **exactly once**
- True real-time detection (O(1) or O(W) per point)
- Uses online algorithms (Welford's for Z-Score)
- 3 algorithms available (MAD excluded - requires full median calculation)

**Toggle between modes in the UI to see the difference!**

### 3. Detection Algorithms

All algorithms are academically validated with proper citations in the code.

#### **1. Z-Score Detection**

**Method**: Measures deviation from mean in standard deviation units.

**Formula**: `z = |x - μ| / σ`

**When to use**:

- Normally distributed data
- No major outliers
- Need maximum speed

**Performance**: O(n) using Welford's online algorithm ⚡

**Threshold**: 3.0 (99.7% coverage for normal distribution)

**Academic basis**: Grubbs (1969)

---

#### **2. MAD (Modified Z-Score)** 🏆 **Most Robust**

**Method**: Uses median and Median Absolute Deviation instead of mean/std.

**Formula**: `Modified Z = 0.6745 × |x - median| / MAD`

**When to use**:

- Data with outliers
- Unknown or non-normal distributions
- Small sample sizes
- Maximum robustness needed

**Performance**: O(n² log n) - slower but most reliable

**Threshold**: 3.5 (Iglewicz & Hoaglin recommendation)

**Academic basis**: Iglewicz & Hoaglin (1993)

**Why robust?**: 50% breakdown point - can handle up to 50% outliers!

**Note**: Only available in Full History mode (requires complete median calculation)

---

#### **3. Moving Average Detection**

**Method**: Sliding window with robust statistics (median & MAD).

**When to use**:

- Data with trends or seasonality
- "Normal" changes over time
- Streaming data
- Need adaptability + robustness

**Performance**: O(n × W × log W) where W = window size

**Parameters**:

- Window size: 20 (adjustable)
- Threshold: 3.5 (modified Z-score scale)

**Academic basis**: Rousseeuw & Croux (1993)

**Why windowed?**: Adapts to changing baselines (e.g., temperature changes seasonally)

---

#### **4. IQR (Interquartile Range)**

**Method**: Uses quartiles to define outlier bounds (Tukey's fences).

**Formula**:

```
IQR = Q3 - Q1
Lower bound = Q1 - (multiplier × IQR)
Upper bound = Q3 + (multiplier × IQR)
```

**When to use**:

- Skewed distributions
- Classic, interpretable method
- Visual box plot understanding

**Performance**: O(n × W × log W)

**Multiplier**: 1.5 (Tukey's original value = ~99.3% coverage)

**Academic basis**: Tukey (1977)

---

### 4. Performance Tracking

The app tracks and displays:

- **True Positives (TP)**: ✅ Correct detections
- **False Positives (FP)**: ❌ False alarms
- **False Negatives (FN)**: ⚠️ Missed anomalies
- **Precision**: TP / (TP + FP) - How accurate are detections?
- **Recall**: TP / (TP + FN) - What % of anomalies caught?
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall correctness

**Chart Legend**:

- 🟢 Green Stars: Correct detections (TP)
- 🔴 Red X: False positives (FP)
- 🟡 Yellow Diamonds: Missed anomalies (FN)
- 🔵 Blue Line: Normal data stream
- 🟠 Orange Circles (transparent): Ground truth (injected anomalies)

---

## 🎮 Usage Guide

### Getting Started

1. **Select Detection Mode**:

   - **Full History**: Academic accuracy, all algorithms
   - **Incremental**: True streaming, 3 algorithms (faster)

2. **Choose Algorithm**:

   - MAD for maximum robustness (if available)
   - Z-Score for speed
   - Moving Average for trends
   - IQR for skewed data

3. **Adjust Parameters**:

   - **Data Generation**: Volatility, drift, anomaly probability/magnitude
   - **Algorithm Settings**: Thresholds, window sizes, multipliers

4. **Click ▶️ Start**: Watch real-time detection!

### Customization

**Data Generation Controls**:

- **Volatility (Random Walk)**: 1-20 (higher = more chaotic)
- **Drift Bias**: -1.0 to +1.0 (trend direction)
- **Anomaly Magnitude**: 10-100 (spike size)
- **Anomaly Probability**: 0-20% (injection rate)

**Algorithm Parameters**:
Each method has tunable parameters with recommended defaults based on academic literature.

### Controls

- **▶️ Start**: Begin/resume data streaming
- **⏸️ Pause**: Pause to examine the chart
- **🔄 Reset**: Clear all data and start fresh

---

## 📚 Technologies Used

| Library   | Version | Purpose                    |
| --------- | ------- | -------------------------- |
| Streamlit | 1.28.1  | Web framework and UI       |
| Plotly    | 5.17.0  | Interactive visualizations |
| NumPy     | 1.24.3  | Numerical computations     |
| Pandas    | 2.0.3   | Data manipulation          |
| SciPy     | 1.11.3  | Statistical functions      |

---

## 🏗️ Architecture

```
app.py (1,200 lines)
├── DataGenerator               # True random walk data generation
│   └── generate_point()        # Random walk: value(t) = value(t-1) + Δ
│
├── AnomalyDetector            # Full history algorithms (retrospective)
│   ├── z_score_detection()    # O(n) using Welford's algorithm
│   ├── moving_average_detection() # O(n×W×log W) robust windowed
│   ├── iqr_detection()        # O(n×W×log W) Tukey's method
│   └── mad_detection()        # O(n²×log n) most robust
│
├── IncrementalDetector        # True streaming algorithms (online)
│   ├── detect_next()          # Process one point at a time
│   ├── _detect_z_score_incremental()      # O(1) per point
│   ├── _detect_moving_average_incremental() # O(W) per point
│   └── _detect_iqr_incremental()          # O(W log W) per point
│
└── main()                     # Streamlit UI
    ├── Detection mode toggle
    ├── Algorithm selection
    ├── Real-time visualization
    └── Performance metrics
```

---

## 🔧 Key Implementation Details

### Academic Rigor

Our implementation features:

✅ **Welford's Algorithm**: O(n) online variance calculation for Z-Score (1962)  
✅ **Robust Statistics**: Median and MAD instead of mean/std where appropriate  
✅ **Input Validation**: Checks for NaN, Inf, empty data  
✅ **Minimum Sample Enforcement**: 10-20 points warm-up per algorithm  
✅ **Fixed Windows**: Reproducible results (IQR uses fixed 50-point window)  
✅ **Robust Fallbacks**: Chain of robust methods when variance = 0  
✅ **Floating-Point Safety**: Epsilon comparisons (1e-10) instead of exact equality  
✅ **Proper Citations**: Academic references in code docstrings

### Performance Optimization

| Algorithm                | Time Complexity          | Space | Notes                  |
| ------------------------ | ------------------------ | ----- | ---------------------- |
| Z-Score (Full)           | O(n)                     | O(n)  | Welford's algorithm    |
| Z-Score (Incremental)    | **O(1)** per point       | O(1)  | True streaming!        |
| Moving Avg (Full)        | O(n×W×log W)             | O(n)  | W = window size        |
| Moving Avg (Incremental) | **O(W)** per point       | O(W)  | Only window stored     |
| IQR (Full)               | O(n×W×log W)             | O(n)  | Fixed window           |
| IQR (Incremental)        | **O(W log W)** per point | O(W)  | Percentile calculation |
| MAD (Full)               | O(n²×log n)              | O(n)  | Most robust, slowest   |

**Memory Management**: Uses `deque(maxlen=200)` to prevent unbounded growth.

### Why These Choices?

1. **Random Walk**: Meets "random data" requirement (no deterministic patterns)
2. **Two Detection Modes**: Compare retrospective accuracy vs. streaming speed
3. **Welford's Algorithm**: Enables O(1) incremental Z-Score
4. **Fixed IQR Window**: Ensures reproducible results
5. **MAD Excluded from Incremental**: Median requires full data history
6. **Robust Fallbacks**: Stay robust even with zero variance

For detailed explanations, see `docs/educational_guide.md`.

---

## 🧪 Testing

### Manual Testing

Run the app and verify:

1. **Start/Stop**: Data streams smoothly, pauses correctly
2. **Mode Toggle**: Switch between Full History and Incremental modes
3. **Algorithm Selection**: MAD only appears in Full History mode
4. **Parameter Changes**: Sliders update detection behavior immediately
5. **Reset**: Clears all data and state properly
6. **Performance Metrics**: TP/FP/FN counts update correctly

### Performance Test

Let the app run for 5-10 minutes:

- ✅ Smooth 0.5s updates
- ✅ Memory stays < 200MB
- ✅ No lag or frame drops

---

## 🌐 Deployment

### Option 1: Streamlit Community Cloud (Recommended - FREE)

```bash
# 1. Push to GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main

# 2. Deploy
# - Visit: https://share.streamlit.io
# - Sign in with GitHub
# - Click "New app"
# - Select repository → branch → app.py
# - Click "Deploy"
# ✅ Live in minutes at: https://[your-app-name].streamlit.app
```

### Option 2: Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

```bash
docker build -t anomaly-detector .
docker run -p 8501:8501 anomaly-detector
```

### Option 3: Heroku

Create `Procfile`:

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Deploy:

```bash
heroku create your-app-name
git push heroku main
```

---

## 🔧 Troubleshooting

**App won't start**:

```bash
python --version  # Must be 3.8+
pip install --upgrade -r requirements.txt
```

**Port already in use**:

```bash
streamlit run app.py --server.port=8502
```

**Slow performance**:

- Use Incremental mode instead of Full History
- Reduce data retention: Change `maxlen=200` to `maxlen=100` in line 571
- Use Z-Score instead of MAD

---

## 📈 Project Deliverables (Per Brief)

### ✅ Requirements Met

| Requirement                    | Status      | Implementation                               |
| ------------------------------ | ----------- | -------------------------------------------- |
| **1. Random Data Generation**  | ✅ COMPLETE | Random walk process (truly unpredictable)    |
| **2. Continuous Data Stream**  | ✅ COMPLETE | Auto-refresh with 0.5s updates               |
| **3. Real-Time Visualization** | ✅ COMPLETE | Plotly interactive charts                    |
| **4. Anomaly Detection**       | ✅ COMPLETE | 4 academically validated algorithms          |
| **5. Clear Marking**           | ✅ COMPLETE | Color-coded markers + performance tracking   |
| **6. Web Interface**           | ✅ COMPLETE | Streamlit (publicly deployable)              |
| **7. Controls**                | ✅ COMPLETE | Start/Pause/Reset + extensive parameters     |
| **8. Multiple Methods**        | ✅ COMPLETE | Z-Score, MAD, Moving Average, IQR            |
| **9. Good UX**                 | ✅ COMPLETE | Clean UI, tooltips, warnings, legends        |
| **10. Clean Code**             | ✅ COMPLETE | Documented, validated, academically rigorous |

### Bonus Features Added

- ✅ **Dual Detection Modes**: Retrospective vs. Incremental
- ✅ **Performance Metrics**: Precision, Recall, F1-Score, Accuracy
- ✅ **Ground Truth Validation**: Compare detected vs. actual anomalies
- ✅ **Academic Validation**: All algorithms cite peer-reviewed sources
- ✅ **Optimization**: O(1) incremental Z-Score using Welford's algorithm
- ✅ **Robust Statistics**: Median and MAD for outlier resistance
- ✅ **Educational Documentation**: Complete guide from first principles

---

## 📚 Further Learning

### Documentation

- **Full Educational Guide**: `docs/educational_guide.md` - Learn anomaly detection from first principles
- **Academic References**: See code docstrings for citations

### Key Academic Papers

1. **Grubbs, F. E. (1969)** - Z-score outlier test
2. **Tukey, J. W. (1977)** - IQR method and box plots
3. **Welford, B. P. (1962)** - Online variance algorithm
4. **Iglewicz & Hoaglin (1993)** - MAD and modified z-score
5. **Rousseeuw & Croux (1993)** - Robust statistics

---

## 🤝 Contributing

Contributions welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author

Built for the real-time anomaly detection challenge - demonstrating production-grade statistical methods with academic rigor.

---

## 📞 Support

- **Issues**: Open a GitHub issue
- **Questions**: Check `docs/educational_guide.md` for detailed explanations
- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)

---

**⭐ If you find this useful, please star the repository!**

---

## Version History

- **v2.0** (Current): True random walk + incremental streaming modes
- **v1.5**: Added MAD algorithm + performance metrics
- **v1.0**: Initial release with 3 algorithms

---

**Status**: ✅ All brief requirements met | Ready for deployment
