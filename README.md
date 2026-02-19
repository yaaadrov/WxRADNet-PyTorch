# WxRADNet-PyTorch

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive framework for **aircraft thunderstorm avoidance** using deep learning-based precipitation nowcasting and advanced pathfinding algorithms. The system processes weather radar data from the Finnish Meteorological Institute (FMI), predicts storm evolution with sequence-to-sequence models, and calculates optimal flight paths through dynamic hazardous weather zones.

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Usage Guide](#usage-guide)
  - [Data Pipeline](#data-pipeline)
  - [Static Pathfinding](#static-pathfinding)
  - [Dynamic Pathfinding](#dynamic-pathfinding)
  - [Masked Dynamic Pathfinding](#masked-dynamic-pathfinding)
  - [Thunderstorm Prediction](#thunderstorm-prediction)
  - [Deep Learning Models](#deep-learning-models)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Model Architectures](#model-architectures)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

### Weather Data Processing
- **Automated Data Pipeline**: Downloads GeoTIFF radar images from FMI's public S3 bucket
- **Smart Segmentation**: Extracts informative regions with configurable intensity thresholds
- **Data Augmentation**: Rotation-based augmentation for training robustness

### Deep Learning Nowcasting
- **Multiple Architectures**: RNN, LSTM, GRU, ConvRNN, ConvLSTM, ConvGRU
- **Seq2Seq Framework**: Encoder-decoder structure for spatiotemporal prediction
- **Pre-trained Models**: 6 pre-trained checkpoints included

### Pathfinding Algorithms
- **Static Avoider**: Visibility graph-based shortest path around static obstacles
- **Dynamic Avoider**: Sliding window approach for time-varying weather
- **Masked Dynamic Avoider**: Spatial masking for efficient large-scale operations
- **Fine-tuning Strategies**: Greedy shortcut finding and smooth angle optimization

### Hull Generation
- **Convex Hull**: Simplified obstacle boundaries
- **Concave Hull**: Detailed obstacle boundaries preserving shape complexity

---

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/WxRADNet-PyTorch.git
cd WxRADNet-PyTorch

# Install dependencies
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/your-username/WxRADNet-PyTorch.git
cd WxRADNet-PyTorch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.2.2 | Deep learning framework |
| numpy | >=2.4.2 | Numerical computing |
| geopandas | >=1.1.2 | Geospatial data handling |
| shapely | >=2.1.2 | Geometry operations |
| networkx | (via scipy) | Graph algorithms |
| rasterio | >=1.4.4 | GeoTIFF processing |
| pydantic | >=2.12.5 | Configuration management |

---

## Quick Start

### 1. Download and Process Radar Data

```bash
python thund_avoider/scripts/run_parser.py
```

This downloads GeoTIFF images from FMI, segments them into informative regions, applies augmentation, and saves processed data as `.npy` files.

### 2. Generate Obstacle Polygons

```bash
python thund_avoider/scripts/run_preprocessor.py
```

Converts raster radar data into vector polygons (convex and concave hulls) representing thunderstorm hazard zones.

### 3. Run Pathfinding

```bash
# Static pathfinding (single timestamp)
python thund_avoider/scripts/run_static_avoider.py

# Dynamic pathfinding (time-varying weather)
python thund_avoider/scripts/run_dynamic_avoider.py

# Masked dynamic pathfinding (spatial masking for large areas)
python thund_avoider/scripts/run_masked_dynamic_avoider.py
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           WxRADNet-PyTorch                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │    Parser    │───▶│ Preprocessor │───▶│     Pathfinding          │   │
│  │              │    │              │    │                          │   │
│  │ • FMI S3     │    │ • Raster→Vec │    │ ┌──────────────────────┐ │   │
│  │ • Segment    │    │ • Hull Gen   │    │ │   Static Avoider     │ │   │
│  │ • Augment    │    │ • Buffering  │    │ │   • Visibility Graph │ │   │
│  └──────────────┘    └──────────────┘    │ │   • Shortest Path    │ │   │
│                                          │ └──────────────────────┘ │   │
│  ┌──────────────┐                        │ ┌──────────────────────┐ │   │
│  │   Models     │                        │ │   Dynamic Avoider    │ │   │
│  │              │                        │ │   • Sliding Window   │ │   │
│  │ • RNN/LSTM   │───────────────────────▶│ │   • Master Graph     │ │   │
│  │ • ConvRNN    │    Predictions         │ │   • Fine-tuning      │ │   │
│  │ • ConvLSTM   │                        │ └──────────────────────┘ │   │
│  │ • ConvGRU    │                        │ ┌──────────────────────┐ │   │
│  └──────────────┘                        │ │   Masked Dynamic     │ │   │
│                                          │ │   • Spatial Masking  │ │   │
│                                          │ │   • Moving BBox      │ │   │
│                                          │ └──────────────────────┘ │   │
│                                          └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
FMI S3 Bucket (GeoTIFF)
        │
        ▼
┌───────────────┐
│    Parser     │  Download, segment, augment
└───────┬───────┘
        │ .npy files
        ▼
┌───────────────┐
│  Preprocessor │  Raster→Vector, hull generation
└───────┬───────┘
        │ .csv files (WKT geometry)
        ▼
┌───────────────┐
│   Avoider     │  Pathfinding algorithms
└───────┬───────┘
        │ Results
        ▼
   Output Files
```

---

## Usage Guide

### Data Pipeline

#### Using the Parser

```python
from thund_avoider.services.parser.parser import Parser
from thund_avoider.settings import settings

parser = Parser(settings.parser_config)

# Download and process data for a date range
parser.get_data()
```

#### Using the Preprocessor

```python
from datetime import datetime
from pathlib import Path
from thund_avoider.services.preprocessor import Preprocessor
from thund_avoider.settings import settings

preprocessor = Preprocessor(settings.preprocessor_config)

# Process a specific timestamp
gdf = preprocessor.get_gdf_for_current_date(
    current_date=datetime(2024, 5, 30, 15, 0),
)

# Save to CSV
preprocessor.save_geodataframe_to_csv(gdf, Path("data/2024_05_30_15_00.csv"))
```

### Static Pathfinding

The static avoider finds the shortest path around weather obstacles at a single point in time.

```python
from shapely import Point
from thund_avoider.services.static_avoider import StaticAvoider
from thund_avoider.settings import settings

avoider = StaticAvoider(settings.static_avoider_config)

# Find shortest path
result = avoider.find_shortest_path(
    gdf=gdf,                          # GeoDataFrame with obstacles
    strategy="concave",               # or "convex"
    A=Point(300000, 6800000),         # Start point (ETRS89/TM35FIN)
    B=Point(400000, 6900000),         # End point
)

# Access results
print(f"Path found: {result.success}")
print(f"Path length: {len(result.path)} points")
print(f"Distance: {result.distance_m:.2f} meters")
```

### Dynamic Pathfinding

The dynamic avoider handles time-varying weather using a sliding window approach.

```python
from shapely import Point
from thund_avoider.services.dynamic_avoider import DynamicAvoider
from thund_avoider.settings import settings

avoider = DynamicAvoider(settings.dynamic_avoider_config)

# Extract time keys from data directory
time_keys = avoider.extract_time_keys(Path("data/2024_05_30"))

# Collect obstacles for all time steps
dict_obstacles = avoider.collect_obstacles(Path("data/2024_05_30"), time_keys)

# Create master graph
G_master, time_valid_edges = avoider.create_master_graph(
    time_keys=time_keys,
    dict_obstacles=dict_obstacles,
)

# Perform pathfinding with fine-tuning
result = avoider.sliding_window_pathfinding(
    start=Point(300000, 6800000),
    end=Point(400000, 6900000),
    window_size=3,
    time_keys=time_keys,
    dict_obstacles=dict_obstacles,
    G_master=G_master,
    time_valid_edges=time_valid_edges,
    with_fine_tuning=True,
)

# Access results
print(f"Success: {result.success}")
print(f"Segments: {result.num_segments}")
print(f"Fine-tuning iterations: {result.fine_tuning_iters}")
```

### Masked Dynamic Pathfinding

The masked dynamic avoider uses spatial masking to efficiently handle large areas.

```python
from shapely import Point
from thund_avoider.services.masked_dynamic_avoider import MaskedDynamicAvoider
from thund_avoider.settings import settings

avoider = MaskedDynamicAvoider(
    masked_preprocessor_config=settings.masked_preprocessor_config,
    dynamic_avoider_config=settings.dynamic_avoider_config,
    predictor_config=settings.predictor_config,
)

# Perform masked pathfinding with fine-tuning
result = avoider.perform_pathfinding_with_finetuning_masked(
    current_pos=Point(300000, 6800000),
    end=Point(400000, 6900000),
    current_time_index=0,
    window_size=3,
    num_preds=7,              # Number of prediction time steps available
    time_keys=time_keys,
    dict_obstacles=dict_obstacles,
    masking_strategy="wide",  # "center", "left", "right", or "wide"
    prediction_mode="deterministic",  # or "predictive"
)

# Access results
print(f"Success: {result.success}")
print(f"Path valid: {result.is_pred_path_valid}")  # Only in predictive mode
```

#### Prediction Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `deterministic` | Uses actual radar data for all time steps | Real-time pathfinding with current data |
| `predictive` | Uses ML predictions for future time steps | Proactive route planning |

In `predictive` mode, the `is_pred_path_valid` field indicates whether the computed path is valid when validated against the actual (not predicted) obstacles.

#### Masking Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `center` | Forward-facing rectangle centered on direction | Default pathfinding |
| `left` | Offset to the left of direction | Avoiding right-side obstacles |
| `right` | Offset to the right of direction | Avoiding left-side obstacles |
| `wide` | Doubled width for wider coverage | Complex obstacle fields |

### Thunderstorm Prediction

The thunderstorm prediction module uses Seq2Seq deep learning models to predict future thunderstorm positions based on historical radar imagery.

#### Using the ThunderstormPredictor

```python
from shapely import Point
from thund_avoider.services.masked_dynamic_avoider import MaskedDynamicAvoider
from thund_avoider.schemas.masked_dynamic_avoider import DirectionVector
from thund_avoider.settings import settings

# Initialize predictor with MaskedDynamicAvoider
avoider = MaskedDynamicAvoider(
    masked_preprocessor_config=settings.masked_preprocessor_config,
    dynamic_avoider_config=settings.dynamic_avoider_config,
    predictor_config=settings.predictor_config,  # Enables predictions
)

# Perform pathfinding with ML-based predictions
result = avoider.perform_pathfinding_with_finetuning_masked(
    current_pos=Point(300000, 6800000),
    end=Point(400000, 6900000),
    current_time_index=0,
    window_size=3,
    num_preds=7,
    time_keys=time_keys,
    dict_obstacles=dict_obstacles,
    prediction_mode="predictive",  # Uses ML predictions
    masking_strategy="wide",
)
```

#### Prediction Pipeline

1. **Input**: 6 historical radar images (30 minutes of history at 5-min intervals)
2. **Processing**: Images are cropped using "left" and "right" strategies, resized to 256x256
3. **Inference**: Seq2Seq model predicts 6 future frames (30 minutes ahead)
4. **Output**: Obstacle polygons for each future time step

#### Available Model Types

| Model | Type | Checkpoint |
|-------|------|------------|
| `RNN` | Standard RNN | `checkpoints/RNN.pt` |
| `LSTM` | Standard LSTM | `checkpoints/LSTM.pt` |
| `GRU` | Standard GRU | `checkpoints/GRU.pt` |
| `ConvRNN` | Convolutional RNN | `checkpoints/ConvRNN.pt` |
| `ConvLSTM` | Convolutional LSTM | `checkpoints/ConvLSTM.pt` |
| `ConvGRU` | Convolutional GRU | `checkpoints/ConvGRU.pt` |

#### Direct Predictor Usage

```python
from shapely import Point
from thund_avoider.services.masked_dynamic_avoider import ThunderstormPredictor
from thund_avoider.services.masked_dynamic_avoider.masked_preprocessor import MaskedPreprocessor
from thund_avoider.settings import settings
from thund_avoider.schemas.masked_dynamic_avoider import DirectionVector

preprocessor = MaskedPreprocessor(settings.masked_preprocessor_config)
predictor = ThunderstormPredictor(settings.predictor_config, preprocessor)

# Generate predictions
result = predictor.predict(
    time_keys=time_keys,
    current_time_index=0,
    current_position=Point(300000, 6800000),
    direction_vector=DirectionVector(dx=100000, dy=100000),
    strategy="concave",
)

# Access predicted obstacles
for time_key, obstacles in result.obstacles_dict.items():
    print(f"{time_key}: {len(obstacles['concave'])} obstacles")
```

### Deep Learning Models

#### Loading Pre-trained Models

```python
import torch
from thund_avoider.models.conv_lstm import ConvLSTMSeq2Seq
from thund_avoider.models.conv_gru import ConvGRUSeq2Seq
from thund_avoider.models.conv_rnn import ConvRNNSeq2Seq

# Load ConvLSTM model
model = ConvLSTMSeq2Seq(
    input_channels=1,
    hidden_channels=64,
    output_channels=1,
    kernel_size=3,
    num_layers=2,
    device="cuda"
)
model.load_state_dict(torch.load("thund_avoider/models/checkpoints/ConvLSTM.pt"))
model.eval()
```

#### Making Predictions

```python
# Input: [batch, seq_len, channels, height, width]
input_sequence = torch.randn(1, 12, 1, 256, 256).cuda()

# Predict future frames
with torch.no_grad():
    predicted = model(input_sequence, future_seq_len=6)

# Output: [batch, future_seq_len, channels, height, width]
print(f"Predicted {predicted.shape[1]} future frames")
```

#### Training New Models

Use the Jupyter notebooks in `notebooks/` directory for training:

```bash
jupyter notebook notebooks/
```

Available notebooks:
- `rnn_training.ipynb` - RNN, LSTM, GRU training
- `conv_rnn_training.ipynb` - ConvRNN training
- `conv_lstm_training.ipynb` - ConvLSTM training
- `conv_gru_training.ipynb` - ConvGRU training

---

## Configuration

All configurations are managed via Pydantic models in `thund_avoider/settings.py`. Use the global `settings` instance for default configurations:

```python
from thund_avoider.settings import settings

# Access default configurations
parser_config = settings.parser_config
preprocessor_config = settings.preprocessor_config
masked_preprocessor_config = settings.masked_preprocessor_config
static_avoider_config = settings.static_avoider_config
dynamic_avoider_config = settings.dynamic_avoider_config
predictor_config = settings.predictor_config
```

### ParserConfig

```python
from thund_avoider.settings import ParserConfig

config = ParserConfig(
    first_date=datetime(2020, 12, 31, 23, 55),  # Start date
    last_date=datetime(2024, 7, 31, 23, 55),    # End date
    delta_minutes=5,                             # Time step in minutes
    intensity_threshold=100,                     # Min pixel intensity
    min_pixels=9000,                             # Min pixels for segment
    image_size=256,                              # Output image size
)
```

### PreprocessorConfig

```python
from thund_avoider.settings import PreprocessorConfig

config = PreprocessorConfig(
    intensity_threshold_low=100,    # Min pixel intensity (0-255)
    intensity_threshold_high=255,   # Max pixel intensity (0-255)
    distance_between=25000,         # 25 * 2 km min distance between storms
    distance_avoid=15000,           # 15km buffer around storms
)
```

### MaskedPreprocessorConfig

```python
from thund_avoider.settings import MaskedPreprocessorConfig

config = MaskedPreprocessorConfig(
    # Inherits all PreprocessorConfig options
    square_side_length_m=250000,    # 250km side length
    bbox_buffer_m=10000,            # 10km buffer for bbox edges
)
```

### StaticAvoiderConfig

```python
from thund_avoider.settings import StaticAvoiderConfig

config = StaticAvoiderConfig(
    buffer=5000,           # 5km additional buffer around obstacles
    tolerance=5000,        # 5km geometry simplification tolerance
    bbox_buffer=50000,     # 50km buffer around start/end points
)
```

### DynamicAvoiderConfig

```python
from thund_avoider.settings import DynamicAvoiderConfig

config = DynamicAvoiderConfig(
    # GraphBuilderConfig
    graph_builder_config=GraphBuilderConfig(
        crs=3067,                  # Coordinate reference system
        buffer=5000,               # Obstacle buffer in meters
        tolerance=5000,            # Simplification tolerance
        k_neighbors=10,            # KNN neighbors for graph
        strategy="concave",        # "concave" or "convex"
    ),
    # FineTunerConfig
    fine_tuner_config=FineTunerConfig(
        max_distance=20000,            # Max segment for densification
        velocity_kmh=900,              # Aircraft speed in km/h
        delta_minutes=5,               # Time step in minutes
        tuning_strategy="greedy",      # "greedy" or "smooth"
    ),
)
```

### PredictorConfig

```python
from thund_avoider.settings import PredictorConfig

config = PredictorConfig(
    model_type="ConvGRU",         # Model type: RNN, LSTM, GRU, ConvRNN, ConvLSTM, ConvGRU
    checkpoints_dir=Path("thund_avoider/models/checkpoints"),
    input_channels=1,             # Number of input channels
    hidden_channels=64,           # Hidden layer size
    output_channels=1,            # Number of output channels
    kernel_size=3,                # Convolution kernel size
    num_layers=2,                 # Number of RNN layers
    image_size=256,               # Input image size
    input_frames=6,               # Number of input frames (30 min history)
    output_frames=6,              # Number of output frames (30 min prediction)
    delta_minutes=5,              # Time step between frames
)
```

---

## API Reference

### Parser

| Method | Description |
|--------|-------------|
| `get_data()` | Download and process radar data for configured date range |
| `save_to_numpy(data, year_month)` | Save processed data as .npy files |
| `count_informative_segments(tif_path)` | Filter low-quality segments |

### Preprocessor

| Method | Description |
|--------|-------------|
| `get_gdf_for_current_date(current_date)` | Process raster to polygons |
| `save_geodataframe_to_csv(gdf, path)` | Save polygons to CSV |

### StaticAvoider

| Method | Description |
|--------|-------------|
| `find_shortest_path(gdf, strategy, A, B)` | Find path around obstacles |

### DynamicAvoider

| Method | Description |
|--------|-------------|
| `extract_time_keys(dir_path)` | Get sorted time keys from directory |
| `collect_obstacles(directory_path, time_keys)` | Load obstacles for all times |
| `create_master_graph(time_keys, dict_obstacles)` | Build master graph |
| `sliding_window_pathfinding(...)` | Main dynamic pathfinding |

### MaskedDynamicAvoider

| Method | Description |
|--------|-------------|
| `perform_pathfinding_masked(...)` | Masked sliding window pathfinding |
| `perform_pathfinding_with_finetuning_masked(...)` | Masked pathfinding with fine-tuning |

### ThunderstormPredictor

| Method | Description |
|--------|-------------|
| `predict(time_keys, current_time_index, ...)` | Generate thunderstorm predictions |

---

## Model Architectures

### Standard RNN Models

```
Input Sequence ──▶ Encoder (RNN/LSTM/GRU) ──▶ Hidden State ──▶ Decoder ──▶ Output Sequence
   [T, H, W]         Flattened to [T, H*W]                      [T, H*W]      [T, H, W]
```

### Convolutional RNN Models

```
Input Sequence ──▶ Encoder (ConvRNN/ConvLSTM/ConvGRU) ──▶ Hidden State ──▶ Decoder ──▶ Output Sequence
   [T, C, H, W]           Spatial structure preserved                [T, C, H, W]    [T, C, H, W]
```

### Available Models

| Model | File | Checkpoint |
|-------|------|------------|
| RNN | `models/rnn.py` | `checkpoints/RNN.pt` |
| LSTM | `models/rnn.py` | `checkpoints/LSTM.pt` |
| GRU | `models/rnn.py` | `checkpoints/GRU.pt` |
| ConvRNN | `models/conv_rnn.py` | `checkpoints/ConvRNN.pt` |
| ConvLSTM | `models/conv_lstm.py` | `checkpoints/ConvLSTM.pt` |
| ConvGRU | `models/conv_gru.py` | `checkpoints/ConvGRU.pt` |

---

## Project Structure

```
WxRADNet-PyTorch/
├── thund_avoider/
│   ├── __init__.py
│   ├── settings.py                    # Pydantic configuration models
│   ├── parser/
│   │   └── parser.py                  # FMI data downloader
│   ├── schemas/
│   │   ├── dynamic_avoider.py         # Path result models
│   │   ├── masked_dynamic_avoider.py  # Masked path result models
│   │   ├── predictor.py               # Predictor result models
│   │   └── preprocessor.py            # Preprocessor schemas
│   ├── services/
│   │   ├── utils.py                   # Shared utilities
│   │   ├── static_avoider.py          # Static pathfinding
│   │   ├── preprocessor.py            # Raster to vector processing
│   │   ├── dynamic_avoider/
│   │   │   ├── __init__.py
│   │   │   ├── core.py                # Main orchestrator
│   │   │   ├── graph_builder.py       # Master graph construction
│   │   │   ├── fine_tuner.py          # Path fine-tuning
│   │   │   └── data_loader.py         # Obstacle data loading
│   │   └── masked_dynamic_avoider/
│   │       ├── __init__.py
│   │       ├── masked_dynamic_avoider.py
│   │       ├── masked_preprocessor.py
│   │       └── predictor.py           # Thunderstorm prediction
│   ├── models/
│   │   ├── rnn.py                     # RNN/LSTM/GRU models
│   │   ├── conv_rnn.py                # ConvRNN model
│   │   ├── conv_lstm.py               # ConvLSTM model
│   │   ├── conv_gru.py                # ConvGRU model
│   │   └── checkpoints/               # Pre-trained weights
│   └── scripts/
│       ├── utils.py                   # Script utilities
│       ├── run_parser.py              # Data pipeline script
│       ├── run_preprocessor.py        # Preprocessing script
│       ├── run_static_avoider.py      # Static pathfinding script
│       ├── run_dynamic_avoider.py     # Dynamic pathfinding script
│       └── run_masked_dynamic_avoider.py  # Masked dynamic pathfinding
├── notebooks/                         # Training notebooks
├── config/                            # Configuration files
│   ├── timestamps.pkl
│   └── ab_points.pkl
├── data/                              # Processed data storage
├── images/                            # GeoTIFF images storage
├── results/                           # Pathfinding results
├── pyproject.toml
└── README.md
```

---

## Data Sources

### Finnish Meteorological Institute (FMI)

The system uses open radar data from FMI's public S3 bucket:

- **Source**: [FMI Open Data](https://en.ilmatieteenlaitos.fi/open-data)
- **Format**: GeoTIFF (composite radar reflectivity)
- **Resolution**: 1km x 1km
- **Frequency**: 5 minutes
- **Coverage**: Finland

### Coordinate System

All spatial data uses **ETRS89/TM35FIN (EPSG:3067)**:

```
Projection: Transverse Mercator
Datum: ETRS89
Units: Meters
Central Meridian: 27°E
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function arguments
- Add docstrings to all public methods
- Run tests before submitting

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Finnish Meteorological Institute (FMI) for providing open radar data
- PyTorch team for the deep learning framework
- Shapely and GeoPandas communities for geospatial tools