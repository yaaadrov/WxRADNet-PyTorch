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
from thund_avoider.parser.parser import Parser
from thund_avoider.settings import PreprocessorConfig

config = PreprocessorConfig(
    base_url="https://fmi-opendata-{}-radar-tutorial.s3-flash-{}",  # FMI S3 pattern
    intensity_threshold_low=100,
    intensity_threshold_high=255,
    distance_between=25000,   # 25 * 2 km minimum between thunderstorms
    distance_avoid=15000,     # 15km buffer around storms
)

parser = Parser(config)

# Download and process data for a date range
parser.get_data()
```

#### Using the Preprocessor

```python
from datetime import datetime
from pathlib import Path
from thund_avoider.services.preprocessor import Preprocessor
from thund_avoider.settings import PreprocessorConfig

config = PreprocessorConfig(
    intensity_threshold_low=100,
    intensity_threshold_high=255,
    distance_between=50000,
    distance_avoid=15000,
)

preprocessor = Preprocessor(config)

# Process a specific timestamp
gdf = preprocessor.get_gpd_for_current_date(
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
from thund_avoider.settings import StaticAvoiderConfig

config = StaticAvoiderConfig(
    buffer=5000,           # 5km additional buffer
    tolerance=5000,        # 5km simplification tolerance
    bbox_buffer=50000,     # 50km buffer for A/B points
    strategy="concave",    # or "convex"
)

avoider = StaticAvoider(config)

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
from thund_avoider.settings import DynamicAvoiderConfig

config = DynamicAvoiderConfig(
    crs="EPSG:3067",           # ETRS89/TM35FIN
    velocity_kmh=900,          # Aircraft speed
    delta_minutes=5,           # Forecast frequency
    buffer=5000,               # Obstacle buffer
    tolerance=5000,            # Simplification tolerance
    k_neighbors=10,            # KNN neighbors for graph
    max_distance=5000,         # Max segment distance for densification
    simplification_tolerance=1000,
    smooth_tolerance=2000,
    max_iter=100,
    delta_length=100,
    strategy="concave",        # or "convex"
    tuning_strategy="greedy",  # or "smooth"
)

avoider = DynamicAvoider(config)

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
from thund_avoider.services.masked_dynamic_avoider.masked_dynamic_avoider import MaskedDynamicAvoider
from thund_avoider.settings import MaskedPreprocessorConfig, DynamicAvoiderConfig

preprocessor_config = MaskedPreprocessorConfig(
    square_side_length_m=250000,  # 250km radar range
    bbox_buffer_m=10000,          # 10km buffer
)

dynamic_config = DynamicAvoiderConfig(
    velocity_kmh=900,
    delta_minutes=5,
    strategy="concave",
    tuning_strategy="greedy",
)

avoider = MaskedDynamicAvoider(
    masked_preprocessor_config=preprocessor_config,
    dynamic_avoider_config=dynamic_config,
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
)
```

#### Masking Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `center` | Forward-facing rectangle centered on direction | Default pathfinding |
| `left` | Offset to the left of direction | Avoiding right-side obstacles |
| `right` | Offset to the right of direction | Avoiding left-side obstacles |
| `wide` | Doubled width for wider coverage | Complex obstacle fields |

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

All configurations are managed via Pydantic models in `thund_avoider/settings.py`.

### PreprocessorConfig

```python
from thund_avoider.settings import PreprocessorConfig

config = PreprocessorConfig(
    base_url="https://fmi-opendata-{}-radar-tutorial.s3-flash-{}",
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
    strategy="concave",    # "concave" or "convex"
)
```

### DynamicAvoiderConfig

```python
from thund_avoider.settings import DynamicAvoiderConfig

config = DynamicAvoiderConfig(
    crs="EPSG:3067",              # Coordinate reference system
    velocity_kmh=900,             # Aircraft speed in km/h
    delta_minutes=5,              # Time step in minutes
    buffer=5000,                  # Obstacle buffer in meters
    tolerance=5000,               # Simplification tolerance
    k_neighbors=10,               # KNN neighbors for graph
    max_distance=5000,            # Max segment for densification
    simplification_tolerance=1000,
    smooth_tolerance=2000,
    max_iter=100,
    delta_length=100,
    strategy="concave",           # "concave" or "convex"
    tuning_strategy="greedy",     # "greedy" or "smooth"
)
```

---

## API Reference

### Parser

| Method | Description |
|--------|-------------|
| `get_data(year_month_start, year_month_end)` | Download and process radar data |
| `save_to_numpy(data, year_month)` | Save processed data as .npy files |
| `count_informative_segments(tif_path)` | Filter low-quality segments |

### Preprocessor

| Method | Description |
|--------|-------------|
| `get_gpd_for_current_date(current_date, tif_path)` | Process raster to polygons |
| `save_geodataframe_to_csv(gdf, path)` | Save polygons to CSV |
| `_convert_raster_to_polygons(tif_path)` | Raster to vector conversion |
| `_union_polygons(polygons, strategy)` | Generate hulls |

### StaticAvoider

| Method | Description |
|--------|-------------|
| `find_shortest_path(gdf, strategy, A, B)` | Find path around obstacles |
| `_build_visibility_graph(vertices, obstacles)` | Create visibility graph |
| `_is_line_valid(line, obstacles)` | Check line-obstacle intersection |

### DynamicAvoider

| Method | Description |
|--------|-------------|
| `extract_time_keys(dir_path)` | Get sorted time keys from directory |
| `collect_obstacles(directory_path, time_keys)` | Load obstacles for all times |
| `create_master_graph(time_keys, dict_obstacles)` | Build master graph |
| `sliding_window_pathfinding(...)` | Main dynamic pathfinding |
| `_greedy_fine_tuning(path, time_keys, strtrees)` | Greedy optimization |
| `_smooth_fine_tuning(path, time_keys, strtrees)` | Smooth optimization |

### MaskedDynamicAvoider

| Method | Description |
|--------|-------------|
| `perform_pathfinding_masked(...)` | Masked sliding window pathfinding |
| `perform_pathfinding_with_finetuning_masked(...)` | Masked pathfinding with fine-tuning |
| `_create_direction_vector(path)` | Create direction vector from path |
| `_prepare_obstacles(...)` | Prepare obstacles with spatial masking |

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
│   │   └── preprocessor.py            # Preprocessor schemas
│   ├── services/
│   │   ├── dynamic_avoider.py         # Dynamic pathfinding
│   │   ├── static_avoider.py          # Static pathfinding
│   │   ├── preprocessor.py            # Raster to vector processing
│   │   └── masked_dynamic_avoider/
│   │       ├── masked_dynamic_avoider.py
│   │       └── masked_preprocessor.py
│   ├── models/
│   │   ├── rnn.py                     # RNN/LSTM/GRU models
│   │   ├── conv_rnn.py                # ConvRNN model
│   │   ├── conv_lstm.py               # ConvLSTM model
│   │   ├── conv_gru.py                # ConvGRU model
│   │   └── checkpoints/               # Pre-trained weights
│   └── scripts/
│       ├── run_parser.py              # Data pipeline script
│       ├── run_preprocessor.py        # Preprocessing script
│       ├── run_static_avoider.py      # Static pathfinding script
│       └── run_dynamic_avoider.py     # Dynamic pathfinding script
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