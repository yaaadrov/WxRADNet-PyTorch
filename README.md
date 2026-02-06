# WxRADNet-PyTorch

This repository provides a comprehensive framework for aircraft pathfinding to avoid thunderstorms, utilizing deep learning for weather prediction and advanced routing algorithms. The system fetches and processes weather radar data, predicts its future evolution using sequence-to-sequence models, and calculates optimal flight paths to navigate around hazardous weather formations.

## Overview

The "ThundAvoider" system is designed as a modular pipeline that handles everything from data acquisition to dynamic route planning. It leverages open data from the Finnish Meteorological Institute (FMI) to train and run its models.

### Key Features

*   **Data Pipeline:** Automated fetching, processing, and augmentation of FMI weather radar GeoTIFF data.
*   **Deep Learning for Nowcasting:** Implementation of multiple Encoder-Decoder models for precipitation nowcasting:
    *   RNN, LSTM, GRU
    *   ConvRNN, ConvLSTM, ConvGRU
*   **Pre-trained Models:** Includes pre-trained checkpoints for all implemented architectures.
*   **Pathfinding Algorithms:**
    *   **Static Avoider:** Calculates shortest paths around static weather obstacles using visibility graphs.
    *   **Dynamic Avoider:** An advanced algorithm for pathfinding in a time-varying environment using a sliding window approach, master graph construction, and path fine-tuning ("greedy" and "smooth" strategies).
*   **Configurable:** Key parameters for data processing, model training, and pathfinding are easily configurable in `thund_avoider/settings.py`.

## System Architecture

The project consists of a multi-stage pipeline:

1.  **Data Parsing (`run_parser.py`)**
    *   The `thund_avoider.parser.Parser` class downloads raw GeoTIFF radar images from FMI's public S3 bucket.
    *   It segments images, filters for informative regions, applies data augmentation (rotation), and saves the processed data as `.npy` files.

2.  **Obstacle Preprocessing (`run_preprocessor.py`)**
    *   The `thund_avoider.services.Preprocessor` converts the raster radar images into vector-based polygons representing thunderstorm obstacles.
    *   It applies an intensity threshold, identifies contiguous regions, and generates both convex and concave hulls to define the hazardous zones. These are saved as `.csv` files for the pathfinding modules.

3.  **Model Training (Jupyter Notebooks)**
    *   The `notebooks/` directory contains notebooks for training various sequence-to-sequence models.
    *   These models learn to predict a future sequence of radar images based on a past sequence, enabling proactive pathfinding.
    *   Training and test losses are saved, and model checkpoints are stored in `thund_avoider/models/checkpoints/`.

4.  **Pathfinding Services**
    *   **Static Avoider (`run_static_avoider.py`):** For a given timestamp, this service treats the weather obstacles as stationary and finds the shortest path between two points.
    *   **Dynamic Avoider (`run_dynamic_avoider.py`):** This service accounts for the evolution of thunderstorms over time. It constructs a "master graph" of all possible routes across a time series of obstacle data. Using a sliding window, it finds a valid path segment-by-segment, creating a route through the dynamic environment.

## Models

The project explores several deep learning architectures for spatio-temporal forecasting of radar data. All models use an encoder-decoder structure to predict a sequence of future frames from a sequence of past frames.

*   **RNN-based Seq2Seq:** Standard RNN, LSTM, and GRU models that operate on flattened radar images.
*   **Convolutional RNN-based Seq2Seq:** ConvRNN, ConvLSTM, and ConvGRU models that use convolutional layers within the recurrent cells to preserve spatial structure and capture spatio-temporal dependencies more effectively.

## Usage

### Prerequisites

Ensure you have Python and the necessary dependencies installed. You can typically install them using uv.

### Running the Pipeline

You can execute the entire pipeline using the scripts provided in the `thund_avoider/scripts/` directory.

1.  **Download and Parse Data:**
    This script fetches data from FMI and saves it locally.
    ```bash
    python thund_avoider/scripts/run_parser.py
    ```

2.  **Preprocess Data into Obstacles:**
    This script converts the downloaded raster data into vector polygons for pathfinding.
    ```bash
    python thund_avoider/scripts/run_preprocessor.py
    ```

3.  **Run Static Pathfinding Simulation:**
    This script runs the static avoidance algorithm on the preprocessed obstacle data.
    ```bash
    python thund_avoider/scripts/run_static_avoider.py
    ```

4.  **Run Dynamic Pathfinding Simulation:**
    This script runs the dynamic avoidance algorithm, which includes path fine-tuning.
    ```bash
    python thund_avoider/scripts/run_dynamic_avoider.py
    ```

### Training New Models

To train the models from scratch or experiment with different hyperparameters, use the Jupyter notebooks located in the `/notebooks` directory. Each notebook is self-contained and manages its own data loading, training loop, and result saving.