# Nuclear Export Signal (NES) Prediction 

A complete pipeline for identifying Nuclear Export Signals in protein sequences using ESM protein language models and neural networks.

## 🔬 Project Overview

This system combines experimental proteomics data with state-of-the-art protein language models to predict NES-containing proteins. The pipeline consists of two main components:

### 1. **Data Processing** (`data_processing.py`)
- Preprocesses experimental protein data from UniProt
- Maps experimental labels to binary NES classifications
- Generates overlapping peptide windows for model training

### 2. **Prediction Pipeline** (`prediction_pipeline.py`)
- Memory-efficient ESM embedding generation
- Neural network classification of peptide sequences
- Protein-level aggregation and NES position identification

## 🎯 Key Features

- **Experimental Data Integration**: Processes real proteomics data with multiple evidence categories
- **Memory Efficiency**: Handles large datasets through batch processing and disk storage
- **Binary Classification**: Robust positive/negative prediction with confidence filtering
- **Position Mapping**: Identifies specific NES regions within protein sequences
- **Comprehensive Output**: Detailed results at both peptide and protein levels

## 📊 Data Labels & Classification

The system processes experimental data with multiple evidence categories:

| Experimental Label | Evidence Type | Binary Classification |
|-------------------|---------------|----------------------|
| Cargo A | Strong NES evidence | **Positive (1)** |
| Low abundant cargo | Weak NES evidence | **Positive (1)** |
| Cargo B | Ambiguous evidence | Uncertain (0) |
| Ambiguous | Unclear evidence | Uncertain (0) |
| Non binder | No Crm1 interaction | **Negative (-1)** |
| NTR/NUP/NPC | Structural proteins | **Negative (-1)** |

*Note: Uncertain samples (0) are filtered out for confident binary classification*

## Quick Start

### 1. Data Preprocessing
```python
from data_processing import generate_peptides_from_csv

# Process experimental data into peptides
all_peptides = generate_peptides_from_csv("deep_proteomics_data_3.csv", window_size=22)
```

### 2. Run Prediction Pipeline
```python
# Load trained model
from neural_net import SimpleDenseNet
trained_model = SimpleDenseNet(esm_emb_dim=2560, hidden_dim=128, dropout=0.2)
trained_model.load_state_dict(torch.load('trained_network.pth'))

# Run full pipeline
from prediction_pipeline import full_prediction_pipeline_efficient
peptide_predictions, protein_predictions, protein_groups, nes_positions = full_prediction_pipeline_efficient(
    all_peptides,
    trained_model,
    output_dir="results"
)
```

### 3. Basic Execution
```bash
python prediction_pipeline.py
```

## 🏗️ System Architecture

```
Experimental Data (CSV)
         ↓
[Data Processing Module]
    • UniProt sequence fetching
    • Label mapping & filtering  
    • Peptide window generation
         ↓
   Peptide Dataset
         ↓
[Prediction Pipeline]
    • ESM embedding generation
    • Neural network classification
    • Protein-level aggregation
         ↓
Results (Peptide + Protein + NES Positions)
```

## 📁 Input/Output

### Input Files
- `deep_proteomics_data_3.csv` - Experimental protein classifications
- `trained_network.pth` - Pre-trained neural network model

### Output Files (in `results/` directory)
- **Peptide Predictions**: Individual peptide classifications with scores
- **Protein Predictions**: Aggregated protein-level results  
- **NES Positions**: Detailed positional information for predicted NES regions
- **Metadata**: Run parameters and performance metrics

## ⚙️ Performance & Efficiency

- **Memory Management**: Processes large datasets without loading all embeddings into RAM
- **Batch Processing**: Configurable batch sizes for GPU and CPU optimization
- **Speed**: ~1000 peptides/minute for embedding, ~50K peptides/minute for prediction
- **Scalability**: Tested on datasets with 100K+ peptides

## Detailed Documentation

For comprehensive information about each component:

- **[Data Processing README](README_data_processing.md)** - Detailed data preprocessing, label mapping, and peptide generation
- **[Prediction Pipeline README](README_protein_classification_evaluation.md)** - Complete pipeline usage, configuration, and troubleshooting

## 🛠️ Requirements

```bash
# Core ML/Data Science
numpy, pandas, scikit-learn, torch, h5py, tqdm

# Bioinformatics  
# ESM protein language model (Facebook Research)
# Custom modules: data_processing, esm_embeddings, neural_net
```

## 🎯 Algorithm Summary

1. **Peptide Classification**: ESM-2 embeddings (2560-dim) → SimpleDenseNet → Binary prediction
2. **Protein Aggregation**: IF any peptide = positive THEN protein = positive  
3. **NES Localization**: Map positive peptides back to protein coordinates

This conservative aggregation approach prioritizes **high sensitivity** for detecting NES-containing proteins.
