# NES Prediction Pipeline

A memory-efficient machine learning pipeline for predicting Nuclear Export Signals (NES) in proteins using ESM embeddings and neural networks.

## Overview

This pipeline processes protein sequences to identify Nuclear Export Signals (NES) - short amino acid sequences that direct proteins for export from the cell nucleus. The system works by:

1. **Peptide Generation**: Splits protein sequences into overlapping 22-amino acid peptides using a sliding window
2. **Binary Classification**: Uses a trained neural network to classify each peptide as NES-positive (1) or non-NES (0)
3. **Protein-Level Prediction**: Labels entire proteins as positive if ANY of their peptides are predicted positive


## Algorithm Details

### Peptide Classification
- Uses ESM-2 protein language model embeddings (2560-dimensional)
- Binary classifier: SimpleDenseNet with 128 hidden units and 0.2 dropout
- Fixed threshold of 0.6 for binary decisions

### Protein Aggregation Rule
**IF** any peptide from protein has prediction = 1 **THEN** protein = 1 **ELSE** protein = 0

This conservative approach ensures high sensitivity for detecting NES-containing proteins.


## Requirements

```bash
# Core dependencies
numpy
pandas
h5py
scikit-learn
torch
tqdm

# Bioinformatics
# ESM model (Facebook's protein language model)
# Custom modules: data_processing, esm_embeddings, neural_net
```

## Input Data Format

The pipeline expects a CSV file (`data_classified.csv`) with protein data containing:
- Protein sequences
- Classification labels (1: NES-positive, -1: confident negative, 0: uncertain)

## Usage

### Basic Run
```python
python prediction_pipeline.py
```

### Custom Configuration
```python
# Load your peptides
all_peptides = generate_peptides_from_csv("your_data.csv", window_size=22)

# Load trained model
trained_model = SimpleDenseNet(esm_emb_dim=2560, hidden_dim=128, dropout=0.2)
trained_model.load_state_dict(torch.load('trained_network.pth'))

# Run pipeline
peptide_predictions, protein_predictions, protein_groups, nes_positions = full_prediction_pipeline_efficient(
    all_peptides,
    trained_model,
    embedding_size=2560,
    embedding_layer=9,
    esm_batch_size=1000,    # Adjust for GPU memory
    pred_batch_size=100000, # Adjust for RAM
    output_hdf5="embeddings.h5",
    output_dir="results"
)
```

## Configuration Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|--------|
| `embedding_size` | ESM model dimension | 2560 | ESM-2 model size |
| `embedding_layer` | Transformer layer to extract | 9 | Middle layer often works best |
| `esm_batch_size` | ESM embedding batch size | 1000 | Reduce if GPU memory limited |
| `pred_batch_size` | Prediction batch size | 100000 | Increase for faster processing |

## Output Files

The pipeline saves results to the `results/` directory with timestamps:

### Peptide Predictions (`peptide_predictions_YYYYMMDD_HHMMSS.csv`)
- Individual peptide sequences with prediction scores and binary labels
- Columns: `protein_id`, `peptide_sequence`, `start_pos`, `end_pos`, `label`, `prediction_score`, `predicted_label`

### Protein Predictions (`protein_predictions_YYYYMMDD_HHMMSS.json`)
- Protein-level aggregated predictions
- Contains: predicted label, max/mean peptide scores, peptide counts, positive ratios

### NES Positions (`nes_positions_YYYYMMDD_HHMMSS.json`)
- Detailed NES region information for positive proteins
- Includes: start/end positions, sequences, scores for each predicted NES region

### Metadata (`metadata_YYYYMMDD_HHMMSS.json`)
- Run parameters, dataset statistics, and accuracy metrics

## Performance

**Memory Efficiency**:
- Processing embeddings in batches
- Storing embeddings on disk, not in RAM
- Cleaning up temporary files automatically


## Troubleshooting

**GPU Memory Issues**: Reduce `esm_batch_size` (try 500 or 250)
**RAM Issues**: Reduce `pred_batch_size` (try 25000 or 10000)
**Disk Space**: Temporary HDF5 files are automatically cleaned up, but ensure sufficient space during processing