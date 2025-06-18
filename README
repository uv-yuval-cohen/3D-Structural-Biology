# Data Processing

This module handles preprocessing of experimental protein data for the task of NES motif prediction.
It transforms labeled protein data into fixed-length peptide windows suitable for training and inference.

## Contents

- `data_processing.py`:
    - Fetches protein sequences from UniProt using their UniProt IDs.
    - Maps experimental labels to binary NES classes.
    - Slices protein sequences into sliding windows (peptides) of size 22.
    - Outputs a list of peptide samples with positions and labels.

## Input

A CSV file (`deep_proteomics_data_3.csv`) with the following structure:
| protein_id | original_value |
|------------|----------------|
| Q9NTX7     | Cargo A        |
| Q9BRP8     | NTR            |
| ...        | ...            |

## 🧠 Label Mapping

| Label                | Meaning                                | Mapped Binary |
|---------------------|-----------------------------------------|---------------|
| Cargo A             | Confirmed NES-containing cargo          | 1             |
| Low abundant cargo  | NES-positive, but low expression        | 1             |
| Cargo B             | Unclear/ambiguous cargo classification  | 0             |
| Ambiguous           | Uncertain or inconsistent evidence      | 0             |
| Non binder          | Confirmed lack of Crm1 binding          | -1            |
| NUP                 | Structural nucleoporin                  | -1            |
| NTR                 | Nuclear transport receptor              | -1            |
| CRM1 Cofactor       | Not cargo                               | -1            |
| NPC                 | Not cargo                               | -1            |

##  Pipeline

1. enrich_csv_with_sequences() – downloads the full sequence for each UniProt ID.
2. relabel_data() – assigns a binary NES label according to our dictionary.
3. generate_peptides_from_csv() – slices protein sequences into peptide windows.

## Output

A list of peptides with format:
  json
{
  "protein_id": "Q9NTX7",
  "peptide_sequence": "MASNDYTQQATQSYGAYPTQP",
  "start_pos": 0,
  "end_pos": 22,
  "label": 1
}
