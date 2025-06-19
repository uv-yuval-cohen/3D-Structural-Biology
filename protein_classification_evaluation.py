import numpy as np
import h5py
import json
import pandas as pd
from datetime import datetime

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from data_processing import generate_peptides_from_csv
from esm_embeddings import get_esm_embeddings, get_esm_model


def get_peptide_embeddings_with_protein_info(all_peptides, embedding_size=2560, embedding_layer=9, batch_size=1000,
                                             output_hdf5="embeddings.h5"):
    """
    Get ESM embeddings for peptides in batches, saving to disk to manage memory.

    Args:
        all_peptides: List of dicts with keys: protein_id, peptide_sequence, start_pos, end_pos, label
        embedding_size: ESM model embedding dimension
        embedding_layer: Transformer layer to extract embeddings from
        batch_size: Number of peptides to process per batch (smaller = less GPU memory)
        output_hdf5: Path to HDF5 file for storing embeddings

    Returns:
        Tuple: (List of dicts with metadata only, HDF5 file path)
    """
    print("Loading ESM-2 model...")
    model_esm, alphabet_esm, batch_converter_esm, device_esm = get_esm_model(embedding_size=embedding_size)

    results = []
    num_peptides = len(all_peptides)

    with h5py.File(output_hdf5, 'w') as h5f:
        embedding_dset = h5f.create_dataset('embeddings', shape=(num_peptides, embedding_size), dtype=np.float32)

        for start_idx in tqdm(range(0, num_peptides, batch_size), desc="ESM Embedding Batches", unit="batch"):
            end_idx = min(start_idx + batch_size, num_peptides)
            batch_peptides = all_peptides[start_idx:end_idx]

            peptide_tuples = [(f"pep_{i}", p['peptide_sequence']) for i, p in enumerate(batch_peptides, start_idx)]
            batch_embeddings = get_esm_embeddings(
                peptide_tuples, model_esm, alphabet_esm, batch_converter_esm, device_esm,
                embedding_layer=embedding_layer, sequence_embedding=True
            )
            embedding_dset[start_idx:end_idx] = batch_embeddings

            # Store only metadata - no embeddings in memory!
            for i, p in enumerate(batch_peptides):
                results.append({
                    "embedding_idx": start_idx + i,
                    "protein_id": p["protein_id"],
                    "peptide_sequence": p["peptide_sequence"],
                    "start_pos": p["start_pos"],
                    "end_pos": p["end_pos"],
                    "label": p["label"]
                })

    print(f"Successfully processed {len(results)} peptides! Embeddings saved to {output_hdf5}")
    return results, output_hdf5


def group_peptides_by_protein(peptide_embeddings):
    """Group peptide embeddings by protein_id for protein-level analysis."""
    protein_groups = {}
    for peptide_info in peptide_embeddings:
        protein_id = peptide_info["protein_id"]
        if protein_id not in protein_groups:
            protein_groups[protein_id] = []
        protein_groups[protein_id].append(peptide_info)
    return protein_groups


def find_best_thresholds(prediction_scores, true_labels):
    """Find optimal thresholds using ROC curve analysis."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(true_labels, prediction_scores)
    youdens_index = tpr - fpr
    optimal_idx = np.argmax(youdens_index)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.3f} (TPR: {tpr[optimal_idx]:.3f}, FPR: {fpr[optimal_idx]:.3f})")
    return optimal_threshold


def get_peptide_predictions_efficient(peptide_embeddings, trained_model, hdf5_file, pred_batch_size=50000):
    """
    Memory-efficient peptide predictions using batched processing.
    Never loads all embeddings into memory at once.

    Args:
        peptide_embeddings: Metadata list (no actual embeddings)
        trained_model: Trained model
        hdf5_file: Path to HDF5 embeddings file
        pred_batch_size: Batch size for predictions (adjust based on GPU memory)
    """
    from neural_net import get_net_scores
    print(f"Getting predictions for {len(peptide_embeddings)} peptides in batches...")

    all_scores = []
    num_peptides = len(peptide_embeddings)

    # Process predictions in batches to avoid loading all embeddings
    with h5py.File(hdf5_file, 'r') as h5f:
        for start_idx in tqdm(range(0, num_peptides, pred_batch_size), desc="Neural Net Predictions", unit="batch"):
            end_idx = min(start_idx + pred_batch_size, num_peptides)
            batch_size = end_idx - start_idx

            # Load only this batch of embeddings
            batch_embeddings = h5f['embeddings'][start_idx:end_idx]

            # Get predictions for this batch
            batch_scores = get_net_scores(trained_model, batch_embeddings)
            all_scores.extend(batch_scores)

            # Clear batch from memory
            del batch_embeddings

    # Find optimal thresholds using all scores
    true_labels = [p["label"] for p in peptide_embeddings]
    THRESHOLD = 0.8

    # Add predictions to metadata (NO embeddings stored!)
    for i, (score, p) in enumerate(zip(all_scores, peptide_embeddings)):
        # Convert score to scalar if it's an array
        score_val = float(score) if np.isscalar(score) else float(score[0])

        p["prediction_score"] = score_val
        p["predicted_label"] = 1 if score_val >= THRESHOLD else 0  # BINARY: 1 or 0
        # Remove embedding_idx as it's no longer needed
        if "embedding_idx" in p:
            del p["embedding_idx"]

    print("✅ Peptide predictions completed efficiently!")
    return peptide_embeddings


def get_protein_level_predictions(peptide_embeddings_with_predictions):
    """
    Efficient protein-level predictions for large datasets.
    Rule: If ANY peptide from protein gets prediction=1, then protein=1. Otherwise protein=0.
    """
    print("Aggregating peptide predictions to protein level...")
    protein_groups = group_peptides_by_protein(peptide_embeddings_with_predictions)
    protein_predictions = {}

    for protein_id, peptides in tqdm(protein_groups.items(), desc="Aggregating proteins", unit="protein"):
        num_positive_peptides = 0  # peptides with prediction = 1
        num_negative_peptides = 0  # peptides with prediction = 0
        max_score = float('-inf')
        score_sum = 0.0

        for peptide in peptides:
            # Count predictions (should only be 0 or 1 in binary case)
            pred_label = peptide["predicted_label"]
            if pred_label == 1:
                num_positive_peptides += 1
            elif pred_label == 0:
                num_negative_peptides += 1
            else:
                print(f"WARNING: Unexpected prediction label {pred_label} for peptide in protein {protein_id}")

            # Track max and sum for mean
            score = peptide["prediction_score"]
            if score > max_score:
                max_score = score
            score_sum += score

        # Protein prediction rule: If ANY peptide is positive, protein is positive
        if num_positive_peptides > 0:
            protein_pred_label = 1
        else:
            protein_pred_label = 0

        mean_score = score_sum / len(peptides)

        protein_predictions[protein_id] = {
            "predicted_label": protein_pred_label,
            "max_peptide_score": float(max_score),
            "mean_peptide_score": float(mean_score),
            "num_peptides": len(peptides),
            "num_positive_peptides": num_positive_peptides,
            "num_negative_peptides": num_negative_peptides,
            "positive_peptide_ratio": num_positive_peptides / len(peptides)
        }

    print(f"✅ Protein-level predictions completed for {len(protein_predictions)} proteins!")
    return protein_predictions, protein_groups


def get_nes_positional_info_efficient(protein_predictions, protein_groups):
    """
    Efficient NES position extraction - no embedding access needed.
    """
    print("Extracting NES positional information...")
    nes_positions = {}

    for protein_id, pred_info in protein_predictions.items():
        if pred_info["predicted_label"] == 1:  # Only positive proteins
            peptides = protein_groups[protein_id]
            positive_peptides = [p for p in peptides if p["predicted_label"] == 1]

            nes_regions = []
            for peptide in positive_peptides:
                nes_regions.append({
                    "start_pos": peptide["start_pos"],
                    "end_pos": peptide["end_pos"],
                    "sequence": peptide["peptide_sequence"],
                    "score": peptide["prediction_score"]
                })

            nes_regions.sort(key=lambda x: x["start_pos"])
            nes_positions[protein_id] = {
                "num_nes_regions": len(nes_regions),
                "nes_regions": nes_regions,
                "earliest_start": min(r["start_pos"] for r in nes_regions),
                "latest_end": max(r["end_pos"] for r in nes_regions),
                "highest_score": max(r["score"] for r in nes_regions)
            }

    print(f"✅ Found NES positions for {len(nes_positions)} positive proteins!")
    return nes_positions


def save_results(peptide_predictions, protein_predictions, nes_positions, protein_accuracy,
                 embedding_size, embedding_layer, batch_size, output_dir="results"):
    """
    Save pipeline results to disk.

    Args:
        peptide_predictions: List of peptide predictions
        protein_predictions: Dict of protein predictions
        nes_positions: Dict of NES positional info
        protein_accuracy: Protein-level accuracy
        embedding_size: ESM model embedding size
        embedding_layer: ESM layer used
        batch_size: Batch size used
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save peptide predictions as CSV
    peptide_df = pd.DataFrame([
        {
            "protein_id": p["protein_id"],
            "peptide_sequence": p["peptide_sequence"],
            "start_pos": p["start_pos"],
            "end_pos": p["end_pos"],
            "label": p["label"],
            "prediction_score": p["prediction_score"],
            "predicted_label": p["predicted_label"]
        } for p in peptide_predictions
    ])
    peptide_csv = f"{output_dir}/peptide_predictions_{timestamp}.csv"
    peptide_df.to_csv(peptide_csv, index=False)
    print(f"Saved peptide predictions to {peptide_csv}")

    # Save protein predictions as JSON
    protein_json = f"{output_dir}/protein_predictions_{timestamp}.json"
    with open(protein_json, 'w') as f:
        json.dump(protein_predictions, f, indent=2)
    print(f"Saved protein predictions to {protein_json}")

    # Save NES positions as JSON
    nes_json = f"{output_dir}/nes_positions_{timestamp}.json"
    with open(nes_json, 'w') as f:
        json.dump(nes_positions, f, indent=2)
    print(f"Saved NES positions to {nes_json}")

    # Save metadata and accuracies
    metadata = {
        "timestamp": timestamp,
        "num_peptides": len(peptide_predictions),
        "num_proteins": len(protein_predictions),
        "embedding_size": embedding_size,
        "embedding_layer": embedding_layer,
        "batch_size": batch_size,
        "protein_accuracy": protein_accuracy
    }
    metadata_json = f"{output_dir}/metadata_{timestamp}.json"
    with open(metadata_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_json}")


def full_prediction_pipeline_efficient(all_peptides, trained_model, embedding_size=2560,
                                       embedding_layer=9, esm_batch_size=1000, pred_batch_size=50000,
                                       output_hdf5="embeddings.h5", output_dir="results"):
    """
    Truly memory-efficient pipeline.

    Args:
        all_peptides: List of dicts with protein info and sequences
        trained_model: Trained SimpleDenseNet model
        embedding_size: ESM model size
        embedding_layer: ESM layer to extract
        esm_batch_size: Batch size for ESM embedding generation (smaller = less GPU memory)
        pred_batch_size: Batch size for neural net predictions (larger = faster, but more RAM)
        output_hdf5: Path to HDF5 file for embeddings
        output_dir: Directory to save results

    Returns:
        Tuple: (peptide_predictions, protein_predictions, protein_groups, nes_positions)
    """
    print("🚀 Starting memory-efficient prediction pipeline...")
    start_time = datetime.now()

    # Step 1: Generate embeddings (saved to disk)
    peptide_embeddings, hdf5_file = get_peptide_embeddings_with_protein_info(
        all_peptides, embedding_size, embedding_layer, esm_batch_size, output_hdf5
    )

    # Step 2: Get predictions efficiently (embeddings stay on disk)
    peptide_predictions = get_peptide_predictions_efficient(
        peptide_embeddings, trained_model, hdf5_file, pred_batch_size
    )

    # Step 3: Protein-level aggregation (no embeddings needed)
    protein_predictions, protein_groups = get_protein_level_predictions(peptide_predictions)

    # Step 4: NES positions (no embeddings needed)
    nes_positions = get_nes_positional_info_efficient(protein_predictions, protein_groups)

    # Step 5: Accuracy
    protein_accuracy = calculate_protein_accuracy(protein_predictions, all_peptides)

    # Step 6: Save results
    save_results(peptide_predictions, protein_predictions, nes_positions,
                 protein_accuracy, embedding_size, embedding_layer,
                 esm_batch_size, output_dir)

    end_time = datetime.now()
    print(f"🎉 Efficient pipeline completed in {(end_time - start_time).total_seconds() / 60:.2f} minutes!")

    # Optional: Clean up HDF5 file to save disk space
    import os
    if os.path.exists(hdf5_file):
        print(f"Cleaning up temporary embeddings file: {hdf5_file}")
        os.remove(hdf5_file)

    return peptide_predictions, protein_predictions, protein_groups, nes_positions


def calculate_protein_accuracy(protein_predictions, all_peptides):
    """Calculate protein-level accuracy metrics."""
    protein_groups = {}
    for peptide in all_peptides:
        protein_id = peptide["protein_id"]
        if protein_id not in protein_groups:
            protein_groups[protein_id] = []
        protein_groups[protein_id].append(peptide["label"])

    true_protein_labels = []
    predicted_protein_labels = []

    for protein_id in protein_groups.keys():
        if protein_id in protein_predictions:
            # Since all peptides from same protein have same label, just take the first one
            peptide_labels = protein_groups[protein_id]

            # Verify all peptides have same label (debugging check)
            if len(set(peptide_labels)) > 1:
                print(f"WARNING: Protein {protein_id} has inconsistent peptide labels: {set(peptide_labels)}")

            true_label = peptide_labels[0]  # All should be same, so take first

            true_protein_labels.append(true_label)
            predicted_protein_labels.append(protein_predictions[protein_id]["predicted_label"])

    accuracy = accuracy_score(true_protein_labels, predicted_protein_labels)

    print(f"\n📊 PROTEIN-LEVEL ACCURACY:")
    print(f"Overall Accuracy: {accuracy:.3f}")
    print(f"Classification Report:")
    print(classification_report(true_protein_labels, predicted_protein_labels,
                                target_names=['Non-NES', 'NES-Positive']))
    return accuracy


if __name__ == "__main__":
    # Load real data from CSV
    print("Loading peptides from data_classified.csv...")
    all_peptides_full = generate_peptides_from_csv("data_classified.csv", window_size=22)
    print(f"Generated {len(all_peptides_full)} total peptides")

    # FILTER TO BINARY: Keep only confident labels, exclude uncertain (0)
    def filter_to_confident_binary(peptides):
        """Keep only confident samples: 1→1 (NES), -1→0 (non-NES). Remove uncertain (0)."""
        filtered_peptides = []
        for p in peptides:
            if p['label'] == 1:
                p['label'] = 1  # Keep as positive
                filtered_peptides.append(p)
            elif p['label'] == -1:
                p['label'] = 0  # Convert to negative
                filtered_peptides.append(p)
            # Skip label == 0 (uncertain samples)
        return filtered_peptides

    original_count = len(all_peptides_full)
    all_peptides_full = filter_to_confident_binary(all_peptides_full)
    filtered_count = len(all_peptides_full)

    print(f"Filtered to confident samples only: {original_count:,} → {filtered_count:,} peptides")
    print("Binary classification: 1=NES positive, 0=non-NES (confident negatives only)")

    # Show FULL dataset distribution
    full_labels = [p['label'] for p in all_peptides_full]
    full_label_counts = dict(zip(*np.unique(full_labels, return_counts=True)))
    print(f"FULL dataset label distribution: {full_label_counts}")

    # Calculate percentages
    total_peptides = len(all_peptides_full)
    for label, count in full_label_counts.items():
        percentage = (count / total_peptides) * 100
        print(f"  Label {label}: {count:,} peptides ({percentage:.1f}%)")

    all_peptides = all_peptides_full
    print(f"\nUsing {len(all_peptides)} peptides for processing")



    # Load your trained model
    import torch
    from neural_net import SimpleDenseNet

    print("Loading trained model...")
    trained_model = SimpleDenseNet(esm_emb_dim=2560, hidden_dim=128, dropout=0.2)
    trained_model.load_state_dict(torch.load('trained_network.pth'))
    trained_model.eval()  # Set to evaluation mode
    print("✅ Model loaded successfully!")

    # Run the efficient pipeline
    peptide_predictions, protein_predictions, protein_groups, nes_positions = full_prediction_pipeline_efficient(
        all_peptides,
        trained_model,
        embedding_size=2560,
        embedding_layer=9,
        esm_batch_size=1000,  # Smaller for 24GB GPU
        pred_batch_size=100000,  # Larger for faster predictions
        output_hdf5="embeddings.h5",
        output_dir="results"
    )

    # Print results summary
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"Processed {len(peptide_predictions)} peptides from {len(protein_predictions)} proteins")

    # Print example results and then all the other results
    print(f"\nExample peptide prediction:")
    p = peptide_predictions[0]
    print(f"Protein ID: {p['protein_id']}, Score: {p['prediction_score']:.3f}, Predicted: {p['predicted_label']}, label: {p['label']}")


    print(f"\nExample protein predictions:")
    for protein_id, pred in list(protein_predictions.items())[:2]:
        print(f"Protein {protein_id}: Label={pred['predicted_label']} "
              f"({pred['num_positive_peptides']} pos, {pred['num_negative_peptides']} neg)")

    # Print NES positions for positive proteins
    if nes_positions:
        print(f"\nNES Positions in positive proteins:")
        for protein_id, pos_info in list(nes_positions.items())[:2]:
            print(f"Protein {protein_id}: {pos_info['num_nes_regions']} NES regions")
            for region in pos_info['nes_regions']:
                print(f"  - Position {region['start_pos']}-{region['end_pos']} (score: {region['score']:.3f})")

    print(f"\n🎉 All results saved to 'results/' directory!")