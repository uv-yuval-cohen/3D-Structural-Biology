import numpy as np
import h5py
import json
import pandas as pd
from datetime import datetime
from esm_embeddings import get_esm_embeddings, get_esm_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm  

UPPER_THRESHOLD = 1  # default upper threshold for positive predictions
LOWER_THRESHOLD = -1  # default lower threshold for negative predictions


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
    return optimal_threshold, -optimal_threshold


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
    UPPER_THRESHOLD, LOWER_THRESHOLD = find_best_thresholds(all_scores, true_labels)

    # Add predictions to metadata (NO embeddings stored!)
    for i, (score, p) in enumerate(zip(all_scores, peptide_embeddings)):
        p["prediction_score"] = float(score)
        p["predicted_label"] = 1 if score > UPPER_THRESHOLD else -1 if score < LOWER_THRESHOLD else 0
        # Remove embedding_idx as it's no longer needed
        if "embedding_idx" in p:
            del p["embedding_idx"]

    print("✅ Peptide predictions completed efficiently!")
    return peptide_embeddings


def get_protein_level_predictions(peptide_embeddings_with_predictions):
    """
    Efficient protein-level predictions for large datasets.
    Single pass through peptides per protein.
    """
    print("Aggregating peptide predictions to protein level...")
    protein_groups = group_peptides_by_protein(peptide_embeddings_with_predictions)
    protein_predictions = {}

    for protein_id, peptides in tqdm(protein_groups.items(), desc="Aggregating proteins", unit="protein"):
        # Single pass through peptides - much more efficient!
        num_positive = 0
        num_uncertain = 0
        num_negative = 0
        max_score = float('-inf')
        score_sum = 0.0

        for peptide in peptides:
            # Count labels
            label = peptide["predicted_label"]
            if label == 1:
                num_positive += 1
            elif label == 0:
                num_uncertain += 1
            else:  # label == -1
                num_negative += 1

            # Track max and sum for mean in same pass
            score = peptide["prediction_score"]
            if score > max_score:
                max_score = score
            score_sum += score

        # Determine protein label using counts (no list checks needed)
        if num_positive > 0:
            protein_label = 1
        elif num_uncertain > 0:
            protein_label = 0
        else:
            protein_label = -1

        mean_score = score_sum / len(peptides)

        protein_predictions[protein_id] = {
            "predicted_label": protein_label,
            "max_peptide_score": float(max_score),
            "mean_peptide_score": float(mean_score),
            "num_peptides": len(peptides),
            "num_positive_peptides": num_positive,
            "num_uncertain_peptides": num_uncertain,
            "num_negative_peptides": num_negative,
            "positive_peptide_ratio": num_positive / len(peptides)
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


def save_results(peptide_predictions, protein_predictions, nes_positions, peptide_accuracy, protein_accuracy,
                 embedding_size, embedding_layer, batch_size, output_dir="results"):
    """
    Save pipeline results to disk.

    Args:
        peptide_predictions: List of peptide predictions
        protein_predictions: Dict of protein predictions
        nes_positions: Dict of NES positional info
        peptide_accuracy: Peptide-level accuracy
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
        "peptide_accuracy": peptide_accuracy,
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

    # Step 5: Accuracies
    peptide_accuracy = calculate_peptide_accuracy(peptide_predictions)
    protein_accuracy = calculate_protein_accuracy(protein_predictions, all_peptides)

    # Step 6: Save results
    save_results(peptide_predictions, protein_predictions, nes_positions,
                 peptide_accuracy, protein_accuracy, embedding_size, embedding_layer,
                 esm_batch_size, output_dir)

    end_time = datetime.now()
    print(f"🎉 Efficient pipeline completed in {(end_time - start_time).total_seconds() / 60:.2f} minutes!")

    # Optional: Clean up HDF5 file to save disk space
    import os
    if os.path.exists(hdf5_file):
        print(f"Cleaning up temporary embeddings file: {hdf5_file}")
        os.remove(hdf5_file)

    return peptide_predictions, protein_predictions, protein_groups, nes_positions


def calculate_peptide_accuracy(peptide_embeddings_with_predictions):
    """Calculate peptide-level accuracy metrics."""
    true_labels = [p["label"] for p in peptide_embeddings_with_predictions]
    predicted_labels = [p["predicted_label"] for p in peptide_embeddings_with_predictions]
    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"\n📊 PEPTIDE-LEVEL ACCURACY:")
    print(f"Overall Accuracy: {accuracy:.3f}")
    print(f"Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=['Negative', 'Uncertain', 'Positive']))
    print(f"Confusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))
    return accuracy


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
            # True label: 1 if any peptide is 1, 0 if any peptide is 0, else -1
            peptide_labels = protein_groups[protein_id]
            true_label = 1 if 1 in peptide_labels else 0 if 0 in peptide_labels else -1

            true_protein_labels.append(true_label)
            predicted_protein_labels.append(protein_predictions[protein_id]["predicted_label"])

    accuracy = accuracy_score(true_protein_labels, predicted_protein_labels)

    print(f"\n📊 PROTEIN-LEVEL ACCURACY:")
    print(f"Overall Accuracy: {accuracy:.3f}")
    print(f"Classification Report:")
    print(classification_report(true_protein_labels, predicted_protein_labels,
                                target_names=['Negative', 'Uncertain', 'Positive']))
    return accuracy


# Example usage:
if __name__ == "__main__":
    # Example data structure (replace with your actual data)
    all_peptides = [
        {"protein_id": "P12345", "peptide_sequence": "MASNDYTQQATQSYGAYPTQP", "start_pos": 0, "end_pos": 22,
         "label": 1},
        {"protein_id": "P12345", "peptide_sequence": "ASNDYTQQATQSYGAYPTQPG", "start_pos": 1, "end_pos": 23,
         "label": 1},
        {"protein_id": "P67890", "peptide_sequence": "LKJHGFDSAPOIUYTREWQAS", "start_pos": 0, "end_pos": 22,
         "label": 0},
    ]

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
        esm_batch_size=2000,  # Smaller for 24GB GPU
        pred_batch_size=100000,  # Larger for faster predictions
        output_hdf5="embeddings.h5",
        output_dir="results"
    )

    # Print results summary
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"Processed {len(peptide_predictions)} peptides from {len(protein_predictions)} proteins")

    # Print example results
    print(f"\nExample peptide prediction:")
    p = peptide_predictions[0]
    print(f"Protein ID: {p['protein_id']}, Score: {p['prediction_score']:.3f}, Predicted: {p['predicted_label']}")

    print(f"\nExample protein predictions:")
    for protein_id, pred in list(protein_predictions.items())[:2]:
        print(f"Protein {protein_id}: Label={pred['predicted_label']} "
              f"({pred['num_positive_peptides']} pos, {pred['num_uncertain_peptides']} unc, {pred['num_negative_peptides']} neg)")

    # Print NES positions for positive proteins
    if nes_positions:
        print(f"\nNES Positions in positive proteins:")
        for protein_id, pos_info in list(nes_positions.items())[:2]:
            print(f"Protein {protein_id}: {pos_info['num_nes_regions']} NES regions")
            for region in pos_info['nes_regions']:
                print(f"  - Position {region['start_pos']}-{region['end_pos']} (score: {region['score']:.3f})")

    print(f"\n🎉 All results saved to 'results/' directory!")
