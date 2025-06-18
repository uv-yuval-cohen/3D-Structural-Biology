import numpy as np
import h5py
import json
import pandas as pd
from datetime import datetime
from esm_embeddings import get_esm_embeddings, get_esm_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

UPPER_THRESHOLD = 1  # default upper threshold for positive predictions
LOWER_THRESHOLD = -1  # default lower threshold for negative predictions

def get_peptide_embeddings_with_protein_info(all_peptides, embedding_size=2560, embedding_layer=9, batch_size=10000,
                                             output_hdf5="embeddings.h5"):
    """
    Get ESM embeddings for peptides in batches, saving to disk to manage memory.

    Args:
        all_peptides: List of dicts with keys: protein_id, peptide_sequence, start_pos, end_pos, label
        embedding_size: ESM model embedding dimension
        embedding_layer: Transformer layer to extract embeddings from
        batch_size: Number of peptides to process per batch
        output_hdf5: Path to HDF5 file for storing embeddings

    Returns:
        Tuple: (List of dicts with metadata and HDF5 indices, HDF5 file path)
    """
    print("Loading ESM-2 model...")
    model_esm, alphabet_esm, batch_converter_esm, device_esm = get_esm_model(embedding_size=embedding_size)

    results = []
    num_peptides = len(all_peptides)

    with h5py.File(output_hdf5, 'w') as h5f:
        embedding_dset = h5f.create_dataset('embeddings', shape=(num_peptides, embedding_size), dtype=np.float32)

        for start_idx in range(0, num_peptides, batch_size):
            end_idx = min(start_idx + batch_size, num_peptides)
            batch_peptides = all_peptides[start_idx:end_idx]
            print(f"Processing batch {start_idx // batch_size + 1}/{(num_peptides - 1) // batch_size + 1} ({len(batch_peptides)} peptides)...")

            peptide_tuples = [(f"pep_{i}", p['peptide_sequence']) for i, p in enumerate(batch_peptides, start_idx)]
            batch_embeddings = get_esm_embeddings(
                peptide_tuples, model_esm, alphabet_esm, batch_converter_esm, device_esm,
                embedding_layer=embedding_layer, sequence_embedding=True
            )
            embedding_dset[start_idx:end_idx] = batch_embeddings

            for i, p in enumerate(batch_peptides):
                results.append({
                    "embedding_idx": start_idx + i,
                    "protein_id": p["protein_id"],
                    "peptide_sequence": p["peptide_sequence"],  # Include for NES positional info
                    "start_pos": p["start_pos"],
                    "end_pos": p["end_pos"],
                    "label": p["label"]
                })

    print(f"Successfully processed {len(results)} peptides! Embeddings saved to {output_hdf5}")
    return results, output_hdf5

def group_peptides_by_protein(peptide_embeddings):
    protein_groups = {}
    for peptide_info in peptide_embeddings:
        protein_id = peptide_info["protein_id"]
        if protein_id not in protein_groups:
            protein_groups[protein_id] = []
        protein_groups[protein_id].append(peptide_info)
    return protein_groups

def find_best_thresholds(prediction_scores, true_labels):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(true_labels, prediction_scores)
    youdens_index = tpr - fpr
    optimal_idx = np.argmax(youdens_index)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.3f} (TPR: {tpr[optimal_idx]:.3f}, FPR: {fpr[optimal_idx]:.3f})")
    return optimal_threshold, -optimal_threshold

def get_peptide_predictions(peptide_embeddings, trained_model, hdf5_file):
    from neural_net import get_net_scores
    print(f"Getting predictions for {len(peptide_embeddings)} peptides...")
    with h5py.File(hdf5_file, 'r') as h5f:
        embeddings_only = h5f['embeddings'][:]
    peptide_scores = get_net_scores(trained_model, embeddings_only)
    true_labels = [p["label"] for p in peptide_embeddings]
    UPPER_THRESHOLD, LOWER_THRESHOLD = find_best_thresholds(peptide_scores, true_labels)
    for i, (score, p) in enumerate(zip(peptide_scores, peptide_embeddings)):
        p["embedding"] = embeddings_only[p["embedding_idx"]]
        p["prediction_score"] = float(score)
        p["predicted_label"] = 1 if score > UPPER_THRESHOLD else -1 if score < LOWER_THRESHOLD else 0
        del p["embedding_idx"]
    print("✅ Peptide predictions completed!")
    return peptide_embeddings

def get_protein_level_predictions(peptide_embeddings_with_predictions):
    print("Aggregating peptide predictions to protein level...")
    protein_groups = group_peptides_by_protein(peptide_embeddings_with_predictions)
    protein_predictions = {}
    for protein_id, peptides in protein_groups.items():
        num_positive = 0
        num_uncertain = 0
        num_negative = 0
        max_score = float('-inf')
        score_sum = 0.0
        for peptide in peptides:
            label = peptide["predicted_label"]
            if label == 1:
                num_positive += 1
            elif label == 0:
                num_uncertain += 1
            else:
                num_negative += 1
            score = peptide["prediction_score"]
            if score > max_score:
                max_score = score
            score_sum += score
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

def get_nes_positional_info(protein_predictions, protein_groups):
    print("Extracting NES positional information...")
    nes_positions = {}
    for protein_id, pred_info in protein_predictions.items():
        if pred_info["predicted_label"] == 1:
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

def full_prediction_pipeline(all_peptides, trained_model, embedding_size=2560, embedding_layer=9, batch_size=10000,
                            output_hdf5="embeddings.h5", output_dir="results"):
    """
    Complete pipeline from raw peptide data to protein predictions, with result saving.

    Args:
        all_peptides: List of dicts with protein info and sequences
        trained_model: Trained SimpleDenseNet model
        embedding_size: ESM model size
        embedding_layer: ESM layer to extract
        batch_size: Number of peptides per batch
        output_hdf5: Path to HDF5 file for embeddings
        output_dir: Directory to save results

    Returns:
        Tuple: (peptide_predictions, protein_predictions, protein_groups, nes_positions)
    """
    print("🚀 Starting full prediction pipeline...")
    start_time = datetime.now()

    # Step 1: Get embeddings
    peptide_embeddings, hdf5_file = get_peptide_embeddings_with_protein_info(
        all_peptides, embedding_size, embedding_layer, batch_size, output_hdf5
    )

    # Step 2: Get peptide predictions
    peptide_predictions = get_peptide_predictions(peptide_embeddings, trained_model, hdf5_file)

    # Step 3: Aggregate to protein predictions
    protein_predictions, protein_groups = get_protein_level_predictions(peptide_predictions)

    # Step 4: Get NES positional info
    nes_positions = get_nes_positional_info(protein_predictions, protein_groups)

    # Step 5: Calculate accuracies
    peptide_accuracy = calculate_peptide_accuracy(peptide_predictions)
    protein_accuracy = calculate_protein_accuracy(protein_predictions, all_peptides)

    # Step 6: Save results
    save_results(peptide_predictions, protein_predictions, nes_positions, peptide_accuracy, protein_accuracy,
                 embedding_size, embedding_layer, batch_size, output_dir)

    end_time = datetime.now()
    print(f"🎉 Full pipeline completed in {(end_time - start_time).total_seconds() / 60:.2f} minutes!")
    return peptide_predictions, protein_predictions, protein_groups, nes_positions

def calculate_peptide_accuracy(peptide_embeddings_with_predictions):
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
            peptide_labels = protein_groups[protein_id]
            true_label = 1 if 1 in peptide_labels else 0 if 0 in peptide_labels else -1
            true_protein_labels.append(true_label)
            predicted_protein_labels.append(protein_predictions[protein_id]["predicted_label"])
    accuracy = accuracy_score(true_protein_labels, predicted_protein_labels)
    print(f"\n📊 PROTEIN-LEVEL ACCURACY:")
    print(f"Overall Accuracy: {accuracy:.3f}")
    print(f"Classification Report:")
    print(classification_report(true_protein_labels, predicted_protein_labels, target_names=['Negative', 'Uncertain', 'Positive']))
    return accuracy

if __name__ == "__main__":
    all_peptides = [
        {"protein_id": "P12345", "peptide_sequence": "MASNDYTQQATQSYGAYPTQP", "start_pos": 0, "end_pos": 22, "label": 1},
        {"protein_id": "P12345", "peptide_sequence": "ASNDYTQQATQSYGAYPTQPG", "start_pos": 1, "end_pos": 23, "label": 1},
        {"protein_id": "P67890", "peptide_sequence": "LKJHGFDSAPOIUYTREWQAS", "start_pos": 0, "end_pos": 22, "label": 0},
    ]
    import torch
    from neural_net import SimpleDenseNet
    print("Loading trained model...")
    trained_model = SimpleDenseNet(esm_emb_dim=2560, hidden_dim=128, dropout=0.2)
    trained_model.load_state_dict(torch.load('trained_network.pth'))
    trained_model.eval()
    print("✅ Model loaded successfully!")
    peptide_predictions, protein_predictions, protein_groups, nes_positions = full_prediction_pipeline(
        all_peptides, trained_model, embedding_size=2560, embedding_layer=9, batch_size=10000,
        output_hdf5="embeddings.h5", output_dir="results"
    )
    print(f"\n📊 RESULTS:")
    print(f"Processed {len(peptide_predictions)} peptides from {len(protein_predictions)} proteins")
    print(f"\nExample peptide prediction:")
    p = peptide_predictions[0]
    print(f"Protein ID: {p['protein_id']}, Score: {p['prediction_score']:.3f}, Predicted: {p['predicted_label']}")
    print(f"\nExample protein predictions:")
    for protein_id, pred in list(protein_predictions.items())[:2]:
        print(f"Protein {protein_id}: Label={pred['predicted_label']} "
              f"({pred['num_positive_peptides']} pos, {pred['num_uncertain_peptides']} unc, {pred['num_negative_peptides']} neg)")
    if nes_positions:
        print(f"\nNES Positions in positive proteins:")
        for protein_id, pos_info in list(nes_positions.items())[:2]:
            print(f"Protein {protein_id}: {pos_info['num_nes_regions']} NES regions")
            for region in pos_info['nes_regions']:
                print(f"  - Position {region['start_pos']}-{region['end_pos']} (score: {region['score']:.3f})")
