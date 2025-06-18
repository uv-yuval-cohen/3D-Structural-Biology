import numpy as np
from esm_embeddings import get_esm_embeddings, get_esm_model

UPPER_THRESHOLD = 1 # default upper threshold for positive predictions
LOWER_THRESHOLD = -1 # default lower threshold for negative predictions

def get_peptide_embeddings_with_protein_info(all_peptides, embedding_size=2560, embedding_layer=9):
    """
    Get ESM embeddings for peptides while maintaining protein connection.

    Args:
        all_peptides: List of dicts with keys: protein_id, peptide_sequence, start_pos, end_pos, label
        embedding_size: ESM model embedding dimension (default: 2560)
        embedding_layer: Which transformer layer to extract embeddings from (default: 33)

    Returns:
        List of dicts with keys: embedding, protein_id, start_pos, end_pos, label
    """

    # Step 1: Load ESM model
    print("Loading ESM-2 model...")
    model_esm, alphabet_esm, batch_converter_esm, device_esm = get_esm_model(embedding_size=embedding_size)

    # Step 2: Convert to format expected by get_esm_embeddings
    print(f"Converting {len(all_peptides)} peptides to ESM format...")
    peptide_tuples = [(f"pep_{i}", peptide['peptide_sequence'])
                      for i, peptide in enumerate(all_peptides)]

    # Step 3: Get embeddings
    print("Getting ESM embeddings...")
    all_embeddings = get_esm_embeddings(
        peptide_tuples,
        model_esm,
        alphabet_esm,
        batch_converter_esm,
        device_esm,
        embedding_layer=embedding_layer,
        sequence_embedding=True
    )

    # Step 4: Map back using indices (this is the magic!)
    print("Mapping embeddings back to protein information...")
    results = []
    for i, embedding in enumerate(all_embeddings):
        results.append({
            "embedding": embedding,  # The 2560-dim vector
            "protein_id": all_peptides[i]["protein_id"],  # Original protein ID
            "start_pos": all_peptides[i]["start_pos"],  # Position in protein
            "end_pos": all_peptides[i]["end_pos"],  # End position
            "label": all_peptides[i]["label"]  # Original label
        })

    print(f"Successfully processed {len(results)} peptides!")
    return results


def group_peptides_by_protein(peptide_embeddings):
    """
    Group peptide embeddings by protein_id for protein-level analysis.

    Args:
        peptide_embeddings: Output from get_peptide_embeddings_with_protein_info()

    Returns:
        Dict mapping protein_id to list of peptide info
    """
    protein_groups = {}
    for peptide_info in peptide_embeddings:
        protein_id = peptide_info["protein_id"]
        if protein_id not in protein_groups:
            protein_groups[protein_id] = []
        protein_groups[protein_id].append(peptide_info)

    return protein_groups


def find_best_thresholds(prediction_scores, true_labels):
    """
    Find optimal thresholds using ROC curve analysis.
    Finds the point that maximizes TPR while minimizing FPR (Youden's Index).

    Args:
        prediction_scores: Array of model prediction scores
        true_labels: Array of true labels (0 or 1)

    Returns:
        Tuple: (upper_threshold, lower_threshold)
    """
    from sklearn.metrics import roc_curve

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, prediction_scores)

    # Find optimal threshold using Youden's Index (TPR - FPR)
    youdens_index = tpr - fpr
    optimal_idx = np.argmax(youdens_index)
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal threshold: {optimal_threshold:.3f} (TPR: {tpr[optimal_idx]:.3f}, FPR: {fpr[optimal_idx]:.3f})")

    # You can adjust these based on your needs
    upper_threshold = optimal_threshold
    lower_threshold = -optimal_threshold

    return upper_threshold, lower_threshold


def get_peptide_predictions(peptide_embeddings, trained_model):
    """
    **NEW FUNCTION** - Get predictions for all peptides efficiently using batch processing.

    Args:
        peptide_embeddings: Output from get_peptide_embeddings_with_protein_info()
        trained_model: Your trained SimpleDenseNet model

    Returns:
        Same list as input but with added 'prediction_score' and 'predicted_label' keys
    """
    from neural_net import get_net_scores

    print(f"Getting predictions for {len(peptide_embeddings)} peptides...")

    # EFFICIENT BATCH PREDICTION: Extract all embeddings into one batch
    embeddings_only = [p["embedding"] for p in peptide_embeddings]

    # Single batch inference - much faster than individual predictions!
    peptide_scores = get_net_scores(trained_model, embeddings_only)

    # Extract true labels for threshold optimization
    true_labels = [p["label"] for p in peptide_embeddings]

    # Find optimal thresholds
    UPPER_THRESHOLD, LOWER_THRESHOLD = find_best_thresholds(peptide_scores, true_labels)

    # Map predictions back using indices (same principle as embedding mapping)
    for i, score in enumerate(peptide_scores):
        peptide_embeddings[i]["prediction_score"] = float(score)  # Convert to Python float
        if score > UPPER_THRESHOLD:
            peptide_embeddings[i]["predicted_label"] = 1 # positive
        elif score < LOWER_THRESHOLD:
            peptide_embeddings[i]["predicted_label"] = -1 # negative
        else:
            peptide_embeddings[i]["predicted_label"] = 0 # unknown

    print("✅ Peptide predictions completed!")
    return peptide_embeddings


def get_protein_level_predictions(peptide_embeddings_with_predictions):
    """
    **MODIFIED** - Get protein-level predictions using 3-label hierarchy.
    - Protein = 1: If ANY peptide is labeled 1 (confident positive)
    - Protein = 0: If NO peptides are 1, but ANY peptide is labeled 0 (uncertain)
    - Protein = -1: If ALL peptides are labeled -1 (confident negative)
    """
    print("Aggregating peptide predictions to protein level...")

    protein_groups = group_peptides_by_protein(peptide_embeddings_with_predictions)
    protein_predictions = {}

    for protein_id, peptides in protein_groups.items():
        # Count each label type
        positive_peptides = [p for p in peptides if p["predicted_label"] == 1]
        uncertain_peptides = [p for p in peptides if p["predicted_label"] == 0]
        negative_peptides = [p for p in peptides if p["predicted_label"] == -1]

        # Hierarchical classification
        if len(positive_peptides) > 0:
            protein_label = 1
        elif len(uncertain_peptides) > 0:
            protein_label = 0
        else:
            protein_label = -1

        max_score = max(p["prediction_score"] for p in peptides)
        mean_score = np.mean([p["prediction_score"] for p in peptides])

        protein_predictions[protein_id] = {
            "predicted_label": protein_label,
            "max_peptide_score": float(max_score),
            "mean_peptide_score": float(mean_score),
            "num_peptides": len(peptides),
            "num_positive_peptides": len(positive_peptides),
            "num_uncertain_peptides": len(uncertain_peptides),
            "num_negative_peptides": len(negative_peptides),
            "positive_peptide_ratio": len(positive_peptides) / len(peptides)
        }

    print(f"✅ Protein-level predictions completed for {len(protein_predictions)} proteins!")
    return protein_predictions


def get_nes_positional_info(peptide_embeddings_with_predictions, protein_predictions):
    """
    **NEW FUNCTION** - Extract NES positional information for positive proteins.

    Args:
        peptide_embeddings_with_predictions: Output from get_peptide_predictions()
        protein_predictions: Output from get_protein_level_predictions()

    Returns:
        Dict mapping protein_id to NES location information
    """
    print("Extracting NES positional information...")

    protein_groups = group_peptides_by_protein(peptide_embeddings_with_predictions)
    nes_positions = {}

    for protein_id, pred_info in protein_predictions.items():
        if pred_info["predicted_label"] == 1:  # Only for positive proteins
            peptides = protein_groups[protein_id]
            positive_peptides = [p for p in peptides if p["predicted_label"] == 1]

            # Extract positions of all positive peptides
            nes_regions = []
            for peptide in positive_peptides:
                nes_regions.append({
                    "start_pos": peptide["start_pos"],
                    "end_pos": peptide["end_pos"],
                    "sequence": peptide.get("peptide_sequence", ""),  # If available
                    "score": peptide["prediction_score"]
                })

            # Sort by start position
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

def full_prediction_pipeline(all_peptides, trained_model, embedding_size=2560, embedding_layer=9):
    """
    **NEW FUNCTION** - Complete pipeline from raw peptide data to protein predictions.

    Args:
        all_peptides: List of dicts with protein info and sequences
        trained_model: Your trained SimpleDenseNet model
        embedding_size: ESM model size (default: 2560)
        embedding_layer: ESM layer to extract (default: 33)

    Returns:
        Tuple: (peptide_predictions, protein_predictions)
    """
    print("🚀 Starting full prediction pipeline...")

    # Step 1: Get embeddings with protein info
    peptide_embeddings = get_peptide_embeddings_with_protein_info(
        all_peptides, embedding_size, embedding_layer
    )

    # Step 2: Get peptide-level predictions (EFFICIENT BATCH PROCESSING)
    peptide_predictions = get_peptide_predictions(peptide_embeddings, trained_model)

    # Step 3: Aggregate to protein-level predictions
    protein_predictions = get_protein_level_predictions(peptide_predictions)

    print("🎉 Full pipeline completed!")
    return peptide_predictions, protein_predictions


def calculate_peptide_accuracy(peptide_embeddings_with_predictions):
    """
    **NEW FUNCTION** - Calculate peptide-level accuracy metrics.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    true_labels = [p["label"] for p in peptide_embeddings_with_predictions]
    predicted_labels = [p["predicted_label"] for p in peptide_embeddings_with_predictions]

    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"\n📊 PEPTIDE-LEVEL ACCURACY:")
    print(f"Overall Accuracy: {accuracy:.3f}")
    print(f"Classification Report:")
    print(classification_report(true_labels, predicted_labels,
                                target_names=['Negative', 'Uncertain', 'Positive']))
    print(f"Confusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))

    return accuracy


def calculate_protein_accuracy(protein_predictions, all_peptides):
    """
    **NEW FUNCTION** - Calculate protein-level accuracy metrics.
    """
    from sklearn.metrics import accuracy_score, classification_report

    # Get true protein labels (if ANY peptide in protein has label=1, protein is positive)
    protein_groups = {}
    for peptide in all_peptides:
        protein_id = peptide["protein_id"]
        if protein_id not in protein_groups:
            protein_groups[protein_id] = []
        protein_groups[protein_id].append(peptide["label"])

    true_protein_labels = []
    predicted_protein_labels = []
    protein_ids = []

    for protein_id in protein_groups.keys():
        if protein_id in protein_predictions:
            # True label: 1 if any peptide is 1, 0 if any peptide is 0, else -1
            peptide_labels = protein_groups[protein_id]
            if 1 in peptide_labels:
                true_label = 1
            elif 0 in peptide_labels:
                true_label = 0
            else:
                true_label = -1

            true_protein_labels.append(true_label)
            predicted_protein_labels.append(protein_predictions[protein_id]["predicted_label"])
            protein_ids.append(protein_id)

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


    peptide_predictions, protein_predictions = full_prediction_pipeline(
        all_peptides,
        trained_model,
        embedding_size=2560,
        embedding_layer=9
    )

    # Print results
    print(f"\n📊 RESULTS:")
    print(f"Processed {len(peptide_predictions)} peptides from {len(protein_predictions)} proteins")

    # Calculate accuracies
    peptide_accuracy = calculate_peptide_accuracy(peptide_predictions)
    protein_accuracy = calculate_protein_accuracy(protein_predictions, all_peptides)

    # Get NES positional information
    nes_positions = get_nes_positional_info(peptide_predictions, protein_predictions)

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
