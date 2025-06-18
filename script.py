import numpy as np
from esm_embeddings import get_esm_embeddings, get_esm_model

UPPER_THRESHOLD = 1
LOWER_THRESHOLD = -1

def get_peptide_embeddings_with_protein_info(all_peptides, embedding_size=2560, embedding_layer=33):
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
    **MODIFIED** - Get protein-level predictions from peptide predictions.
    If ANY peptide in a protein is predicted as positive (label=1),
    then the entire protein is classified as positive.

    Args:
        peptide_embeddings_with_predictions: Output from get_peptide_predictions()

    Returns:
        Dict mapping protein_id to protein-level prediction
    """
    print("Aggregating peptide predictions to protein level...")

    # Group by protein and aggregate
    protein_groups = group_peptides_by_protein(peptide_embeddings_with_predictions)
    protein_predictions = {}

    for protein_id, peptides in protein_groups.items():
        # If ANY peptide is positive, protein is positive
        has_positive_peptide = any(p["predicted_label"] == 1 for p in peptides)
        max_score = max(p["prediction_score"] for p in peptides)
        mean_score = np.mean([p["prediction_score"] for p in peptides])

        protein_predictions[protein_id] = {
            "predicted_label": 1 if has_positive_peptide else 0,
            "max_peptide_score": float(max_score),
            "mean_peptide_score": float(mean_score),  # NEW: Average score
            "num_peptides": len(peptides),
            "num_positive_peptides": sum(p["predicted_label"] for p in peptides),
            "positive_peptide_ratio": sum(p["predicted_label"] for p in peptides) / len(peptides)  # NEW
        }

    print(f"✅ Protein-level predictions completed for {len(protein_predictions)} proteins!")
    return protein_predictions


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

    # Example peptide prediction
    print(f"\nExample peptide prediction:")
    p = peptide_predictions[0]
    print(f"Protein ID: {p['protein_id']}, Score: {p['prediction_score']:.3f}, Predicted: {p['predicted_label']}")

    # Example protein prediction
    print(f"\nExample protein predictions:")
    for protein_id, pred in list(protein_predictions.items())[:2]:
        print(f"Protein {protein_id}: {pred['predicted_label']} "
              f"({pred['num_positive_peptides']}/{pred['num_peptides']} peptides positive)")

    # OPTION 2: Step-by-step (if you need more control)
    # Step 1: Get embeddings
    # peptide_embeddings = get_peptide_embeddings_with_protein_info(all_peptides)
    # Step 2: Get predictions
    # peptide_predictions = get_peptide_predictions(peptide_embeddings, trained_model)
    # Step 3: Aggregate to proteins
    # protein_predictions = get_protein_level_predictions(peptide_predictions)
