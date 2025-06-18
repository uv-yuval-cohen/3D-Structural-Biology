import numpy as np
from esm_embeddings import get_esm_embeddings, get_esm_model


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
            "embedding": embedding,                           # The 2560-dim vector
            "protein_id": all_peptides[i]["protein_id"],     # Original protein ID
            "start_pos": all_peptides[i]["start_pos"],       # Position in protein
            "end_pos": all_peptides[i]["end_pos"],           # End position
            "label": all_peptides[i]["label"]                # Original label
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


def get_protein_level_predictions(peptide_embeddings, trained_model):
    """
    Get protein-level predictions based on peptide-level predictions.
    If ANY peptide in a protein is predicted as positive (label=1), 
    then the entire protein is classified as positive.
    
    Args:
        peptide_embeddings: Output from get_peptide_embeddings_with_protein_info()
        trained_model: Your trained SimpleDenseNet model
    
    Returns:
        Dict mapping protein_id to protein-level prediction
    """
    from neural_net import get_net_scores
    
    # Get predictions for all peptides
    embeddings_only = [p["embedding"] for p in peptide_embeddings]
    peptide_scores = get_net_scores(trained_model, embeddings_only)
    
    # Add scores back to peptide info
    for i, score in enumerate(peptide_scores):
        peptide_embeddings[i]["prediction_score"] = score
        peptide_embeddings[i]["predicted_label"] = 1 if score > 0 else 0
    
    # Group by protein and aggregate
    protein_groups = group_peptides_by_protein(peptide_embeddings)
    protein_predictions = {}
    
    for protein_id, peptides in protein_groups.items():
        # If ANY peptide is positive, protein is positive
        has_positive_peptide = any(p["predicted_label"] == 1 for p in peptides)
        max_score = max(p["prediction_score"] for p in peptides)
        
        protein_predictions[protein_id] = {
            "predicted_label": 1 if has_positive_peptide else 0,
            "max_peptide_score": max_score,
            "num_peptides": len(peptides),
            "num_positive_peptides": sum(p["predicted_label"] for p in peptides)
        }
    
    return protein_predictions


# Example usage:
if __name__ == "__main__":
    # Example data structure (replace with your actual data)
    all_peptides = [
        {"protein_id": "P12345", "peptide_sequence": "MASNDYTQQATQSYGAYPTQP", "start_pos": 0, "end_pos": 22, "label": 1},
        {"protein_id": "P12345", "peptide_sequence": "ASNDYTQQATQSYGAYPTQPG", "start_pos": 1, "end_pos": 23, "label": 1},
        {"protein_id": "P67890", "peptide_sequence": "LKJHGFDSAPOIUYTREWQAS", "start_pos": 0, "end_pos": 22, "label": 0},
    ]
    
    # Get embeddings with protein info preserved
    peptide_embeddings = get_peptide_embeddings_with_protein_info(
        all_peptides, 
        embedding_size=2560,
        embedding_layer=33
    )
    
    # Print example of what you get
    print(f"\nExample peptide embedding info:")
    print(f"Protein ID: {peptide_embeddings[0]['protein_id']}")
    print(f"Start position: {peptide_embeddings[0]['start_pos']}")
    print(f"Embedding shape: {peptide_embeddings[0]['embedding'].shape}")
    print(f"Label: {peptide_embeddings[0]['label']}")
    
    # Group by protein for analysis
    protein_groups = group_peptides_by_protein(peptide_embeddings)
    print(f"\nFound {len(protein_groups)} unique proteins")
    for protein_id, peptides in protein_groups.items():
        print(f"Protein {protein_id}: {len(peptides)} peptides")
