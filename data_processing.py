import pandas as pd
import requests
from Bio import SeqIO

"""Classification Guide for Experimental Labels:

    This mapping defines which protein labels are considered NES-positive (1) 
    and which are treated as NES-negative (0) based on experimental annotations.

    Label interpretations:
        - "Cargo A"               → 1 (Confirmed NES-containing cargo)
        - "Low abundant cargo"    → 1 (NES-positive, but low expression levels)
        - "Cargo B"               → 0 (Unclear/ambiguous cargo classification)
        - "Ambiguous"             → 0 (Uncertain or inconsistent evidence)
        - "NTR"                   → -1 (Nuclear transport receptor, not cargo)
        - "NUP"                   → -1 (Nucleoporin protein, structural, not cargo)
        - "Non binder"            → -1 (Validated lack of Crm1 binding)
        - "CRM1 Cofactor"       → -1 (Cofactor, not considered cargo)
        - "NPC"                 → -1 (Nuclear pore complex component, not cargo)
"""

LABEL_TO_BINARY = {
    "Cargo A": 1,
    "Low abundant cargo": 1,
    "Cargo B": 0, #maybe
    "Ambiguous": 0, #maybe
    "NTR": -1,
    "NUP": -1,
    "Non-binder": -1,
    "CRM1 Cofactor": -1,
    "NPC":-1
}


def fetch_sequence_from_uniprot(protein_id):
    """
        Fetches the protein sequence from UniProt given a UniProt protein ID.

        Parameters:
        -----------
        protein_id : str
            The UniProt identifier of the protein.

        Returns:
        --------
        str or None
            The protein sequence as a string, or None if retrieval failed.
        """

    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.fasta"
    try:
        response = requests.get(url, headers={"Accept": "text/x-fasta"}, verify=False)
        if response.status_code == 200:
            lines = response.text.strip().split("\n")
            sequence = "".join(lines[1:])  # Skip header line
            return sequence
        else:
            print(f"Failed for {protein_id} with status {response.status_code}")
    except Exception as e:
        print(f"Error fetching {protein_id}: {e}")
    return None

def enrich_csv_with_sequences(input_csv, output_csv):
    """
    Loads a CSV containing UniProt IDs and functional labels, fetches the protein sequences
    from UniProt, and adds both the sequence and binary label (NES presence: 1 or 0).

    Input CSV should have two columns:
        - Column 1: protein_id (e.g., "Q9NTX7")
        - Column 2: original_value (functional label, e.g., "Cargo A", "NUP", etc.)

    Parameters:
        input_csv (str): Path to input CSV file.
        output_csv (str): Path to save the enriched CSV file.
    """

    # Load input CSV (assumes first column is UniProt ID, second is other data)
    df = pd.read_csv(input_csv)
    df.columns = ["protein_id", "original_value"]

    # Add sequence for each protein
    sequences = []

    for pid in df["protein_id"]:
        seq = fetch_sequence_from_uniprot(pid)
        print(seq)
        sequences.append(seq)

    df["sequence"] = sequences

    # Drop rows where sequence wasn't found
    df = df.dropna(subset=["sequence"])

    # Save to new CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

def label_category(value):
    """
        Classify a string label into a binary NES class.

        Args:
            value (str or None): The label describing the protein's classification
                                 (e.g., 'Cargo A', 'Non binder', etc.).

        Returns:
            int:
                - 1 for known or likely NES-positive classes (e.g., "Cargo A")
                - 0 for ambiguous or NES-negative classes (e.g., "Cargo B", "Non binder")
                - -1 if the label is missing or unrecognized
        """
    if pd.isna(value):
        return -1
    val = str(value).strip().lower()
    for label, binary in LABEL_TO_BINARY.items():
        if val == label.lower():
            return binary
    return -1

def relabel_data(input_csv: str, output_csv: str):
    """
        Read a CSV file with textual NES classification labels and convert them to binary values.

        The function:
        - Loads a CSV with at least a column named "original_value"
        - Applies binary classification (1 for NES-positive, 0 for NES-negative, -1 for unknown)
        - Saves the modified DataFrame with the binary labels to a new CSV

        Args:
            input_csv (str): Path to the input CSV file
            output_csv (str): Path to save the output CSV file with binary-labeled data
        """
    df = pd.read_csv(input_csv)
    df["original_value"] = df["original_value"].apply(label_category)
    # Remove rows where sequence is missing
    df = df.dropna(subset=["sequence"])
    df.to_csv(output_csv, index=False)
    print(f"Saved classified values to {output_csv}")

 # enrich_csv_with_sequences("deep_proteomics_data_3.csv",'data.csv')
# relabel_data('data.csv' , "data_classified.csv")

def generate_peptides_from_csv(csv_path, window_size=22):
    """
    Reads a CSV file with columns: 'protein_id', 'original_value', 'sequence',
    and generates a list of peptide windows (as dicts) of fixed size from each protein.

    Returns:
        List[Dict]: each dict contains:
            - protein_id (str)
            - peptide_sequence (str)
            - start_pos (int)
            - end_pos (int)
            - label (int)
    """
    df = pd.read_csv(csv_path)
    all_peptides = []

    for _, row in df.iterrows():
        protein_id = row["protein_id"]
        sequence = row["sequence"]
        print(sequence)
        print(protein_id)
        print(f"len of sequence {len(sequence)}")

        label = int(row["original_value"])

        for i in range(len(sequence) - window_size + 1):
            peptide = sequence[i:i+window_size]
            peptide_entry = {
                "protein_id": protein_id,
                "peptide_sequence": peptide,
                "start_pos": i,
                "end_pos": i + window_size,
                "label": label
            }
            all_peptides.append(peptide_entry)

    return all_peptides


def extract_mitochondrial_proteins_to_csv(fasta_path="human_proteins.fasta", output_csv="mitochondrial_proteins.csv"):
    """
    Extracts mitochondrial proteins from a FASTA file and saves them to a CSV file.

    Criteria:
        - 'mitochondrial' (case-insensitive) must appear in the FASTA description line.

    Output CSV columns:
        - protein_id
        - description
        - sequence
    """
    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        if "mitochondrial" in record.description.lower():
            # Extract UniProt ID if possible
            parts = record.id.split("|")
            protein_id = parts[1] if len(parts) > 1 else record.id
            records.append({
                "protein_id": protein_id,
                "description": record.description,
                "sequence": str(record.seq)
            })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} mitochondrial proteins to {output_csv}")


def create_negative_peptides_from_mito(input_csv="mitochondrial_proteins.csv",
                                       output_csv="mitochondrial_negatives.csv",
                                       peptide_length=22):
    """
    Extracts the first 22 amino acids from each sequence in the mitochondrial_proteins.csv
    and saves them as a single-column CSV file.

    Args:
        input_csv (str): Path to input CSV with a 'sequence' column.
        output_csv (str): Path to save the resulting peptide list.
        peptide_length (int): Length of peptides to extract from start of sequence.
    """
    df = pd.read_csv(input_csv)
    negatives = []

    for seq in df["sequence"]:
        if isinstance(seq, str) and len(seq) >= peptide_length:
            negatives.append(seq[:peptide_length])

    result_df = pd.DataFrame(negatives, columns=["peptide"])
    result_df.to_csv(output_csv, index=False)
    print(f"Saved {len(result_df)} negative peptides to {output_csv}")

create_negative_peptides_from_mito("mitochondrial_proteins.csv", "mitochondrial_negatives.csv")


def main():
    extract_mitochondrial_proteins_to_csv()

    # input_file = "deep_proteomics_data_3.csv"
    # enriched_file = "data.csv"
    # classified_file = "data_classified.csv"
    #
    # enrich_csv_with_sequences(input_file, enriched_file)
    # relabel_data(enriched_file, classified_file)
    # peptides = generate_peptides_from_csv(classified_file)
    #
    # print(f"Generated {len(peptides)} peptides.")
    # return peptides


