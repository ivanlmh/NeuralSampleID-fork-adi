import os
import json
import torch
import numpy as np
from collections import defaultdict
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

from eval import load_memmap_data, get_index

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def calculate_map(ground_truth, predictions, k=10):
    """
    Computes the Mean Average Precision (MAP) at k.

    Parameters:
    - ground_truth: Dictionary mapping query IDs to correct matches.
    - predictions: Dictionary where each query ID maps to its list of retrieved tracks.
    - k: Number of top results to consider.

    Returns:
    - MAP score.
    """
    average_precisions = []
    
    print(f"Number of queries in predictions: {len(predictions)}")
    print(f"Number of queries in ground_truth: {len(ground_truth)}")

    # Check if there's any overlap between prediction keys and ground truth
    overlap = set(predictions.keys()).intersection(set(ground_truth.keys()))
    print(f"Overlap between predictions and ground truth: {len(overlap)}/{len(predictions)}")

    for q_id, retrieved_list in predictions.items():

        if q_id not in ground_truth:
            print(f"Warning: Query {q_id} not in ground truth")
        else: 
            # Get the correct reference tracks for this query
            correct_refs = ground_truth[q_id]

        num_relevant = 0
        precision_values = []

        for i, retrieved_id in enumerate(retrieved_list[:k]):
            # # Check if this retrieved reference is in the correct list for this query
            if retrieved_id in correct_refs:
            # if q_id in ground_truth.get(retrieved_id, []):
                num_relevant += 1
                precision_values.append(num_relevant / (i + 1))  # Precision@i

        ap = np.mean(precision_values) if precision_values else 0
        average_precisions.append(ap)

        
        # Debug for first few queries
        if q_id in list(predictions.keys())[:5]:
            print(f"Query {q_id}: AP = {ap:.4f}, Relevant items: {num_relevant}/{min(k, len(retrieved_list))}")


    return np.mean(average_precisions) if average_precisions else 0



def extract_test_ids(lookup_table):
    starts = []
    lengths = []
    
    # Initialize the first string and starting index
    current_string = lookup_table[0]
    current_start = 0
    
    # Iterate through the list to detect changes in strings
    for i in range(1, len(lookup_table)):
        if lookup_table[i] != current_string:
            # When a new string is found, record the start and length of the previous group
            starts.append(current_start)
            lengths.append(i - current_start)
            
            # Update the current string and starting index
            current_string = lookup_table[i]
            current_start = i
    
    # Add the last group
    starts.append(current_start)
    lengths.append(len(lookup_table) - current_start)
    
    return np.array(starts), np.array(lengths)


def load_sample_class(csv_path='samples_new_new.csv', sample_class=None):
    """
    Load and filter sample information from CSV based on specified classifications.
    
    Args:
        csv_path (str): Path to the CSV file.
        sample_class (str, optional): Filter by classification:
            - 'beat': Only samples of type "beat"
            - 'riff': Only samples of type "riff"
            - 'interpolation_no': Only samples with interpolation="no"
            - 'interpolation_maybe': Only samples with interpolation not equal to "no"
            - 'high_time_stretching': Samples with tempo ratio > 1.05
            - 'low_time_stretching': Samples with 1.0 <= tempo ratio <= 1.05
            
    Returns:
        dict: Mapping of query track IDs to sample types that match all criteria.
    """
    import pandas as pd
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} total samples from CSV")

        # Print all the query IDs before filtering
        all_query_ids = sorted(set(df['query_track_id'].astype(str)))
        print(f"All unique query IDs in CSV: {len(all_query_ids)}")
        
        # Apply filter based on sample_class
        if sample_class:
            if sample_class == 'beat' or sample_class == 'riff' or sample_class == '1-note':
                filtered_df = df[df['sample_type'].str.lower() == sample_class.lower()]
                print(f"Found {len(filtered_df)} samples of type '{sample_class}'")
            
            elif sample_class == 'interpolation_no':
                filtered_df = df[df['interpolation'].str.lower() == 'no']
                print(f"Found {len(filtered_df)} samples with interpolation='no'")
            
            elif sample_class == 'interpolation_maybe':
                filtered_df = df[(df['interpolation'].str.lower() == 'yes')]# | (df['interpolation'].str.lower() == 'probably')]
                print(f"Found {len(filtered_df)} samples with interpolation not equal to 'no'")
            
            elif sample_class == 'high_time_stretching':
                # Handle potential missing or non-numeric values
                numeric_df = df[pd.to_numeric(df['estimated_tempo_ratio'], errors='coerce').notna()]
                filtered_df = numeric_df[abs(pd.to_numeric(numeric_df['estimated_tempo_ratio'])-1) > 0.05]
                print(f"Found {len(filtered_df)} samples with time stretching factor > 1.05")
            
            elif sample_class == 'low_time_stretching':
                # Handle potential missing or non-numeric values
                numeric_df = df[pd.to_numeric(df['estimated_tempo_ratio'], errors='coerce').notna()]
                filtered_df = numeric_df[abs(pd.to_numeric(numeric_df['estimated_tempo_ratio'])-1) < 0.05]
                print(f"Found {len(filtered_df)} samples with 1.0 <= time stretching factor <= 1.05")
            
            else:
                print(f"Unknown sample class: {sample_class}, returning all samples")
                filtered_df = df
        else:
            filtered_df = df
        
        # Create the final mapping
        sample_types = {str(row['query_track_id']): True for _, row in filtered_df.iterrows()}
        print(f"Returning {len(sample_types)} filtered sample mappings")
        return sample_types
        
    except Exception as e:
        print(f"Error loading or filtering sample types: {e}")
        return {}


def eval_faiss_map_clf(emb_dir, classifier, emb_dummy_dir=None,
                       index_type='ivfpq', nogpu=False, max_train=1e7,
                       k_probe=3, n_centroids=32, k_map=20, sample_class=None):
    """
    Evaluation using classifier logits instead of cosine similarity.
    """

    # Load sample types if filtering is requested
    filtered_samples = None
    if sample_class:
        filtered_samples = load_sample_class(sample_class=sample_class)
        print(f"Will filter evaluation to sample class: {sample_class}")

    classifier.to(device).eval()

    query_nmatrix_path = os.path.join(emb_dir, 'query_nmatrix.npy')
    ref_nmatrix_dir = os.path.join(emb_dir, 'ref_nmatrix')

    # Load FAISS index
    query, query_shape = load_memmap_data(emb_dir, 'query_full_db')
    db, db_shape = load_memmap_data(emb_dir, 'ref_db')
    if emb_dummy_dir is None:
        emb_dummy_dir = emb_dir
    dummy_db, dummy_db_shape = load_memmap_data(emb_dummy_dir, 'dummy_db')

    index = get_index(index_type, dummy_db, dummy_db.shape, (not nogpu), max_train, n_centroids=n_centroids)
    index.add(dummy_db)
    index.add(db)
    del dummy_db

    # Load lookup tables
    query_lookup = json.load(open(f'{emb_dir}/query_full_db_lookup.json', 'r'))
    ref_lookup = json.load(open(f'{emb_dir}/ref_db_lookup.json', 'r'))  

    with open('data/gt_dict.json', 'r') as fp:
        ground_truth = json.load(fp)
    
    # Invert the ground truth to match our prediction structure
    inverted_ground_truth = {}
    for ref_id, query_list in ground_truth.items():
        for query_id in query_list:
            if query_id not in inverted_ground_truth:
                inverted_ground_truth[query_id] = []
            inverted_ground_truth[query_id].append(ref_id)
        # Use inverted_ground_truth instead of ground_truth for MAP calculation

    # Load query node matrices
    query_nmatrix = np.load(query_nmatrix_path, allow_pickle=True).item()
    test_ids, max_test_seq_len = extract_test_ids(query_lookup)
    ref_song_starts, _ = extract_test_ids(ref_lookup)
    
    predictions = {}

    processed_queries = 0
    filtered_queries = 0

    print("Starting FAISS-based retrieval and ranking...")

    for ix, test_id in enumerate(test_ids):
        q_id = query_lookup[test_id].split("_")[0]

        # Skip this query if it doesn't match the filter
        if sample_class and filtered_samples:
            if q_id not in filtered_samples:
                print(f"Warning: Query {q_id} not found in sample types mapping")
                continue
                
            actual_type = filtered_samples.get(q_id, "unknown")
            if not actual_type:#.lower() != sample_class.lower():
                print(f"Skipping query {q_id} with type '{actual_type}' (looking for '{sample_class}')")
                continue
            
            print(f"Including query {q_id} with type '{actual_type}'")
            filtered_queries += 1
        
        processed_queries += 1

        max_len = max_test_seq_len[ix]
        q = query[test_id: test_id + max_len, :]

        _, I = index.search(q, k_probe)

        candidates, freqs = np.unique(I[I >= 0], return_counts=True)
        print(f"\nQuery {ix}: Retrieved {len(candidates)} candidates.")

        hist = defaultdict(int)
        for cid, freq in zip(candidates, freqs):
            if cid < dummy_db_shape[0]:
                continue
            cid = cid - dummy_db_shape[0]
            match = ref_lookup[cid]
            if match == q_id:
                continue

            # Get correct segment index in the retrieved song
            song_start_idx = ref_song_starts[ref_song_starts <= cid].max()
            ref_seg_idx = cid - song_start_idx

            # Load the reference node matrix
            ref_nmatrix_path = os.path.join(ref_nmatrix_dir, f"{match}.npy")
            if not os.path.exists(ref_nmatrix_path):
                print(f"Missing reference matrix for {match}, skipping...")
                continue

            ref_nmatrix = np.load(ref_nmatrix_path)  # (num_segments, C, N)
            if ref_seg_idx >= ref_nmatrix.shape[0]:
                print(f"Segment index {ref_seg_idx} out of bounds for {match}, skipping...")
                continue  

            nm_candidate = torch.tensor(ref_nmatrix[ref_seg_idx]).to(device)
            nm_query = torch.tensor(query_nmatrix[q_id]).to(device)

            # Ensure nm_candidate is repeated across nm_query's segments
            nm_candidate = nm_candidate.unsqueeze(0).repeat(nm_query.shape[0], 1, 1)  # (num_segments, C, N)

            # Compute classifier logits in batch mode
            logits = classifier(nm_query, nm_candidate)  # (num_segments, 1)

            clf_score = logits.max().item()
            # print(f"Classifier score for {match}: {classifier_score:.4f} (before freq weighting)")

            # Multiply by frequency
            # weighted_score = clf_score * np.log1p(freq) if clf_score > 0.5 else 0
            weighted_score = clf_score if clf_score > 0.5 else 0
            hist[match] += weighted_score
            # print(f"Updated hist[{match}] = {hist[match]:.4f} (after weighting with freq={freq})")

        print(f"Top 10 scores for {q_id}: {sorted(hist.items(), key=lambda x: x[1], reverse=True)[:10]}")
        if ix % 5 == 0:
            print(f"Processed {ix} / {len(test_ids)} queries...")

        predictions[q_id] = sorted(hist, key=hist.get, reverse=True)
        
    # Compute MAP
    print("\nComputing MAP score...")

    if sample_class and filtered_samples:
        print(f"Processed {processed_queries} total queries, {filtered_queries} were '{sample_class}' type")
        
        # Create the filtered mappings
        filtered_ground_truth = {}
        for q_id in predictions.keys():
            if q_id in inverted_ground_truth:
                filtered_ground_truth[q_id] = inverted_ground_truth[q_id]
            else:
                print(f"Warning: Query {q_id} in predictions but not in inverted ground truth")

        print(f"Filtered ground truth has {len(filtered_ground_truth)} entries")
        

        map_score = calculate_map(filtered_ground_truth, predictions, k=k_map)
        print(f"MAP@{k_map} calculated on {len(predictions)} {sample_class} samples")
    else:
        map_score = calculate_map(ground_truth, predictions, k=k_map)
        print(f"MAP@{k_map} calculated on all {len(predictions)} samples")


    # map_score = calculate_map(ground_truth, predictions, k=k_map)
    np.save(f'{emb_dir}/predictions.npy', predictions)
    np.save(f'{emb_dir}/map_score.npy', map_score)
    
    print(f"MAP score computed: {map_score:.4f}")
    print(f"Saved predictions and MAP score to {emb_dir}")

    return map_score, k_map

