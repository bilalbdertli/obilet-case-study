import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
FEATURES_FILE = "hotel_features.json"
EMBEDDINGS_FILE_HF = "hotel_embeddings_hf.json"
EMBED_MODEL_HF = "sentence-transformers/all-MiniLM-L6-v2"

# Global model instance (load once)
MODEL = None

# ──────────────────────────────────────────────────────────────────────────────
# Load Data and Model
# ──────────────────────────────────────────────────────────────────────────────
def load_sentence_transformer_model():
    global MODEL
    if MODEL is None:
        print(f"Loading HuggingFace model: {EMBED_MODEL_HF}...")
        try:
            MODEL = SentenceTransformer(EMBED_MODEL_HF)
            print("HuggingFace model loaded successfully.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            MODEL = None
    return MODEL

def load_feature_database():
    """Load the hotel features database (for keyword search)."""
    if not os.path.exists(FEATURES_FILE):
        print(f"Error: {FEATURES_FILE} not found.")
        return None
    with open(FEATURES_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {FEATURES_FILE}.")
            return None

def load_embeddings_database():
    """Load the hotel embeddings database (for semantic search)."""
    if not os.path.exists(EMBEDDINGS_FILE_HF):
        print(f"Error: {EMBEDDINGS_FILE_HF} not found.")
        return None
    with open(EMBEDDINGS_FILE_HF, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {EMBEDDINGS_FILE_HF}.")
            return None

# ──────────────────────────────────────────────────────────────────────────────
# Search Functions
# ──────────────────────────────────────────────────────────────────────────────
def keyword_search_rooms(criteria, feature_db):
    """
    Search for rooms matching given criteria using keyword matching.
    Args:
        criteria (dict): Dictionary with keys like 'room_type', 'view_type', 
                         'amenities' (list), 'capacity', 'capacity_type' ('exact' or 'maximum').
        feature_db (dict): The loaded hotel features database.
    Returns:
        list: A list of original_urls of matching rooms.
    """
    if not feature_db:
        return []
        
    matching_urls = []
    for room_id, features in feature_db.items():
        if not isinstance(features, dict): # Skip if features are not a dict
            continue

        match = True
        
        # Room Type
        if 'room_type' in criteria:
            room_type_feature = str(features.get('room_type', '')).lower()
            if criteria['room_type'].lower() not in room_type_feature:
                match = False
        
        # Capacity
        if 'capacity' in criteria and match:
            room_capacity = features.get('capacity')
            if room_capacity is None: # Room has no capacity info
                match = False
            else:
                try:
                    room_capacity = int(room_capacity)
                    required_capacity = int(criteria['capacity'])
                    capacity_type = criteria.get('capacity_type', 'exact')

                    if capacity_type == 'maximum':
                        if room_capacity > required_capacity:
                            match = False
                    elif capacity_type == 'exact': # Default or explicit exact
                        if room_capacity != required_capacity:
                             match = False
                    # Could add 'minimum' if needed
                    # elif capacity_type == 'minimum':
                    #     if room_capacity < required_capacity:
                    #         match = False
                except ValueError: # If capacity is not a number
                    match = False


        # View Type
        if 'view_type' in criteria and match:
            view_type_feature = str(features.get('view_type', '')).lower()
            if criteria['view_type'].lower() not in view_type_feature:
                match = False
        
        # Amenities (all required amenities must be present)
        if 'amenities' in criteria and match:
            room_amenities = [str(a).lower() for a in features.get('amenities', [])]
            required_amenities = [str(ra).lower() for ra in criteria['amenities']]
            if not all(req_a in room_amenities for req_a in required_amenities):
                match = False
        
        if match:
            matching_urls.append(features.get('original_url', room_id))
            
    return list(set(matching_urls)) # Return unique URLs

def get_query_embedding_hf(text: str, model_instance):
    """Generate an embedding for a query string."""
    if model_instance is None:
        raise ValueError("SentenceTransformer model is not loaded.")
    return model_instance.encode(text)

def semantic_search_rooms(query_text, embedding_db, model_instance, top_k=5):
    """
    Search for rooms using semantic similarity.
    Args:
        query_text (str): The user's free-form query.
        embedding_db (list): The loaded hotel embeddings database.
        model_instance (SentenceTransformer): The loaded sentence transformer model.
        top_k (int): Number of top results to return.
    Returns:
        list: List of dicts, each with "url" and "score".
    """
    if not embedding_db or model_instance is None:
        return []

    query_embedding = get_query_embedding_hf(query_text, model_instance)
    query_embedding_np = np.array(query_embedding).reshape(1, -1)

    room_embeddings_matrix = []
    valid_rooms_data = []

    for item in embedding_db:
        if "embedding" in item and isinstance(item["embedding"], list):
            room_embeddings_matrix.append(item["embedding"])
            valid_rooms_data.append(item)
    
    if not room_embeddings_matrix:
        return []
        
    room_embeddings_matrix_np = np.array(room_embeddings_matrix)
    
    similarities = cosine_similarity(query_embedding_np, room_embeddings_matrix_np)[0]
    
    # Get top_k indices, sorted by score descending
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for i in top_k_indices:
        if i < len(valid_rooms_data) and similarities[i] > 0: # Optionally add a similarity threshold
            results.append({
                "url": valid_rooms_data[i]["url"],
                "descriptor": valid_rooms_data[i].get("descriptor", "N/A"),
                "score": float(similarities[i])
            })
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Process Specific Queries from Case Study
# ──────────────────────────────────────────────────────────────────────────────
def run_case_study_queries(feature_db):
    """Runs the predefined queries from the oBilet case study."""
    results = {}
    
    print("\n--- Running Case Study Queries (Keyword-Based) ---")

    # Query 1: Double rooms with a sea view
    criteria1 = {'room_type': 'double', 'view_type': 'sea'}
    results["1. Double rooms with a sea view"] = keyword_search_rooms(criteria1, feature_db)
    
    # Query 2: Rooms with a balcony and air conditioning, with a city view
    criteria2 = {'amenities': ['balcony', 'air conditioning'], 'view_type': 'city'}
    results["2. Rooms with a balcony and air conditioning, with a city view"] = keyword_search_rooms(criteria2, feature_db)
    
    # Query 3: Triple rooms with a desk
    criteria3 = {'room_type': 'triple', 'amenities': ['desk']}
    results["3. Triple rooms with a desk"] = keyword_search_rooms(criteria3, feature_db)
    
    # Query 4: Rooms with a maximum capacity of 4 people
    criteria4 = {'capacity': 4, 'capacity_type': 'maximum'}
    results["4. Rooms with a maximum capacity of 4 people"] = keyword_search_rooms(criteria4, feature_db)
    
    # Print results
    for query_desc, urls in results.items():
        print(f"\n{query_desc}:")
        if urls:
            for url in urls:
                print(f"  - {url}")
        else:
            print("  No matching rooms found.")
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Load data
    feature_db = load_feature_database()
    embedding_db = load_embeddings_database()
    st_model = load_sentence_transformer_model()

    if not feature_db:
        print("Exiting: Feature database is required for keyword search.")
        return

    # Run the specific case study queries (primarily keyword-based)
    run_case_study_queries(feature_db)

    # --- Example of Semantic Search (for more free-form queries) ---
    if embedding_db and st_model:
        print("\n\n--- Example Semantic Search ---")
        semantic_query = "a cozy room with a nice ocean view and a place to work"
        print(f"Semantic Query: \"{semantic_query}\"")
        semantic_results = semantic_search_rooms(semantic_query, embedding_db, st_model, top_k=3)
        
        if semantic_results:
            for res in semantic_results:
                print(f"  - URL: {res['url']} (Score: {res['score']:.4f})")
                # print(f"    Descriptor: {res['descriptor']}")
        else:
            print("  No semantic matches found for this example query.")
    else:
        print("\nSkipping semantic search example as embeddings or model could not be loaded.")

if __name__ == "__main__":
    main()
