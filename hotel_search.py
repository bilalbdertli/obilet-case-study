import json
import numpy as np
import os
import base64
from pathlib import Path
import time
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from azure.core.credentials import AzureKeyCredential
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Constants
DATABASE_FILE = "hotel_features.json"
EMBEDDINGS_FILE = "room_embeddings.json"
PROMPT_DIR = "prompts"

# Load environment variables
load_dotenv()

# Get Azure API credentials from environment variables
llm_model_name = os.getenv("VLM_MODEL_NAME")
llm_api_key = os.getenv("AZURE_LLAMA_4_MAVERICK_17B_API_KEY")
llm_endpoint = os.getenv("VLM_END_POINT")
llm_api_version = os.getenv("VLM_API_VERSION")

def initialize_model(json_response=True):
    """Initialize the Azure AI Chat model with the appropriate API key."""
    api_key = llm_api_key 
    
    kwargs = {}
    if json_response:
        kwargs["response_format"] = {"type": "json_object"}
    
    model = AzureAIChatCompletionsModel(
        model_name=llm_model_name,  
        endpoint=llm_endpoint,
        credential=AzureKeyCredential(api_key),
        api_version=llm_api_version,
        **kwargs
    )
    return model

def load_prompt_from_file(file_path):
    """Load prompt content from a file."""
    with open(file_path, "r") as file:
        return file.read()

def encode_image(image_path):
    """Encode image to base64 for API transmission"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_database():
    """Load the hotel features database"""
    with open(DATABASE_FILE, 'r') as f:
        return json.load(f)

def analyze_image(image_path, image_url):
    """Analyze hotel room image using Llama 4 Maverick 17B"""
    model = initialize_model(json_response=True)
    
    # Create system prompt
    system_prompt = """You are an AI assistant specialized in analyzing hotel room images. 
    Please analyze the image and extract the following features:
    1. Room type (single, double, triple, etc.)
    2. Maximum capacity (number of people)
    3. View type (sea view, city view, garden view, or none visible)
    4. Amenities present (desk, balcony, air conditioning, TV, etc.)
    
    Return ONLY a JSON object with these keys: room_type, capacity, view_type, amenities (list)
    """
    
    system_message = SystemMessage(content=system_prompt)
    
    # Create human message with image
    base64_image = encode_image(image_path)
    human_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpg;base64,{base64_image}"},
        }
    ]
    
    human_message = HumanMessage(content=human_content)
    
    # Create messages list
    messages = [system_message, human_message]
    
    try:
        # Get response
        response = model.invoke(messages)
        
        # Extract content
        result_json = json.loads(response.content)
        
        # Add original URL
        result_json["original_url"] = image_url
        
        return result_json
    
    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        return {"error": str(e), "original_url": image_url}

def generate_room_description(features):
    """Generate a descriptive text from room features for embedding"""
    room_type = features.get('room_type', 'unknown')
    capacity = features.get('capacity', 'unknown')
    view_type = features.get('view_type', 'no specific view')
    amenities = ', '.join(features.get('amenities', []))
    
    description = f"A {room_type} room with capacity for {capacity} people. "
    description += f"It has a {view_type}. "
    description += f"The room comes with the following amenities: {amenities}."
    
    return description

def get_embedding(text):
    """Get embedding for the given text using Llama 4 Maverick"""
    model = initialize_model(json_response=False)
    
    system_prompt = """You are an AI assistant that generates vector embeddings. 
    For the input text, generate a 1536-dimensional embedding vector that captures its semantic meaning.
    Return only the vector as a JSON array of numbers."""
    
    system_message = SystemMessage(content=system_prompt)
    human_message = HumanMessage(content=text)
    
    messages = [system_message, human_message]
    
    try:
        # This direct approach might not work with Llama 4 for embeddings,
        # so we're using a simpler approach by extracting keywords instead
        response = model.invoke(messages)
        
        # Alternative approach: extract key terms and use them for matching
        # This is simpler than true embeddings but can work for basic semantic matching
        system_prompt = """Extract the 20 most important semantic keywords from the text below.
        Return only a JSON array of keywords."""
        
        system_message = SystemMessage(content=system_prompt)
        human_message = HumanMessage(content=text)
        
        messages = [system_message, human_message]
        
        response = model.invoke(messages)
        keywords = json.loads(response.content)
        
        # Convert keywords to a simple binary vector (0/1) of most common hotel features
        # This is a simplified approach when true embeddings aren't available
        common_features = ["single", "double", "triple", "suite", "sea", "city", "mountain", 
                        "view", "balcony", "desk", "tv", "air conditioning", "shower", 
                        "bathtub", "wifi", "minibar", "capacity", "people"]
        
        vector = [1 if feature in " ".join(keywords).lower() else 0 for feature in common_features]
        return vector
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Return a fallback embedding (all zeros)
        return [0] * 18  # length of common_features

def generate_and_save_embeddings(database):
    """Generate and save embeddings for all rooms in the database"""
    embeddings = {}
    
    for url, features in database.items():
        description = generate_room_description(features)
        embedding = get_embedding(description)
        embeddings[url] = {
            "description": description,
            "embedding": embedding
        }
    
    with open(EMBEDDINGS_FILE, 'w') as f:
        json.dump(embeddings, f)
    
    return embeddings

def load_or_generate_embeddings(database):
    """Load embeddings from file or generate if not available"""
    try:
        with open(EMBEDDINGS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Generating embeddings for the first time...")
        return generate_and_save_embeddings(database)

def semantic_search(query, embeddings, top_n=25):
    """Perform semantic search based on embeddings"""
    query_embedding = get_embedding(query)
    similarity_scores = {}
    
    for url, data in embeddings.items():
        room_embedding = data["embedding"]
        
        # Calculate simple match score (count of matching features)
        # This is a simplified approach when true embeddings aren't available
        matches = sum(1 for q, r in zip(query_embedding, room_embedding) if q == 1 and r == 1)
        total_query_features = sum(query_embedding)
        
        # Avoid division by zero
        if total_query_features > 0:
            similarity = matches / total_query_features
        else:
            similarity = 0
            
        similarity_scores[url] = similarity
    
    sorted_results = sorted(
        similarity_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    return [url for url, score in sorted_results[:top_n] if score > 0]

def keyword_search(criteria, database):
    """Keyword-based search function"""
    matching_urls = []
    
    for url, features in database.items():
        match = True
        
        # Check room type if specified
        if 'room_type' in criteria and criteria['room_type'].lower() != features.get('room_type', '').lower():
            match = False
            continue
            
        # Check capacity if specified
        if 'capacity' in criteria:
            # Handle "maximum" capacity query
            if criteria.get('capacity_type') == 'maximum':
                if features.get('capacity', 0) > criteria['capacity']:
                    match = False
                    continue
            # Handle exact capacity match
            elif features.get('capacity', 0) != criteria['capacity']:
                match = False
                continue
                
        # Check view type if specified
        if 'view_type' in criteria and criteria['view_type'].lower() not in features.get('view_type', '').lower():
            match = False
            continue
            
        # Check amenities if specified
        if 'amenities' in criteria:
            room_amenities = [amenity.lower() for amenity in features.get('amenities', [])]
            for required_amenity in criteria['amenities']:
                if required_amenity.lower() not in room_amenities:
                    match = False
                    break
        
        if match:
            matching_urls.append(url)
    
    return matching_urls

def hybrid_search(keyword_criteria, semantic_query, database, embeddings, weights=(0.7, 0.3)):
    """
    Hybrid search combining keyword-based and semantic search
    
    Args:
        keyword_criteria: Criteria for keyword search
        semantic_query: Natural language query for semantic search
        database: The room features database
        embeddings: The room embeddings
        weights: Weights for (keyword, semantic) results
    
    Returns:
        List of URLs sorted by combined score
    """
    # Get keyword search results (1 for match, 0 for no match)
    keyword_results = {}
    for url in database.keys():
        if url in keyword_search(keyword_criteria, database):
            keyword_results[url] = 1.0
        else:
            keyword_results[url] = 0.0
    
    # Get semantic search results
    semantic_urls = semantic_search(semantic_query, embeddings)
    semantic_results = {}
    for url in database.keys():
        if url in semantic_urls:
            # Create a score based on position in the results
            position = semantic_urls.index(url) + 1
            semantic_results[url] = 1.0 / position
        else:
            semantic_results[url] = 0.0
    
    # Combine results using weights
    combined_scores = {}
    for url in database.keys():
        keyword_score = keyword_results.get(url, 0)
        semantic_score = semantic_results.get(url, 0)
        combined_scores[url] = (
            weights[0] * keyword_score + 
            weights[1] * semantic_score
        )
    
    # Sort by combined score
    sorted_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Return only URLs with non-zero scores
    return [url for url, score in sorted_results if score > 0]

def run_predefined_queries(database, embeddings):
    """Run the four predefined queries using hybrid search"""
    results = {}
    
    # Query 1: Double rooms with a sea view
    query1_keyword = {
        'room_type': 'double',
        'view_type': 'sea'
    }
    query1_semantic = "I want a double room with a beautiful sea view"
    results['Double rooms with a sea view'] = hybrid_search(
        query1_keyword, query1_semantic, database, embeddings
    )
    
    # Query 2: Rooms with a balcony and air conditioning, with a city view
    query2_keyword = {
        'view_type': 'city',
        'amenities': ['balcony', 'air conditioning']
    }
    query2_semantic = "I need a room that has both a balcony and air conditioning, and offers a view of the city"
    results['Rooms with a balcony and air conditioning, with a city view'] = hybrid_search(
        query2_keyword, query2_semantic, database, embeddings
    )
    
    # Query 3: Triple rooms with a desk
    query3_keyword = {
        'room_type': 'triple',
        'amenities': ['desk']
    }
    query3_semantic = "I need a triple room that has a desk for working"
    results['Triple rooms with a desk'] = hybrid_search(
        query3_keyword, query3_semantic, database, embeddings
    )
    
    # Query 4: Rooms with a maximum capacity of 4 people
    query4_keyword = {
        'capacity': 4,
        'capacity_type': 'maximum'
    }
    query4_semantic = "I need a room that can accommodate at most 4 people"
    results['Rooms with a maximum capacity of 4 people'] = hybrid_search(
        query4_keyword, query4_semantic, database, embeddings
    )
    
    return results

def create_streamlit_app():
    """Create a simple Streamlit interface for the hotel room search system"""
    st.title("Hotel Room Search System")
    
    # Load database and embeddings
    database = load_database()
    embeddings = load_or_generate_embeddings(database)
    
    st.write(f"Loaded database with {len(database)} hotel room images")
    
    # Sidebar for search options
    st.sidebar.header("Search Options")
    search_type = st.sidebar.radio(
        "Search Type",
        ["Predefined Queries", "Custom Search"]
    )
    
    if search_type == "Predefined Queries":
        st.header("Predefined Queries")
        
        if st.button("Run All Predefined Queries"):
            results = run_predefined_queries(database, embeddings)
            
            for query, urls in results.items():
                st.subheader(query)
                
                if urls:
                    for url in urls:
                        st.write(f"- {url}")
                        # Display the image
                        st.image(url, width=300)
                else:
                    st.write("No matching rooms found")
    
    else:
        st.header("Custom Search")
        
        # Room type selection
        room_type = st.selectbox(
            "Room Type",
            ["Any", "Single", "Double", "Triple", "Suite"]
        )
        
        # Capacity selection
        capacity_type = st.radio(
            "Capacity Type",
            ["Exact", "Maximum"]
        )
        capacity = st.number_input("Capacity (number of people)", min_value=1, max_value=10, value=2)
        
        # View type selection
        view_type = st.selectbox(
            "View Type",
            ["Any", "Sea View", "City View", "Mountain View", "Garden View"]
        )
        
        # Amenities selection
        amenities = st.multiselect(
            "Required Amenities",
            ["Desk", "Balcony", "Air Conditioning", "TV", "Bathtub", "Shower", "Mini-bar"]
        )
        
        # Semantic query input
        semantic_query = st.text_input("Describe what you're looking for (for semantic search)")
        
        if st.button("Search"):
            # Build criteria dictionary
            criteria = {}
            
            if room_type != "Any":
                criteria["room_type"] = room_type.lower()
                
            criteria["capacity"] = capacity
            criteria["capacity_type"] = capacity_type.lower()
                
            if view_type != "Any":
                criteria["view_type"] = view_type.lower().replace(" view", "")
                
            if amenities:
                criteria["amenities"] = [a.lower() for a in amenities]
            
            # Perform search
            if semantic_query:
                results = hybrid_search(criteria, semantic_query, database, embeddings)
            else:
                results = keyword_search(criteria, database)
            
            # Display results
            st.subheader("Search Results")
            if results:
                for url in results:
                    st.write(f"- {url}")
                    # Display the image
                    st.image(url, width=300)
            else:
                st.write("No matching rooms found")

def main():
    # Load database
    database = load_database()
    print(f"Loaded database with {len(database)} hotel room images")
    
    # Load or generate embeddings
    embeddings = load_or_generate_embeddings(database)
    print(f"Loaded embeddings for {len(embeddings)} rooms")
    
    # Run the predefined queries
    query_results = run_predefined_queries(database, embeddings)
    
    # Display the results
    print("\n=== SEARCH RESULTS ===")
    for query, urls in query_results.items():
        print(f"\n{query}:")
        if urls:
            for url in urls:
                print(f"- {url}")
        else:
            print("No matching rooms found")
    
    # Save results to a JSON file
    with open("search_results.json", "w") as f:
        json.dump(query_results, f, indent=2)
    print("\nResults saved to search_results.json")
    
    print("\nTo launch the web interface, run 'streamlit run this_script.py'")

if __name__ == "__main__":
    # Check if running under Streamlit
    if 'STREAMLIT_RUN_PATH' in os.environ:
        create_streamlit_app()
    else:
        main()
