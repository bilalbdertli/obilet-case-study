import json
import pprint

# Load the feature database
DATABASE_FILE = "hotel_features.json"

def load_database():
    """Load the hotel features database"""
    with open(DATABASE_FILE, 'r') as f:
        return json.load(f)

def search_rooms(criteria, database):
    """
    Search for rooms matching the given criteria
    
    Args:
        criteria: A dictionary with keys corresponding to room features
        database: The hotel features database
        
    Returns:
        A list of URLs of matching rooms
    """
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
            matching_urls.append(features.get('original_url', url))
    
    return matching_urls

def run_predefined_queries(database):
    """Run the four predefined queries from the case study"""
    results = {}
    
    # Query 1: Double rooms with a sea view
    query1 = {
        'room_type': 'double',
        'view_type': 'sea'
    }
    results['Double rooms with a sea view'] = search_rooms(query1, database)
    
    # Query 2: Rooms with a balcony and air conditioning, with a city view
    query2 = {
        'view_type': 'city',
        'amenities': ['balcony', 'air conditioning']
    }
    results['Rooms with a balcony and air conditioning, with a city view'] = search_rooms(query2, database)
    
    # Query 3: Triple rooms with a desk
    query3 = {
        'room_type': 'triple',
        'amenities': ['desk']
    }
    results['Triple rooms with a desk'] = search_rooms(query3, database)
    
    # Query 4: Rooms with a maximum capacity of 4 people
    query4 = {
        'capacity': 4,
        'capacity_type': 'maximum'
    }
    results['Rooms with a maximum capacity of 4 people'] = search_rooms(query4, database)
    
    return results

def main():
    # Load the database
    database = load_database()
    print(f"Loaded database with {len(database)} hotel room images")
    
    # Run the predefined queries
    query_results = run_predefined_queries(database)
    
    # Display the results
    print("\n=== SEARCH RESULTS ===")
    for query, urls in query_results.items():
        print(f"\n{query}:")
        if urls:
            for url in urls:
                print(f"- {url}")
        else:
            print("No matching rooms found")
    
    # Optional: Save results to a JSON file
    with open("search_results.json", "w") as f:
        json.dump(query_results, f, indent=2)
    print("\nResults saved to search_results.json")

if __name__ == "__main__":
    main()
