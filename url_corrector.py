import json
import os
import re # For extracting the image number

# --- Configuration ---
INPUT_FEATURES_FILE = "hotel_features.json"
OUTPUT_FEATURES_FILE = "hotel_features_corrected.json"

INPUT_EMBEDDINGS_FILE = "hotel_embeddings_hf.json"
OUTPUT_EMBEDDINGS_FILE = "hotel_embeddings_hf_corrected.json"

BASE_WEB_URL = "https://static.obilet.com.s3.eu-central-1.amazonaws.com/CaseStudy/HotelImages/"

def extract_image_number_from_filename(filename_key):
    """
    Extracts the image number from the local-style filename.
    Example: "https_..._HotelImages_1.jpg" -> "1"
    """
    match = re.search(r'_(\d+)\.jpg$', filename_key)
    if match:
        return match.group(1)
    print(f"Warning: Could not extract image number from '{filename_key}'")
    return None

def construct_correct_url(image_number):
    """Constructs the full HTTPS URL for a given image number."""
    if image_number:
        return f"{BASE_WEB_URL}{image_number}.jpg"
    return None

def correct_hotel_features(input_filepath, output_filepath):
    """Corrects URLs in the hotel_features.json file."""
    if not os.path.exists(input_filepath):
        print(f"Error: Input file '{input_filepath}' not found.")
        return

    with open(input_filepath, 'r') as f:
        try:
            features_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from '{input_filepath}': {e}")
            return

    corrected_features_data = {}
    updated_count = 0

    for local_filename_key, features_dict in features_data.items():
        image_number = extract_image_number_from_filename(local_filename_key)
        correct_url = construct_correct_url(image_number)

        if correct_url:
            # Update the original_url field within the features dictionary
            if isinstance(features_dict, dict):
                features_dict["original_url"] = correct_url
                updated_count +=1
            else:
                print(f"Warning: Entry for '{local_filename_key}' is not a dictionary. Skipping.")
            
            # We can keep the original local_filename_key for the corrected data,
            # or change the key to the correct_url if preferred.
            # For now, let's keep the original key and ensure the 'original_url' field is correct.
            corrected_features_data[local_filename_key] = features_dict
        else:
            # If URL couldn't be corrected, keep original entry
            corrected_features_data[local_filename_key] = features_dict


    with open(output_filepath, 'w') as f:
        json.dump(corrected_features_data, f, indent=2)
    
    print(f"Corrected URLs in '{input_filepath}' and saved to '{output_filepath}'.")
    print(f"Total entries processed: {len(features_data)}")
    print(f"Entries with updated 'original_url': {updated_count}")


def correct_hotel_embeddings(input_filepath, output_filepath):
    """Corrects URLs in the hotel_embeddings_hf.json file."""
    if not os.path.exists(input_filepath):
        print(f"Error: Input file '{input_filepath}' not found.")
        return

    with open(input_filepath, 'r') as f:
        try:
            embeddings_data = json.load(f) # This should be a list
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from '{input_filepath}': {e}")
            return

    if not isinstance(embeddings_data, list):
        print(f"Error: Expected '{input_filepath}' to contain a JSON list.")
        return

    updated_count = 0
    for item_dict in embeddings_data:
        if "url" in item_dict and isinstance(item_dict["url"], str):
            local_style_url = item_dict["url"]
            image_number = extract_image_number_from_filename(local_style_url)
            correct_url = construct_correct_url(image_number)

            if correct_url:
                item_dict["url"] = correct_url # Update the main "url" field
                updated_count +=1

                # Also update original_url within the nested "features" dictionary, if it exists
                if "features" in item_dict and isinstance(item_dict["features"], dict):
                    if "original_url" in item_dict["features"]:
                        item_dict["features"]["original_url"] = correct_url
        else:
            print(f"Warning: Item in embeddings list missing 'url' field or 'url' is not a string: {item_dict}")


    with open(output_filepath, 'w') as f:
        json.dump(embeddings_data, f, indent=2)

    print(f"Corrected URLs in '{input_filepath}' and saved to '{output_filepath}'.")
    print(f"Total items processed in embeddings list: {len(embeddings_data)}")
    print(f"Items with updated 'url' (and nested 'original_url'): {updated_count}")


if __name__ == "__main__":
    print("--- Correcting hotel_features.json ---")
    # First, save your provided JSON content into a file named "hotel_features.json"
    # For demonstration, I'll create it here if it doesn't exist, using a small part of your data.
    # In your actual use, ensure the full file is present.
    if not os.path.exists(INPUT_FEATURES_FILE):
        print(f"'{INPUT_FEATURES_FILE}' not found. Please create it with your JSON data.")
        # Example:
        # dummy_features_content = {
        #   "https_s3.eu-central-1.amazonaws.com_static.obilet.com_CaseStudy_HotelImages_1.jpg": {
        #     "room_type": "double", "capacity": 2, "view_type": "mountain view",
        #     "amenities": ["balcony", "air conditioning"],
        #     "original_url": "https_s3.eu-central-1.amazonaws.com_static.obilet.com_CaseStudy_HotelImages_1.jpg"
        #   }}
        # with open(INPUT_FEATURES_FILE, 'w') as f:
        #    json.dump(dummy_features_content, f, indent=2)
    else:
        correct_hotel_features(INPUT_FEATURES_FILE, OUTPUT_FEATURES_FILE)

    print("\n--- Correcting hotel_embeddings_hf.json ---")
    # Ensure your "hotel_embeddings_hf.json" file is present.
    if not os.path.exists(INPUT_EMBEDDINGS_FILE):
        print(f"'{INPUT_EMBEDDINGS_FILE}' not found. Please create it.")
        # Example:
        # dummy_embeddings_content = [{
        #    "url": "https_s3.eu-central-1.amazonaws.com_static.obilet.com_CaseStudy_HotelImages_1.jpg",
        #    "descriptor": "double room...", "features": {"original_url": "https_..._1.jpg"}, "embedding": []
        # }]
        # with open(INPUT_EMBEDDINGS_FILE, 'w') as f:
        #    json.dump(dummy_embeddings_content, f, indent=2)
    else:
        correct_hotel_embeddings(INPUT_EMBEDDINGS_FILE, OUTPUT_EMBEDDINGS_FILE)

    print("\nCorrection process complete. Please check the '_corrected.json' files.")
    print(f"After correction, use '{OUTPUT_FEATURES_FILE}' and '{OUTPUT_EMBEDDINGS_FILE}' in your Streamlit app.")

