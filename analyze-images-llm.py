import os
from dotenv import load_dotenv
import gradio as gr
import json
import base64
import re
import io
import uuid
import time
from azure.core.credentials import AzureKeyCredential
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI API credentials from environment variables
llm_model_name = os.getenv("VLM_MODEL_NAME")
llm_api_key = os.getenv("AZURE_LLAMA_4_MAVERICK_17B_API_KEY")
llm_endpoint = os.getenv("VLM_END_POINT")
llm_api_version = os.getenv("VLM_API_VERSION")


IMAGE_DIR = "input-images"
OUTPUT_FILE = "hotel_features.json"
PROMPT_DIR = "agent-prompts"

# Loading prompts from their corresponding txt files and setting them to a dictionary.
def load_prompt_from_file(filepath):
    """Loads a prompt from a text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

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

def encode_image(image_path):
    """Encode image to base64 for API transmission"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path, image_url):
    """Analyze hotel room image using GPT-4 Vision API"""
    model = initialize_model(json_response=True)
    system_prompt = load_prompt_from_file(
        os.path.join(PROMPT_DIR, "system_prompt.txt")
    )
    system_message = SystemMessage(content=system_prompt)
    messages=[]
    base64_image = encode_image(image_path)
    human_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpg;base64,{base64_image}"},
            }
        ]   
    initial_human_message = HumanMessage(content=human_content)
    # system_message = SystemMessage(content=load_prompt_from_file(os.path.join(PROMPT_DIR, "system_prompt.txt")))
    messages.extend([system_message, initial_human_message])
    print("Calling LLM via API...")
    try:
        response = model.invoke(messages)
        print("LLM Response Received.")
        # Add AI response to history *immediately* for context in potential retries
        messages.append(AIMessage(content=response.content)) # Add the actual AIMessage object
        content = response.content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0]
        else:
            json_str = content

        features = json.loads(json_str)
        features["original_url"] = image_url
        return features
    except Exception as e:
            print(f"Error parsing response for {image_path}: {e}")
            return {"error": str(e), "original_url": image_url}
    
def main():
    # Process all images
    features_database = {}
    
    # Convert to actual URLs for the database
    url_mapping = {}
    for i in range(1, 26):  # For all 25 images
        local_filename = f"https_static.obilet.com.s3.eu-central-1.amazonaws.com_CaseStudy_HotelImages_{i}.jpg"
        original_url = f"https://static.obilet.com.s3.eu-central-1.amazonaws.com/CaseStudy/HotelImages/{i}.jpg"
        url_mapping[local_filename] = original_url
    
    for image_file in os.listdir(IMAGE_DIR):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(IMAGE_DIR, image_file)
            original_url = url_mapping.get(image_file, image_file)
            
            print(f"Analyzing {image_file}...")
            features = analyze_image(image_path, original_url)
            
            # Store results
            features_database[original_url] = features
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
    
    # Save the features database
    with open(OUTPUT_FILE, "w") as f:
        json.dump(features_database, f, indent=2)
    
    print(f"Analysis complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()