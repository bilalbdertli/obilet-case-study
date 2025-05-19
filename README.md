# obilet-case-study



## Project Report: AI-Powered Hotel Room Visual Search System for oBilet

**Date:** May 19, 2025
**Author:** Bilal Berkam Dertli

**1. Introduction**

This report details the development of an AI-powered system designed to enhance oBilet's hotel reservation platform. The system allows users to perform advanced searches for hotel rooms based on visual preferences by analyzing and filtering hotel room images. The primary goal was to develop a proof-of-concept that can identify rooms meeting specific user criteria from a given dataset of images, fulfilling the tasks outlined in the Jr. AI Specialist case study.

**2. System Architecture and Workflow**

The system follows a multi-stage pipeline, processing hotel room images to enable both keyword-based and semantic search capabilities:

1.  **Image Acquisition:**
    *   **Script:** `download-images.py`
    *   **Process:** The initial set of 25 hotel room images, provided as URLs, were downloaded and saved locally into the `input-images/` directory. This ensures local access for processing.

2.  **Image Analysis & Feature Extraction (Image-to-Text):**
    *   **Script:** `analyze-images-llm.py` 
    *   **Process:** Each downloaded image was analyzed using a powerful vision-language model (Llama4 Maverick 17B). The model was prompted to extract key visual features from each room image.
    *   **Output:** The extracted features (e.g., room type, capacity, view type, amenities) were structured and saved into `hotel_features.json`. Each entry in this JSON file maps an image identifier to its corresponding set of features, including the `original_url`.

3.  **Textual Descriptor Generation & Vector Embedding:**
    *   **Script/Notebook:** `vector-embeddings-kaggle.ipynb` (or `vector-search.py` if adapted).
    *   **Process:**
        *   For each room in `hotel_features.json`, a concise textual descriptor was programmatically generated (e.g., "double room with a capacity for 2 people offering a mountain view equipped with balcony, air conditioning.").
        *   A pre-trained Sentence Transformer model from HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) was used to convert these textual descriptors into high-dimensional vector embeddings.
    *   **Output:** The generated embeddings, along with the original URLs, descriptors, and features, were stored in `hotel_embeddings_hf.json`. This file serves as the database for semantic search.

4.  **Search and Query Processing:**
    *   **Script:** `hotel_search.py` (integrates functionalities previously explored in `keyword-based-search.py` and `vector-search.py`).
    *   **Process:** This core script loads both `hotel_features.json` and `hotel_embeddings_hf.json`.
        *   **Keyword-Based Search:** Implements logic to filter rooms from `hotel_features.json` based on exact or partial matches of specified criteria (room type, amenities, view, capacity). This is the primary method for addressing the specific case study queries.
        *   **Vector-Based Semantic Search:** For a given free-text user query, the script embeds the query using the same Sentence Transformer model and then calculates cosine similarity against all room embeddings in `hotel_embeddings_hf.json` to find the most semantically similar rooms.
    *   **Output:** For each query, the script returns a list of matching image URLs. The results for the case study queries were also saved to `search-results.json`.

**3. Key Components and Technologies**

*   **Programming Language:** Python 3.11.9
*   **Image Downloading:** Llama 4 API call .
*   **Image Analysis (Image-to-Text):** Llama4 Maverick 17B was used for initial feature extraction.
*   **Feature Storage:** JSON format (`hotel_features.json`).
*   **Text Embedding:**
    *   HuggingFace `sentence-transformers` library.
    *   Model: `all-MiniLM-L6-v2`.
*   **Embedding Storage:** JSON format (`hotel_embeddings_hf.json`).
*   **Search Logic:**
    *   Custom Python functions for keyword filtering.
    *   `scikit-learn` for cosine similarity calculation in semantic search.
    *   `numpy` for numerical operations with embeddings.
*   **Development Environment:** Local Python environment, Jupyter Notebook (for `vector-embeddings-kaggle.ipynb`).

**4. Implementation of Case Study Queries**

The `hotel_search.py` script successfully addressed the four specific user queries outlined in the case study using primarily the keyword-based search mechanism on the `hotel_features.json` data. The results are as follows:

*   **1. Double rooms with a sea view:**
    *   `https_s3.eu-central-1.amazonaws.com_static.obilet.com_CaseStudy_HotelImages_23.jpg`
    *   `https_s3.eu-central-1.amazonaws.com_static.obilet.com_CaseStudy_HotelImages_14.jpg`
    *   `https_s3.eu-central-1.amazonaws.com_static.obilet.com_CaseStudy_HotelImages_13.jpg`
    *   `https_s3.eu-central-1.amazonaws.com_static.obilet.com_CaseStudy_HotelImages_21.jpg`
    *   `https_s3.eu-central-1.amazonaws.com_static.obilet.com_CaseStudy_HotelImages_17.jpg`
*   **2. Rooms with a balcony and air conditioning, with a city view:**
    *   No matching rooms found. (This indicates no single image in the dataset had all three features simultaneously as per the extracted data).
*   **3. Triple rooms with a desk:**
    *   `https_s3.eu-central-1.amazonaws.com_static.obilet.com_CaseStudy_HotelImages_2.jpg`
    *   `https_s3.eu-central-1.amazonaws.com_static.obilet.com_CaseStudy_HotelImages_7.jpg`
    *   `https_s3.eu-central-1.amazonaws.com_static.obilet.com_CaseStudy_HotelImages_8.jpg`
*   **4. Rooms with a maximum capacity of 4 people:**
    *   (List of 25 URLs as provided in the output, indicating all rooms met this criterion based on extracted capacity data).
    *   These results can also be found in `search_results.json` file.
**5. Semantic Search Capability**

In addition to keyword search, the system incorporates vector-based semantic search. This allows for more nuanced, free-form queries. The `hotel_search.py` script includes a demonstration of this, where a query like "a cozy room with a nice ocean view and a place to work" returns semantically similar rooms based on their generated descriptors and embeddings. This provides a more flexible search experience beyond strict keyword matching.

**6. Current Status & Deliverables**

The core functionalities of the system have been implemented. The key deliverables and files in the project directory are:

*   **Image Data:** `input-images/` (containing 25 downloaded images).
*   **Feature Data:** `hotel_features.json` (structured attributes for each image).
*   **Embedding Data:** `hotel_embeddings_hf.json` (text descriptors and vector embeddings).
*   **Scripts:**
    *   `download-images.py`: For image acquisition.
    *   `analyze-images-llm.py`: Conceptual script for image analysis (actual analysis performed via API).
    *   `vector-embeddings-kaggle.ipynb`: Notebook for generating embeddings.
    *   `hotel_search.py`: Main script for query processing and search.
    *   (`keyword-based-search.py`, `vector-search.py`): Earlier exploratory/modular scripts.
*   **Results:** `search-results.json` (output for the specific case study queries).
*   **Prompts:** `agent-prompts/system_prompt.txt`, `prompts/system_prompt.txt` (indicate exploration towards agent-based architectures).

**7. Future Enhancements & Optional Components**

The current system provides a strong foundation. Potential future enhancements include:

*   **Hybrid Search:** Develop a more sophisticated strategy to combine keyword and semantic search results for optimal relevance.
*   **Visual User Interface (UI):** Implement a user-friendly interface (e.g., using Streamlit or Flask) to allow interactive querying and display of image results.
*   **Agent-Based Architecture:** Fully implement an agent-to-agent or multi-module communication architecture (as suggested by the `agent-prompts` folders) for more complex query understanding, planning, and execution. This would provide extra credit as per the case study.
*   **Refined Feature Extraction:** Experiment with different prompting strategies for the vision model or use more specialized models to potentially improve the accuracy and granularity of extracted features.
*   **Scalability:** For larger datasets, integrate a dedicated vector database (e.g., FAISS, Pinecone) for more efficient semantic search.
*   **Error Handling & Logging:** Implement more robust error handling and logging throughout the pipeline.

**8. Conclusion**

The developed system successfully demonstrates the capability to analyze hotel room images, extract relevant features, and perform both keyword-based and semantic searches to match user criteria. It fulfills the core requirements of the oBilet case study by providing a functional pipeline and addressing the specific queries. The modular design and the inclusion of semantic search offer a flexible and extensible solution for enhancing visual search on oBilet's platform.
