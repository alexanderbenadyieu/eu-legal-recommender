#!/usr/bin/env python3
"""
Streamlit interface for the EU Legal Recommender System.

This web app provides a user-friendly interface for interacting with the 
recommender system, including options for query-based, document ID-based,
and profile-based recommendations.
"""
import streamlit as st
import sys
import os
import json
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
import dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
dotenv.load_dotenv(Path(__file__).parent.parent / ".env")

# Import and apply dictionary patches to prevent dictionary addition errors
from src.utils.dict_patch import apply_patches
apply_patches()  # Apply patches before any other imports

# Import recommender components
from src.models.pinecone_recommender import PineconeRecommender
from src.models.personalized_recommender import PersonalizedRecommender
from src.models.features import FeatureProcessor
from src.config import PINECONE, EMBEDDER

# Import UI components
from streamlit_app.components.ui import (
    display_recommendations_with_formatting,
    display_profile_info,
    get_available_profiles,
    get_sample_document_ids,
    load_profile_data
)

# Import document cache
from streamlit_app.document_cache import (
    get_document_from_cache,
    add_document_to_cache,
    get_recommendations_from_cache,
    add_recommendations_to_cache,
    generate_cache_key,
    get_document_from_database
)

# Set up page config
st.set_page_config(
    page_title="EU Legal Recommender",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def draw_header():
    """Draw the header and app title."""
    # Add custom CSS for better styling of document preview and summaries
    st.markdown("""
    <style>
    .document-preview {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .summary-box {
        background-color: #eff7ff;
        border-left: 3px solid #0d6efd;
        padding: 10px;
        margin-top: 10px;
        border-radius: 3px;
    }
    
    .recommendation-card {
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .recommendation-card h3 {
        color: #0d6efd;
        border-bottom: 1px solid #e9ecef;
        padding-bottom: 8px;
    }
    
    .recommendation-card a {
        color: #0d6efd;
        text-decoration: none;
        font-weight: bold;
    }
    
    .recommendation-card a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Set logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        # Use a reliable source for the EU flag image
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Flag_of_Europe.svg/1280px-Flag_of_Europe.svg.png", width=80)
    with col2:
        st.title("EU Legal Recommender System")
        st.markdown("Get personalized recommendations for EU legal documents based on query, document ID, or client profile.")

def load_environment_variables():
    """Load environment variables from inputs or .env file."""
    # API keys
    default_api_key = os.environ.get("PINECONE_API_KEY", "pcsk_qJoJ7_GwY1eWQLu6274ViWq47FNZrcbMM8XpmgXkQ98rWzePpLyVsgM4HpEbQFDTzEYmP")
    default_environment = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
    
    pinecone_api_key = st.session_state.get("pinecone_api_key", default_api_key)
    pinecone_environment = st.session_state.get("pinecone_environment", default_environment)
    
    if not pinecone_api_key:
        st.error("Pinecone API key is required to use the recommender system.")
        st.stop()
    
    # Set environment variables
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["PINECONE_ENVIRONMENT"] = pinecone_environment
    
    return pinecone_api_key

def draw_sidebar():
    """Draw the sidebar with configuration options."""
    st.sidebar.title("Configuration")
    
    # API Keys
    st.sidebar.header("API Keys")
    with st.sidebar.expander("API Settings", expanded=False):
        # Get default values from environment variables or use hardcoded values
        default_api_key = os.environ.get("PINECONE_API_KEY", "pcsk_qJoJ7_GwY1eWQLu6274ViWq47FNZrcbMM8XpmgXkQ98rWzePpLyVsgM4HpEbQFDTzEYmP")
        default_environment = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
        
        pinecone_api_key = st.text_input(
            "Pinecone API Key", 
            value=st.session_state.get("pinecone_api_key", default_api_key),
            type="password",
            key="pinecone_api_key"
        )
        
        pinecone_environment = st.text_input(
            "Pinecone Environment",
            value=st.session_state.get("pinecone_environment", default_environment),
            key="pinecone_environment"
        )
    
    # Recommendation Settings
    st.sidebar.header("Recommendation Settings")
    
    top_k = st.sidebar.slider(
        "Number of recommendations", 
        min_value=1, 
        max_value=20, 
        value=st.session_state.get("top_k", 5),
        key="top_k"
    )
    
    filter_type = st.sidebar.selectbox(
        "Filter by document type",
        ["", "regulation", "directive", "decision", "recommendation", "other"],
        index=0,
        key="filter_type"
    )
    
    # Temporal boosting
    use_temporal_boost = st.sidebar.checkbox(
        "Use temporal boosting", 
        value=st.session_state.get("use_temporal_boost", False),
        key="use_temporal_boost"
    )
    
    if use_temporal_boost:
        temporal_boost = st.sidebar.slider(
            "Temporal boost factor", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.get("temporal_boost", 0.3),
            step=0.1,
            key="temporal_boost"
        )
        
        reference_date = st.sidebar.date_input(
            "Reference date", 
            value=st.session_state.get("reference_date", datetime.now()),
            key="reference_date"
        )
    
    # Visualization Settings
    st.sidebar.header("Display Settings")
    show_visualizations = st.sidebar.checkbox(
        "Show visualizations", 
        value=st.session_state.get("show_visualizations", True),
        key="show_visualizations"
    )
    
    # Cache Management
    st.sidebar.header("Cache Management")
    with st.sidebar.expander("Cache Controls", expanded=False):
        from streamlit_app.document_cache import get_cache_stats, clear_cache
        
        stats = get_cache_stats()
        st.write(f"Document cache: {stats['document_count']} items")
        st.write(f"Recommendation cache: {stats['recommendation_count']} items")
        
        if st.button("Clear Cache"):
            clear_cache()
            st.success("Cache cleared successfully!")
            st.experimental_rerun()

def run_recommendations():
    """Run the recommendation engine based on current settings with caching."""
    try:
        # Get mode and parameters
        mode = st.session_state.recommendation_mode
        
        # Get the appropriate profile based on the current mode
        profile_key = None
        selected_profile = None
        if mode == "query_with_profile":
            profile_key = "profile"
        elif mode == "document_with_profile":
            profile_key = "profile_for_doc"
        elif mode == "profile_only":
            profile_key = "profile_only"
        
        if profile_key:
            selected_profile = st.session_state.get(profile_key, None)
        
        # Check if results are in cache
        cache_params = {
            "mode": mode,
            "query": st.session_state.get("query", ""),
            "document_id": st.session_state.get("document_id", ""),
            "profile_name": selected_profile,  # Use the selected profile name based on mode
            "top_k": st.session_state.get("top_k", 5),
            "filter_type": st.session_state.get("filter_type", ""),
            "use_temporal_boost": st.session_state.get("use_temporal_boost", False),
            "temporal_boost": st.session_state.get("temporal_boost", 0.3) if st.session_state.get("use_temporal_boost", False) else 0,
            "reference_date": st.session_state.get("reference_date", datetime.now()).strftime("%Y-%m-%d") if st.session_state.get("use_temporal_boost", False) else ""
        }
        
        cache_key = generate_cache_key(cache_params)
        cached_recommendations = get_recommendations_from_cache(cache_key)
        
        if cached_recommendations:
            st.success("Retrieved recommendations from cache")
            display_recommendations_with_formatting(
                cached_recommendations, 
                mode,
                show_visualizations=st.session_state.get("show_visualizations", True)
            )
            return
        
        with st.spinner("Generating recommendations..."):
            # Load environment variables
            api_key = load_environment_variables()
            
            # Initialize feature processor
            feature_processor = FeatureProcessor()
            
            # Create appropriate recommender based on whether a profile is being used
            if mode in ["query_with_profile", "document_with_profile", "profile_only"]:
                # Use PersonalizedRecommender if a profile is involved
                logger.info(f"Using PersonalizedRecommender with profile: {selected_profile}")
                try:
                    recommender = PersonalizedRecommender(
                        api_key=api_key,
                        index_name=PINECONE["index_name"],
                        embedder_model=EMBEDDER["model_name"],
                        feature_processor=feature_processor
                    )
                except Exception as e:
                    st.error(f"Failed to initialize recommender: {str(e)}")
                    logger.error(f"Recommender initialization error: {str(e)}")
                    return []
                
                if selected_profile:
                    st.write(f"Using profile: {selected_profile}")
                    profile_data, user_id = load_profile_data(selected_profile)
                else:
                    st.error(f"No profile selected for {mode} mode. Please select a profile and try again.")
                    return []
                
                # Extract profile information
                profile = profile_data.get('profile', {})
                
                # Set up client preferences from categorical preferences
                client_preferences = None
                if 'categorical_preferences' in profile:
                    client_preferences = profile['categorical_preferences']
                    logger.info(f"Loaded client preferences with {len(client_preferences)} categories")
                
                # Add historical documents if available
                if 'historical_documents' in profile and profile['historical_documents']:
                    historical_docs = profile['historical_documents']
                    logger.info(f"Adding {len(historical_docs)} historical documents to user profile")
                    for doc_id in historical_docs:
                        try:
                            recommender.add_historical_document(user_id, doc_id)
                            logger.info(f"Added historical document {doc_id} to user profile")
                        except Exception as e:
                            logger.warning(f"Failed to add historical document {doc_id}: {str(e)}")
                
                # Create expert profile if available
                if 'expert_profile' in profile and 'description' in profile['expert_profile']:
                    expert_description = profile['expert_profile']['description']
                    if expert_description:
                        try:
                            logger.info(f"Creating expert profile for user {user_id}")
                            recommender.create_expert_profile(user_id, expert_description)
                            logger.info(f"Created expert profile for user {user_id}")
                        except Exception as e:
                            logger.warning(f"Failed to create expert profile: {str(e)}")
            else:
                # For query-only or document-only mode, use standard PineconeRecommender
                logger.info("Using standard PineconeRecommender")
                recommender = PineconeRecommender(
                    api_key=api_key,
                    index_name=PINECONE["index_name"],
                    embedder_model=EMBEDDER["model_name"],
                    feature_processor=feature_processor
                )
            
            # Create filter if document type is specified
            filter_dict = None
            if st.session_state.filter_type:
                filter_dict = {"document_type": st.session_state.filter_type}
            
            # Get recommendations based on mode
            if mode == "query_only":
                # Query-only mode
                recommendations = recommender.get_recommendations(
                    query_text=st.session_state.query,
                    top_k=st.session_state.top_k,
                    filter=filter_dict,
                    temporal_boost=st.session_state.temporal_boost if st.session_state.use_temporal_boost else None,
                    reference_date=st.session_state.reference_date.strftime("%Y-%m-%d") if st.session_state.use_temporal_boost else None
                )
                
            elif mode == "query_with_profile":
                # Query with profile mode
                # Use active_profile if set, otherwise use the loaded user_id
                active_profile = st.session_state.get("active_profile", user_id)
                recommendations = recommender.get_personalized_recommendations(
                    user_id=active_profile,
                    query_text=st.session_state.query,
                    top_k=st.session_state.top_k,
                    filter=filter_dict
                )
                
            elif mode == "document_only":
                # Document ID-only mode
                recommendations = recommender.get_recommendations_by_id(
                    document_id=st.session_state.document_id,
                    top_k=st.session_state.top_k,
                    filter=filter_dict,
                    temporal_boost=st.session_state.temporal_boost if st.session_state.use_temporal_boost else None,
                    reference_date=st.session_state.reference_date.strftime("%Y-%m-%d") if st.session_state.use_temporal_boost else None
                )
                
            elif mode == "document_with_profile":
                # Document ID with profile mode
                recommendations = recommender.get_recommendations_by_id(
                    document_id=st.session_state.document_id,
                    top_k=st.session_state.top_k,
                    filter=filter_dict,
                    client_preferences=client_preferences if client_preferences else None
                )
                
            elif mode == "profile_only":
                # Profile-only mode (derive query from profile)
                # Generate a query from the profile data
                profile_query = "relevant documents"
                
                # First try to use expert profile description
                if 'expert_profile' in profile and 'description' in profile['expert_profile']:
                    # Use the first 20 words of the expert description as a query
                    words = profile['expert_profile']['description'].split()[:20]
                    if words:
                        profile_query = " ".join(words)
                        logger.info(f"Generated query from expert profile: '{profile_query[:50]}...'")
                
                # If no expert profile or fallback, use top categorical preferences
                if profile_query == "relevant documents" and 'categorical_preferences' in profile:
                    client_preferences = profile['categorical_preferences']
                    top_categories = sorted(client_preferences.items(), key=lambda x: x[1], reverse=True)[:3]
                    if top_categories:
                        profile_query = " ".join([cat for cat, _ in top_categories])
                        logger.info(f"Generated query from top categories: '{profile_query}'")
                
                # Display the generated query
                st.info(f"Generated query from profile: '{profile_query[:100]}...'")
                
                # Use active_profile if set, otherwise use the loaded user_id
                active_profile = st.session_state.get("active_profile", user_id)
                recommendations = recommender.get_personalized_recommendations(
                    user_id=active_profile,
                    query_text=profile_query,
                    top_k=st.session_state.top_k,
                    filter=filter_dict
                )
            
            # Cache recommendations
            if recommendations:
                add_recommendations_to_cache(cache_key, recommendations)
                logger.info(f"Added {len(recommendations)} recommendations to cache with key: {cache_key}")
            
            # Display recommendations
            display_recommendations_with_formatting(
                recommendations, 
                mode,
                show_visualizations=st.session_state.get("show_visualizations", True)
            )
            
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        logger.error(f"Error in run_recommendations: {str(e)}", exc_info=True)

def main():
    """Main application."""
    # Draw header with logo and title
    draw_header()
    
    # Initialize session state
    if "recommendation_mode" not in st.session_state:
        st.session_state.recommendation_mode = "query_only"
    
    # Initialize other session state variables to prevent 'has no attribute' errors
    if "profile" not in st.session_state:
        st.session_state.profile = None
    
    if "active_profile" not in st.session_state:
        st.session_state.active_profile = None
    
    # Draw sidebar with configuration
    draw_sidebar()
    
    # Create three tabs for different recommendation modes
    tab1, tab2, tab3 = st.tabs([
        "📝 Query-based", 
        "📄 Document ID-based", 
        "👤 Profile-based"
    ])
    
    with tab1:
        st.header("Query-based Recommendations")
        
        # Query input
        query = st.text_area(
            "Enter your query", 
            height=100,
            value=st.session_state.get("query", ""),
            key="query",
            placeholder="Enter a legal question or topic to find relevant EU documents..."
        )
        
        # Use profile option
        use_profile = st.checkbox(
            "Use client profile", 
            value=st.session_state.get("use_profile_with_query", False),
            key="use_profile_with_query",
            help="Enable to personalize recommendations based on a client profile"
        )
        
        if use_profile:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                available_profiles = get_available_profiles()
                default_index = 0
                if "profile" in st.session_state and st.session_state.profile in available_profiles:
                    default_index = available_profiles.index(st.session_state.profile)
                
                profile = st.selectbox(
                    "Client Profile (Optional)",
                    available_profiles,
                    index=default_index,
                    key="profile"
                )
                
                # We don't need to set session state manually, the selectbox does it automatically
                # with the key="profile" parameter
                
            with col2:
                st.write("")
                st.write("")
                if st.button("View Profile Details", key="view_profile_details"):
                    if st.session_state.profile:
                        profile_data, _ = load_profile_data(st.session_state.profile)
                        display_profile_info(profile_data)
                    else:
                        st.error("No profile selected.")
            
            if st.button("Get Query + Profile Recommendations", type="primary"):
                st.session_state.recommendation_mode = "query_with_profile"
                run_recommendations()
        else:
            if st.button("Get Query Recommendations", type="primary"):
                st.session_state.recommendation_mode = "query_only"
                run_recommendations()
    
    with tab2:
        st.header("Document ID-based Recommendations")
        
        # Document ID input
        col1, col2 = st.columns([2, 3])
        
        with col1:
            document_id = st.selectbox(
                "Select document ID",
                get_sample_document_ids(),
                index=0,
                key="document_id"
            )
        
        with col2:
            # Alternative: Free text input for document ID
            custom_id = st.text_input(
                "Or enter custom document ID (CELEX number)",
                value="",
                key="custom_document_id",
                placeholder="e.g., 32023R0456"
            )
            
            if custom_id:
                st.session_state.document_id = custom_id
                
        # Document preview - show metadata for the selected document
        current_doc_id = st.session_state.document_id
        if current_doc_id:
            st.subheader("Document Preview")
            
            # Create a spinner while fetching document metadata
            with st.spinner(f"Loading document metadata for {current_doc_id}..."):
                # Get document data from cache or Pinecone
                try:
                    # Try to get from cache first
                    document = get_document_from_cache(current_doc_id)
                    
                    if not document:
                        # If not in cache, try to get from SQLite database
                        document = get_document_from_database(current_doc_id)
                        
                        # Cache the document
                        if document:
                            add_document_to_cache(current_doc_id, document)
                    
                    if document:
                        # Extract metadata
                        metadata = document.get('metadata', {})
                        if not metadata and isinstance(document, dict):
                            # If no metadata key but document is a dict, use it directly
                            metadata = document
                            
                        title = metadata.get('title', 'No title available')
                        doc_type = metadata.get('document_type', 'Unknown')
                        
                        # Get date information with fallbacks
                        date = metadata.get('date', metadata.get('publication_date', metadata.get('adoption_date', 'N/A')))
                        if date == 'N/A' or not date:
                            date = metadata.get('year', 'N/A')
                        
                        celex = metadata.get('celex_number', current_doc_id)
                        subject_matters = metadata.get('subject_matters', [])
                        
                        # Get EUR-Lex URL
                        eurlex_url = metadata.get('url', '')
                        if not eurlex_url and celex != 'N/A':
                            eurlex_url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
                        
                        # Get summary if available
                        summary = metadata.get('summary', document.get('summary', ''))
                        if not summary and 'text' in metadata and metadata['text']:
                            # Use first 300 chars of text as preview
                            summary = metadata['text'][:300] + '...' if len(metadata['text']) > 300 else metadata['text']
                        
                        # Create a formatted display of the document info
                        st.info(f"Found document: {title}")
                        
                        # Display document info in a nice box
                        with st.container():
                            st.markdown(f"""<div class='document-preview'>
                                <h4>{title}</h4>
                                <p><strong>Document ID:</strong> {current_doc_id}</p>
                                <p><strong>CELEX:</strong> {celex}</p>
                                <p><strong>Type:</strong> {doc_type} | <strong>Date:</strong> {date}</p>
                                <p><strong>Subject Matters:</strong> {', '.join(subject_matters[:5])}{' ...' if len(subject_matters) > 5 else ''}</p>
                                {f'<p><a href="{eurlex_url}" target="_blank">View on EUR-Lex</a></p>' if eurlex_url else ''}
                                {f'<div class="summary-box"><strong>Summary:</strong><br>{summary}</div>' if summary else ''}
                            </div>""", unsafe_allow_html=True)
                            
                    else:
                        st.warning(f"Could not retrieve metadata for document {current_doc_id}. Check if the ID is correct.")
                        
                except Exception as e:
                    st.error(f"Error retrieving document metadata: {str(e)}")
                    logger.error(f"Error in document preview: {str(e)}", exc_info=True)
        
        # Use profile option
        use_profile_with_doc = st.checkbox(
            "Use client profile", 
            value=st.session_state.get("use_profile_with_doc", False),
            key="use_profile_with_doc",
            help="Enable to personalize recommendations based on a client profile"
        )
        
        if use_profile_with_doc:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                available_profiles = get_available_profiles()
                default_index = 0
                if "profile_for_doc" in st.session_state and st.session_state.profile_for_doc in available_profiles:
                    default_index = available_profiles.index(st.session_state.profile_for_doc)
                
                profile_for_doc = st.selectbox(
                    "Client Profile (Optional)",
                    available_profiles,
                    index=default_index,
                    key="profile_for_doc"
                )
                
                # We don't need to set session state manually, the selectbox does it automatically
                # with the key="profile_for_doc" parameter
                
            with col2:
                st.write("")
                st.write("")
                if st.button("View Profile Details", key="view_profile_doc_details"):
                    if st.session_state.profile_for_doc:
                        profile_data, _ = load_profile_data(st.session_state.profile_for_doc)
                        display_profile_info(profile_data)
                    else:
                        st.error("No profile selected.")
            
            # Store the profile name but with a different key to avoid widget conflicts
            st.session_state.active_profile = st.session_state.profile_for_doc
            
            if st.button("Get Document + Profile Recommendations", type="primary"):
                st.session_state.recommendation_mode = "document_with_profile"
                run_recommendations()
        else:
            if st.button("Get Document Recommendations", type="primary"):
                st.session_state.recommendation_mode = "document_only"
                run_recommendations()
    
    with tab3:
        st.header("Profile-based Recommendations")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            available_profiles = get_available_profiles()
            default_index = 0
            if "profile_only" in st.session_state and st.session_state.profile_only in available_profiles:
                default_index = available_profiles.index(st.session_state.profile_only)
            
            profile_only = st.selectbox(
                "Client Profile",
                available_profiles,
                index=default_index,
                key="profile_only"
            )
            
            # We don't need to set session state manually, the selectbox does it automatically
            # with the key="profile_only" parameter
        
        with col2:
            st.write("")
            st.write("")
            if st.button("View Profile Details", key="view_profile_only_details"):
                if st.session_state.profile_only:
                    profile_data, _ = load_profile_data(st.session_state.profile_only)
                    display_profile_info(profile_data)
                else:
                    st.error("No profile selected.")
        
        # Store the profile name but with a different key to avoid widget conflicts
        st.session_state.active_profile = st.session_state.profile_only
        
        # Add explanation about profile-only mode
        st.info("""
        Profile-only mode generates recommendations based solely on the client profile without requiring a query or document ID.
        The system will analyze the profile's expert description, historical documents, and categorical preferences to find relevant documents.
        """)
        
        if st.button("Get Profile-Only Recommendations", type="primary"):
            st.session_state.recommendation_mode = "profile_only"
            run_recommendations()
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <p>EU Legal Recommender System - Built with Streamlit</p>
    <p>© 2025 - Uses Legal-BERT and Pinecone for document recommendations</p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
