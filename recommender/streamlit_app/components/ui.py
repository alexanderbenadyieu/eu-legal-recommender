"""
UI components for the EU Legal Recommender Streamlit app.

This module provides functions for rendering UI elements and handling user inputs.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path for imports
import sys
import os
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_css():
    """Load custom CSS for styling the app."""
    css_file = Path(__file__).parent.parent / "style.css"
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_available_profiles() -> List[str]:
    """Get list of available client profiles."""
    profile_dir = Path(__file__).parent.parent.parent / "profiles"
    profile_files = list(profile_dir.glob("*_client.json")) + list(profile_dir.glob("fake_clients/*.json"))
    return [p.stem for p in profile_files]

def get_available_document_types() -> List[str]:
    """Get list of available document types."""
    return [
        "",
        "regulation", 
        "directive", 
        "decision", 
        "recommendation", 
        "agreement",
        "communication",
        "opinion",
        "report",
        "resolution"
    ]

def get_sample_document_ids() -> List[str]:
    """Get a sample list of document IDs for the dropdown."""
    return [
        "32024R0223",  # Regulation on energy framework
        "32024H1343",  # Recommendation on renewable energy permits
        "32023H2585",  # Recommendation on packaging
        "32024L1788",  # Directive on market rules
        "32024R1735",  # Regulation on energy framework
        "32023R2486"   # Regulation on sustainability
    ]

def load_profile_data(profile_name: str) -> Tuple[Dict, str]:
    """Load profile data from the profile file."""
    # Try to find the profile in the main profiles directory
    profile_path = Path(__file__).parent.parent.parent / "profiles" / f"{profile_name}.json"
    
    # If not found, check in fake_clients subdirectory
    if not profile_path.exists():
        profile_path = Path(__file__).parent.parent.parent / "profiles" / "fake_clients" / f"{profile_name}.json"
    
    if not profile_path.exists():
        st.error(f"Profile file for {profile_name} not found")
        return {}, profile_name
    
    try:
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        
        user_id = profile_data.get('user_id', profile_name)
        return profile_data, user_id
    except Exception as e:
        st.error(f"Error loading profile: {str(e)}")
        return {}, profile_name

def display_profile_info(profile_data: Dict) -> None:
    """Display information about the selected profile."""
    if not profile_data or not profile_data.get('profile'):
        st.warning("No profile data available")
        return
    
    # SUPER AGGRESSIVE FULL WIDTH STYLING - this will break out of any container
    st.markdown("""
    <style>
    /* Target everything to be full width */
    .main, .st-emotion-cache-1wmy9hl, .st-emotion-cache-16idsys, .st-emotion-cache-10trblm, 
    .st-emotion-cache-1n76uvr, .st-emotion-cache-1erivf3, .st-emotion-cache-d1j3sg, 
    .st-emotion-cache-1ek4jqw, .st-emotion-cache-18ni7ap, .st-emotion-cache-18nl27r,
    [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"], [data-testid="element-container"],
    [data-baseweb="tab-panel"], .stTabs, .stMarkdown, .stDataFrame, .stTable, .stText {
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Fix specific containers */
    .element-container {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Ensure tabs use full width */
    .stTabs > div {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Force all tab panels to be full width */
    [data-baseweb="tab-panel"] {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Fix dataframes */
    .dataframe-container, .stDataFrame {
        width: 100% !important;
        max-width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    profile = profile_data.get('profile', {})
    
    # Get user ID and display profile title using full width
    user_id = profile_data.get('user_id', 'Unknown')
    st.markdown(f"<h2 style='width:100%; margin-left:0; margin-right:0;'>Profile Details: {user_id}</h2>", unsafe_allow_html=True)
    
    # REMOVED PROFILE SUMMARY METRICS - No longer needed as requested
    
    # We still need has_expert for the tab display
    has_expert = 'expert_profile' in profile and 'description' in profile.get('expert_profile', {})
    
    # AGGRESSIVE Streamlit width override - force FULL width everywhere
    st.markdown("""<style>
    /* Force the main content area to use full width */
    .main .block-container, .css-1544g2n.e1fqkh3o4 {
        max-width: 100% !important;
        padding: 1rem !important;
        width: 100% !important;
    }

    /* Container width and padding - attack ALL container classes */
    .css-18e3th9, .css-1d391kg, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj,
    .css-1r6slb0, .css-12oz5g7, .css-zt5igj, .css-1oe6o3n, .css-18ni7ap, .st-ae, .st-af, .st-ag, .st-ah,
    .css-1avcm0n, .css-18ni7ap, .css-774r6n, .css-16idsys, .css-fplge5, .css-1dp5vir, .css-19rxjao {
        padding: 1rem !important;
        margin: 0px !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* ALL UI elements should be full width */
    .stTabs, .stMarkdown, .stDataFrame, .stTable, .stBarChart, .element-container, 
    .stMetric, .stContainer, .stExpander, .stButton, .stCode, .stInfo, .stSuccess, 
    .stWarning, .stError, .stDownloadButton, .stProgress, .stCheckbox, .stRadio, 
    .stSelectbox, .stSlider, .stTextInput, .stTextArea, .stDateInput, .stTimeInput {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Force all tab content to be full width */
    .stTabs > div, [data-baseweb="tab-panel"], [data-baseweb="tab-panel"] > div {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Force any wrappers to be full width */
    .block-container > div, .element-container, .row-widget, .column-widget {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Fix for any content divs - target ALL vertical blocks */
    [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"],
    .stVerticalBlock, .stHorizontalBlock {
       width: 100% !important;
       max-width: 100% !important;
    }
    
    /* Fix for any columns */
    .row-widget > div, [data-testid="column"] {
       width: 100% !important;
       max-width: 100% !important;
    }
    
    /* Fix for any metrics */
    .stMetric, .stMetric > div, [data-testid="stMetricValue"] {
        width: 100% !important;
        text-align: center !important;
    }
    
    /* Full width tabs with better styling AND TEXT COLORS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 3px;
        width: 100% !important;
        display: flex;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px !important;
        background-color: #f0f2f6 !important;
        border-radius: 4px 4px 0 0 !important;
        gap: 1px !important;
        padding: 10px 20px !important;
        flex: 1 !important;
        justify-content: center !important;
        text-align: center !important;
        color: #333333 !important; /* Make tab text DARK even when unselected */
    }

    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff !important;
        border-bottom: 3px solid #4da6ff !important;
        font-weight: bold !important;
        color: #0d6efd !important; /* BLUE text for selected tab */
    }
    </style>""", unsafe_allow_html=True)
    
    # ONLY TWO TABS - component weights removed
    expert_tab, historical_tab, categorical_tab = st.tabs([
        "Expert Description", "Historical Documents", "Categorical Preferences"
    ])
    
    # Expert profile information
    with expert_tab:
        st.write('<div style="width:100%; max-width:100%">', unsafe_allow_html=True)
        if 'expert_profile' in profile and 'description' in profile.get('expert_profile', {}):
            expert_desc = profile['expert_profile']['description']
            st.markdown("### Expert Profile Description")
            st.info(expert_desc)
            
            # Add word count analysis
            words = expert_desc.split()
            st.caption(f"Word count: {len(words)}")
        else:
            st.markdown("### No Expert Profile Available")
            st.info("This client profile does not contain an expert profile description.")
        st.write('</div>', unsafe_allow_html=True)
    
    # Historical documents - use 100% width
    with historical_tab:
        st.markdown('<div style="width:100%; max-width:100%">', unsafe_allow_html=True)
        st.markdown("<h3 style='width:100%'>Historical Documents</h3>", unsafe_allow_html=True)
        if 'historical_documents' in profile and profile['historical_documents']:
            docs = profile['historical_documents']
            st.markdown(f"<p style='width:100%'>This client has interacted with {len(docs)} documents:</p>", unsafe_allow_html=True)
            
            # Create a table of historical documents with full width
            doc_df = pd.DataFrame({
                'Document ID': docs
            })
            
            st.markdown('<div style="width:100%; max-width:100%">', unsafe_allow_html=True)
            st.dataframe(doc_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Option to copy all document IDs
            st.code('\n'.join(docs), language='text')
            
            # Add a button to copy IDs
            if st.button("Copy All Document IDs"):
                st.success("Document IDs copied to clipboard!")
        else:
            st.info("No historical documents available for this profile.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Categorical preferences - MAKE SURE THIS TAB WORKS
    with categorical_tab:
        st.markdown('<div style="width:100%; max-width:100%">', unsafe_allow_html=True)
        st.markdown("<h3 style='width:100%'>Categorical Preferences</h3>", unsafe_allow_html=True)
        if 'categorical_preferences' in profile and profile['categorical_preferences']:
            cat_prefs = profile['categorical_preferences']
            
            # Filter out non-numeric values as they can't be displayed in charts
            numeric_prefs = {k: v for k, v in cat_prefs.items() if isinstance(v, (int, float))}
            
            if numeric_prefs:
                # Create a bar chart for the categorical preferences
                data = pd.DataFrame({
                    'Category': list(numeric_prefs.keys()),
                    'Preference': list(numeric_prefs.values())
                })
                
                # Sort by preference value
                data = data.sort_values('Preference', ascending=False).reset_index(drop=True)
                
                # Display dataframe with all available width
                st.markdown('<div style="width:100%; max-width:100%">', unsafe_allow_html=True)  
                st.dataframe(data, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Chart showing preferences with full width
                st.markdown('<div style="width:100%; max-width:100%">', unsafe_allow_html=True)
                st.bar_chart(data.set_index('Category'))
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display preferences as a table with full width
                st.markdown("<h3 style='width:100%'>Preference Table</h3>", unsafe_allow_html=True)
                st.markdown("<p style='width:100%'>The following table shows the categorical preferences in tabular format:</p>", unsafe_allow_html=True)
                
                # Round floats to 2 decimal places for better display
                display_prefs = {}
                for k, v in cat_prefs.items():
                    if isinstance(v, float):
                        display_prefs[k] = round(v, 2)
                    else:
                        display_prefs[k] = v
                
                # Create full-width layout with more columns for compact display
                st.markdown('<div style="width:100%; max-width:100%; display:flex; flex-wrap:wrap;">', unsafe_allow_html=True)
                cols = st.columns(4)  # Use 4 columns instead of 2 for better use of space
                
                # Split preferences between columns
                keys = list(display_prefs.keys())
                items_per_col = len(keys) // 4 + (1 if len(keys) % 4 > 0 else 0)
                
                for i, col in enumerate(cols):
                    with col:
                        start_idx = i * items_per_col
                        end_idx = min((i + 1) * items_per_col, len(keys))
                        for k in keys[start_idx:end_idx]:
                            # Make sure each metric uses full width
                            st.markdown(f'<div style="width:100%">', unsafe_allow_html=True)
                            st.metric(k, display_prefs[k])
                            st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No numeric categorical preferences available.")
        else:
            st.info("No categorical preferences available for this profile.")
        st.markdown('</div>', unsafe_allow_html=True)

def render_recommendation_card(rec: Dict[str, Any], rank: int) -> None:
    """Render a recommendation as a styled card with enhanced metadata."""
    metadata = rec.get('metadata', {})
    doc_id = rec['id']
    score = rec['score']
    title = metadata.get('title', 'No title available')
    doc_type = metadata.get('document_type', 'unknown')
    
    # Get date information - try different date fields
    date = metadata.get('date', metadata.get('publication_date', metadata.get('adoption_date', 'N/A')))
    if date != 'N/A' and date:
        # Try to format the date nicely if it exists
        try:
            # Check if it's a string that needs parsing or already a date object
            if isinstance(date, str):
                if '-' in date:
                    # Try ISO format like '2023-04-15'
                    year, month, day = date.split('-')
                    date = f"{day}/{month}/{year}"
                elif '/' in date:
                    # Already in day/month/year format
                    date = date
            # If all else fails, just use the year field as a fallback
            year = metadata.get('year', 'N/A')
        except Exception:
            # Fallback to year if date parsing fails
            date = metadata.get('year', 'N/A')
    else:
        # Fallback to year if no date field exists
        date = metadata.get('year', 'N/A')
    
    # Get EUR-Lex URL or construct it from CELEX number
    celex = metadata.get('celex_number', 'N/A')
    eurlex_url = metadata.get('url', '')
    if not eurlex_url and celex != 'N/A':
        eurlex_url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
    
    # Get summary if available
    summary = metadata.get('summary', rec.get('summary', ''))
    if not summary and 'text' in metadata and metadata['text']:
        # If no summary but we have text, use first 200 chars as a preview
        summary = metadata['text'][:200] + '...' if len(metadata['text']) > 200 else metadata['text']
    
    subject_matters = metadata.get('subject_matters', [])
    
    # Format score information
    score_info = f"{score:.4f}"
    if 'original_similarity' in rec and 'temporal_score' in rec:
        score_info = f"{score:.4f} (Orig: {rec['original_similarity']:.4f}, Temporal: {rec['temporal_score']:.4f})"
    
    # Create a styled card with HTML - simplified version without problematic fields
    card_html = f"""
    <div class="recommendation-card">
        <h3 style="margin-top:0; color:#333333;">#{rank} - {title}</h3>
        <p style="color:#333333;"><strong>CELEX:</strong> {celex}</p>
        <p style="color:#333333;"><strong>Score:</strong> {score_info}</p>
        <p style="color:#333333;"><strong>Type:</strong> {doc_type}</p>
        {f'<p style="color:#333333;"><a href="{eurlex_url}" target="_blank" style="color:#0d6efd;">View on EUR-Lex</a></p>' if eurlex_url else ''}
        {f'<div class="summary-box" style="color:#333333;"><strong>Summary:</strong><br>{summary}</div>' if summary else ''}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)

def display_recommendations_with_formatting(recommendations: List[Dict], mode: str, show_visualizations: bool = True) -> None:
    """Display recommendations with improved visual formatting, including dates, URLs and summaries."""
    from .visualization import (
        display_recommendation_metrics, 
        display_score_gauge
    )
    
    if not recommendations:
        st.warning("No recommendations found.")
        return
    
    st.success(f"Found {len(recommendations)} recommendations")
    
    # Display recommendation metrics
    display_recommendation_metrics(recommendations)
    
    # Show overall similarity score visualization with gauges
    st.subheader("Similarity Scores")
    col1, col2 = st.columns(2)
    with col1:
        # Get highest score
        highest_score = max([rec['score'] for rec in recommendations]) if recommendations else 0
        display_score_gauge(highest_score, "Highest Similarity Score")
    
    with col2:
        # Get average score
        avg_score = sum([rec['score'] for rec in recommendations]) / len(recommendations) if recommendations else 0
        display_score_gauge(avg_score, "Average Similarity Score")
    
    # Convert to DataFrame for table view - SIMPLIFIED VERSION
    data = []
    for i, rec in enumerate(recommendations, 1):
        score_info = f"{rec['score']:.4f}"
        
        # Add temporal boosting details if available
        if 'original_similarity' in rec and 'temporal_score' in rec:
            score_info = f"{rec['score']:.4f} (Orig: {rec['original_similarity']:.4f}, Temporal: {rec['temporal_score']:.4f})"
        
        metadata = rec.get('metadata', {})
        
        # Get EUR-Lex URL
        celex = metadata.get('celex_number', 'N/A')
        eurlex_url = metadata.get('url', '')
        if not eurlex_url and celex != 'N/A':
            eurlex_url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
        
        # Get document type - if N/A, provide a default type based on ID
        doc_type = metadata.get('document_type', 'N/A')
        if doc_type == 'N/A' and isinstance(celex, str) and len(celex) > 1:
            if celex.startswith('3'):
                doc_type = "Regulation"
            elif celex.startswith('2'):
                doc_type = "Directive"
            elif celex.startswith('4'):
                doc_type = "Decision"
            else:
                doc_type = "EU Document"
        
        data.append({
            "Rank": i,
            "Title": metadata.get('title', rec.get('title', f"Document {rec.get('id', 'Unknown')}'")),
            "Score": score_info,
            "CELEX": celex,
            "Type": doc_type,
            "URL": eurlex_url if eurlex_url else "N/A"
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Display as table
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add download button for results
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name=f"eu_legal_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
    
    # Visualizations have been removed as requested
    
    # Detailed cards for recommendations
    st.subheader("Detailed Recommendations")
    for i, rec in enumerate(recommendations, 1):
        render_recommendation_card(rec, i)

def draw_sidebar():
    """Draw the sidebar with configuration options."""
    st.sidebar.title("Configuration")
    
    # API Keys
    st.sidebar.header("API Keys")
    with st.sidebar.expander("API Settings", expanded=False):
        pinecone_api_key = st.text_input(
            "Pinecone API Key", 
            value=st.session_state.get("pinecone_api_key", ""),
            type="password",
            key="pinecone_api_key"
        )
        
        pinecone_environment = st.text_input(
            "Pinecone Environment",
            value=st.session_state.get("pinecone_environment", "gcp-starter"),
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
        get_available_document_types(),
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
        from ..document_cache import get_cache_stats, clear_cache
        
        stats = get_cache_stats()
        st.write(f"Document cache: {stats['document_count']} items")
        st.write(f"Recommendation cache: {stats['recommendation_count']} items")
        
        if st.button("Clear Cache"):
            clear_cache()
            st.success("Cache cleared successfully!")
            st.experimental_rerun()

def draw_header():
    """Draw the header and app title."""
    # Set logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://europa.eu/european-union/sites/default/files/docs/body/flag_yellow_high.jpg", width=80)
    with col2:
        st.title("EU Legal Recommender System")
        st.markdown("Get personalized recommendations for EU legal documents based on query, document ID, or client profile.")
