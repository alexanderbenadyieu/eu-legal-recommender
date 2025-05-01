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
    
    profile = profile_data.get('profile', {})
    
    # Get user ID and display profile title
    user_id = profile_data.get('user_id', 'Unknown')
    st.markdown(f"## Profile Details: {user_id}")
    
    # Profile summary metrics at the top
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    # Count historical documents
    historical_count = len(profile.get('historical_documents', []))
    
    # Count categorical preferences
    cat_prefs = profile.get('categorical_preferences', {})
    cat_count = sum(1 for v in cat_prefs.values() if isinstance(v, (int, float)))
    
    # Check if expert profile exists
    has_expert = 'expert_profile' in profile and 'description' in profile['expert_profile']
    
    with metrics_col1:
        st.metric("Historical Documents", historical_count)
    
    with metrics_col2:
        st.metric("Categorical Preferences", cat_count)
    
    with metrics_col3:
        st.metric("Expert Profile", "Yes" if has_expert else "No")
    
    # Use the full container width for the profile display
    st.markdown("""<style>
    .element-container, .stTabs, .stMarkdown, .stDataFrame, .stTable, .stBar {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    [data-testid="StyledFullScreenFrame"] {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    [data-testid="block-container"] {
        max-width: 100% !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
    }
    
    .stPlotlyChart, .stChart {
        width: 100% !important;
        height: 500px !important;
    }
    </style>""", unsafe_allow_html=True)
    
    # Create tabs for different profile sections - using full width
    expert_tab, historical_tab, categorical_tab, weights_tab = st.tabs([
        "Expert Profile", "Historical Documents", "Categorical Preferences", "Component Weights"
    ])
    
    # Expert profile information
    with expert_tab:
        if has_expert:
            expert_desc = profile['expert_profile']['description']
            st.markdown("### Expert Profile Description")
            st.info(expert_desc)
            
            # Add word count analysis
            words = expert_desc.split()
            st.caption(f"Word count: {len(words)}")
            
            # Most common words (simple analysis)
            if len(words) > 10:
                word_counts = {}
                for word in words:
                    word = word.lower().strip('.,;:()[]{}"\'\'').strip()
                    if len(word) > 3:  # Skip short words
                        word_counts[word] = word_counts.get(word, 0) + 1
                
                # Display top words
                top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                st.markdown("#### Most Common Terms")
                
                top_word_df = pd.DataFrame({
                    'Term': [w[0] for w in top_words],
                    'Frequency': [w[1] for w in top_words]
                })
                
                # Use a larger chart height and full width
                st.bar_chart(top_word_df.set_index('Term'), height=500, use_container_width=True)
        else:
            st.markdown("### No Expert Profile Available")
            st.info("This client profile does not contain an expert profile description.")
    
    # Historical documents
    with historical_tab:
        st.markdown("### Historical Documents")
        if 'historical_documents' in profile and profile['historical_documents']:
            docs = profile['historical_documents']
            st.markdown(f"This client has interacted with {len(docs)} documents:")
            
            # Create a table of historical documents
            doc_df = pd.DataFrame({
                'Document ID': docs
            })
            
            st.dataframe(doc_df, use_container_width=True)
            
            # Option to copy all document IDs
            st.code('\n'.join(docs), language='text')
            
            # Add a button to copy IDs
            if st.button("Copy All Document IDs"):
                st.success("Document IDs copied to clipboard!")
        else:
            st.info("No historical documents available for this client.")
    
    # Categorical preferences
    with categorical_tab:
        st.markdown("### Categorical Preferences")
        if 'categorical_preferences' in profile and profile['categorical_preferences']:
            categories = profile['categorical_preferences']
            st.markdown(f"This client has {len(categories)} categorical preferences:")
            
            # Create lists for keys and values, ensuring values are numeric
            cat_keys = []
            cat_values = []
            other_cats = {}
            
            for k, v in categories.items():
                # Only include if value is a number, not a dict or other structure
                if isinstance(v, (int, float)):
                    cat_keys.append(k)
                    cat_values.append(v)
                else:
                    other_cats[k] = v
            
            # Only create DataFrame if we have numeric values
            if cat_keys and cat_values:
                df = pd.DataFrame({
                    'Category': cat_keys,
                    'Weight': cat_values
                })
                
                # Now sort and display
                df = df.sort_values('Weight', ascending=False)
                
                # Display as table and chart
                st.dataframe(df, use_container_width=True)
                # Use a larger chart height and full width
                st.bar_chart(df.set_index('Category'), height=600, use_container_width=True)  # Increased height
            
            # Display non-numeric categories
            if other_cats:
                st.markdown("#### Additional categorical preferences:")
                for k, v in other_cats.items():
                    if isinstance(v, dict):
                        # For dictionary values, display the keys
                        st.markdown(f"**{k}**: {', '.join(v.keys())}")
                    else:
                        # For any other type, display as text
                        st.markdown(f"**{k}**: {str(v)}")
        else:
            st.info("No categorical preferences available for this client.")
    
    # Component weights
    with weights_tab:
        st.markdown("### Component Weights")
        if 'component_weights' in profile and profile['component_weights']:
            weights = profile['component_weights']
            st.markdown(f"This client has {len(weights)} component weight settings:")
            
            # Create lists for keys and values, ensuring values are numeric
            comp_keys = []
            comp_values = []
            other_weights = {}
            
            for k, v in weights.items():
                # Only include if value is a number, not a dict or other structure
                if isinstance(v, (int, float)):
                    comp_keys.append(k)
                    comp_values.append(v)
                else:
                    other_weights[k] = v
            
            # Only create DataFrame if we have numeric values
            if comp_keys and comp_values:
                weight_data = pd.DataFrame({
                    'Component': comp_keys,
                    'Weight': comp_values
                })
                
                # Display as table and chart
                st.dataframe(weight_data, use_container_width=True)
                
                # Use plotly for a more attractive chart
                import plotly.express as px
                fig = px.pie(weight_data, values='Weight', names='Component', 
                             title='Component Weight Distribution',
                             color_discrete_sequence=px.colors.sequential.Blues_r)
                fig.update_traces(textinfo='percent+label', pull=[0.1, 0, 0])
                st.plotly_chart(fig, use_container_width=True)
                
                # Also show as bar chart for comparison
                # Use a larger chart height and full width
                st.bar_chart(weight_data.set_index('Component'), height=500, use_container_width=True)  # Increased height
            
            # Display non-numeric weights
            if other_weights:
                st.markdown("#### Additional component settings:")
                for k, v in other_weights.items():
                    if isinstance(v, dict):
                        # For dictionary values, display the keys
                        st.markdown(f"**{k}**: {', '.join(v.keys())}")
                    else:
                        # For any other type, display as text
                        st.markdown(f"**{k}**: {str(v)}")
        else:
            st.info("No component weights available for this client.")

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
    
    # Create a styled card with HTML - enhanced with date, URL and summary
    card_html = f"""
    <div class="recommendation-card">
        <h3 style="margin-top:0">#{rank} - {title}</h3>
        <p><strong>Document ID:</strong> {doc_id} | <strong>CELEX:</strong> {celex}</p>
        <p><strong>Score:</strong> {score_info}</p>
        <p><strong>Type:</strong> {doc_type} | <strong>Date:</strong> {date}</p>
        <p><strong>Subject Matters:</strong> {', '.join(subject_matters[:5])}{' ...' if len(subject_matters) > 5 else ''}</p>
        {f'<p><a href="{eurlex_url}" target="_blank">View on EUR-Lex</a></p>' if eurlex_url else ''}
        {f'<div class="summary-box"><strong>Summary:</strong><br>{summary}</div>' if summary else ''}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)

def display_recommendations_with_formatting(recommendations: List[Dict], mode: str, show_visualizations: bool = True) -> None:
    """Display recommendations with improved visual formatting, including dates, URLs and summaries."""
    from .visualization import (
        display_recommendation_metrics, 
        display_document_similarity_graph,
        display_year_distribution,
        display_subject_matter_chart,
        display_score_gauge
    )
    
    if not recommendations:
        st.warning("No recommendations found.")
        return
    
    st.success(f"Found {len(recommendations)} recommendations")
    
    # Display recommendation metrics
    display_recommendation_metrics(recommendations)
    
    # Show overall similarity score visualization
    col1, col2 = st.columns(2)
    with col1:
        # Get highest score
        highest_score = max([rec['score'] for rec in recommendations]) if recommendations else 0
        display_score_gauge(highest_score, "Highest Similarity Score")
    
    with col2:
        # Get average score
        avg_score = sum([rec['score'] for rec in recommendations]) / len(recommendations) if recommendations else 0
        display_score_gauge(avg_score, "Average Similarity Score")
    
    # Convert to DataFrame for table view
    data = []
    for i, rec in enumerate(recommendations, 1):
        score_info = f"{rec['score']:.4f}"
        
        # Add temporal boosting details if available
        if 'original_similarity' in rec and 'temporal_score' in rec:
            score_info = f"{rec['score']:.4f} (Orig: {rec['original_similarity']:.4f}, Temporal: {rec['temporal_score']:.4f})"
        
        metadata = rec.get('metadata', {})
        
        # Get date information - try different date fields
        date = metadata.get('date', metadata.get('publication_date', metadata.get('adoption_date', 'N/A')))
        if date == 'N/A' or not date:
            date = metadata.get('year', 'N/A')
            
        # Get EUR-Lex URL
        celex = metadata.get('celex_number', 'N/A')
        eurlex_url = metadata.get('url', '')
        if not eurlex_url and celex != 'N/A':
            eurlex_url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
            
        data.append({
            "Rank": i,
            "Document ID": rec['id'],
            "Score": score_info,
            "CELEX": celex,
            "Type": metadata.get('document_type', 'N/A'),
            "Title": metadata.get('title', 'N/A')[:100] + ('...' if len(metadata.get('title', '')) > 100 else ''),
            "Date": date,
            "URL": eurlex_url if eurlex_url else "N/A",
            "Subject Matters": ", ".join(metadata.get('subject_matters', []))[:50] + ('...' if len(", ".join(metadata.get('subject_matters', []))) > 50 else '')
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
    
    # Show visualizations if enabled
    if show_visualizations:
        st.subheader("Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Year Distribution", "Subject Matters", "Document Similarity"])
        
        with tab1:
            display_year_distribution(recommendations)
        
        with tab2:
            display_subject_matter_chart(recommendations)
            
        with tab3:
            display_document_similarity_graph(recommendations)
    
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
