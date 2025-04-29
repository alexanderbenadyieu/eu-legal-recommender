"""
Visualization components for the EU Legal Recommender Streamlit app.

This module provides functions for visualizing recommendations and other data.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Dict, Any, Optional

def render_document_type_badge(doc_type: str) -> str:
    """Generate HTML for a styled document type badge."""
    doc_type = doc_type.lower() if doc_type else "other"
    class_name = f"badge badge-{doc_type}" if doc_type in ["regulation", "directive", "decision", "recommendation"] else "badge badge-other"
    return f'<span class="{class_name}">{doc_type.capitalize()}</span>'

def display_score_gauge(score: float, title: str = "Similarity Score") -> None:
    """Display a gauge chart for the similarity score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 16, "color": "#1e3a8a"}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#1e3a8a"},
            "bar": {"color": "#3b82f6"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#e5e7eb",
            "steps": [
                {"range": [0, 0.3], "color": "#fee2e2"},
                {"range": [0.3, 0.7], "color": "#fef3c7"},
                {"range": [0.7, 1], "color": "#dcfce7"}
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]}
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=30, r=30, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1e3a8a", "family": "Arial"}
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def display_year_distribution(recommendations: List[Dict[str, Any]]) -> None:
    """Display a histogram of document years."""
    years = []
    for rec in recommendations:
        metadata = rec.get('metadata', {})
        if 'year' in metadata and metadata['year']:
            try:
                year = int(metadata['year'])
                if 1950 <= year <= datetime.now().year:
                    years.append(year)
            except (ValueError, TypeError):
                pass
    
    if not years:
        st.info("No year data available for visualization.")
        return
    
    # Create histogram
    fig = px.histogram(
        x=years,
        nbins=min(len(set(years)), 20),
        title="Distribution of Document Years",
        labels={"x": "Year", "y": "Count"},
        color_discrete_sequence=["#3b82f6"]
    )
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1e3a8a", "family": "Arial"},
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="#e5e7eb"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="#e5e7eb")
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_subject_matter_chart(recommendations: List[Dict[str, Any]], top_n: int = 10) -> None:
    """Display a bar chart of the most common subject matters."""
    subject_counts = {}
    
    for rec in recommendations:
        metadata = rec.get('metadata', {})
        subjects = metadata.get('subject_matters', [])
        
        for subject in subjects:
            if subject in subject_counts:
                subject_counts[subject] += 1
            else:
                subject_counts[subject] = 1
    
    if not subject_counts:
        st.info("No subject matter data available for visualization.")
        return
    
    # Sort by frequency and get top N
    top_subjects = sorted(
        subject_counts.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_n]
    
    subject_df = pd.DataFrame(
        top_subjects,
        columns=["Subject Matter", "Count"]
    )
    
    fig = px.bar(
        subject_df,
        x="Count",
        y="Subject Matter",
        orientation='h',
        title=f"Top {len(top_subjects)} Subject Matters",
        color_discrete_sequence=["#3b82f6"]
    )
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1e3a8a", "family": "Arial"},
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="#e5e7eb"),
        yaxis=dict(showgrid=False)
    )
    
    # Reverse y-axis to show highest count at the top
    fig.update_yaxes(autorange="reversed")
    
    st.plotly_chart(fig, use_container_width=True)

def display_document_similarity_graph(recommendations: List[Dict[str, Any]]) -> None:
    """Display a network graph of document similarity."""
    if len(recommendations) < 3:
        st.info("Need at least 3 documents to create a similarity graph.")
        return
    
    # Create nodes for each document
    nodes = []
    for i, rec in enumerate(recommendations):
        doc_id = rec['id']
        score = rec['score']
        title = rec.get('metadata', {}).get('title', doc_id)
        doc_type = rec.get('metadata', {}).get('document_type', 'other')
        
        nodes.append({
            'id': i,
            'label': f"{doc_id}",
            'title': title[:30] + ("..." if len(title) > 30 else ""),
            'score': score,
            'type': doc_type
        })
    
    # Create a simple force-directed graph
    # Since we don't have direct similarity between all pairs, we'll connect to the origin
    center_node = {'id': len(nodes), 'label': 'Query', 'color': '#1e3a8a', 'size': 20}
    nodes.append(center_node)
    
    # Create edges from center to all other nodes
    edges = []
    for i in range(len(recommendations)):
        score = recommendations[i]['score']
        edges.append({
            'from': len(nodes) - 1,  # Center node
            'to': i,
            'value': score * 10,  # Scale for visibility
            'title': f"Similarity: {score:.4f}"
        })
    
    # Using custom HTML/JS for force graph
    import json
    
    # Convert Python objects to JSON strings for JavaScript
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    
    html_content = f"""
    <div id="mynetwork" style="width:100%;height:400px;background-color:#f8fafc;border-radius:10px;"></div>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script type="text/javascript">
        // Create nodes data
        var nodes = new vis.DataSet({nodes_json});
        
        // Create edges data
        var edges = new vis.DataSet({edges_json});
        
        // Create network
        var container = document.getElementById('mynetwork');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        var options = {{
            nodes: {{
                shape: 'dot',
                size: 16,
                font: {{
                    size: 12,
                    color: '#1e3a8a'
                }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                width: 2,
                shadow: true,
                smooth: {{
                    type: 'continuous'
                }}
            }},
            physics: {{
                stabilization: true,
                barnesHut: {{
                    gravitationalConstant: -2000,
                    centralGravity: 0.3,
                    springLength: 150,
                    springConstant: 0.04,
                    damping: 0.09
                }}
            }}
        }};
        var network = new vis.Network(container, data, options);
    </script>
    """.replace("{nodes_json}", nodes_json).replace("{edges_json}", edges_json)
    
    # Display using HTML component
    st.components.v1.html(html_content, height=400)

def display_recommendation_metrics(recommendations: List[Dict[str, Any]]) -> None:
    """Display key metrics about the recommendations."""
    col1, col2, col3, col4 = st.columns(4)
    
    avg_score = np.mean([rec['score'] for rec in recommendations]) if recommendations else 0
    
    # Count unique document types
    doc_types = set()
    years = []
    for rec in recommendations:
        metadata = rec.get('metadata', {})
        if 'document_type' in metadata:
            doc_types.add(metadata['document_type'])
        if 'year' in metadata:
            try:
                years.append(int(metadata['year']))
            except (ValueError, TypeError):
                pass
    
    # Calculate year range
    year_range = f"{min(years)}-{max(years)}" if years else "N/A"
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(recommendations)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Documents</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{avg_score:.3f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg Score</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(doc_types)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Doc Types</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{year_range}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Year Range</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
