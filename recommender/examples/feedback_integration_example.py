#!/usr/bin/env python
"""
Example demonstrating the integration of user feedback in the EU Legal Recommender system.

This script shows how to:
1. Collect user feedback on recommendations
2. Apply feedback to boost relevant documents
3. Analyze feedback statistics
"""

import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.pinecone_recommender import PineconeRecommender
from src.models.personalized_recommender import PersonalizedRecommender
from src.models.feedback import FeedbackManager
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger('feedback_example')

def simulate_user_feedback(feedback_manager, user_id, recommendations, ratings):
    """
    Simulate user providing feedback on recommendations.
    
    Args:
        feedback_manager: FeedbackManager instance
        user_id: ID of the user providing feedback
        recommendations: List of recommendation results
        ratings: Dictionary mapping document IDs to ratings
    """
    print(f"\nSimulating feedback from user {user_id}...")
    
    for result in recommendations:
        doc_id = result['id']
        if doc_id in ratings:
            rating = ratings[doc_id]
            feedback_manager.add_feedback(
                user_id=user_id,
                document_id=doc_id,
                rating=rating,
                query="renewable energy",
                feedback_type='rating'
            )
            print(f"  - Added rating {rating} for document {doc_id}")

def visualize_feedback_statistics(feedback_manager):
    """
    Visualize feedback statistics.
    
    Args:
        feedback_manager: FeedbackManager instance
    """
    stats = feedback_manager.get_feedback_statistics()
    
    if stats['total_feedback_count'] == 0:
        print("No feedback data available for visualization.")
        return
    
    # Set up plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot rating distribution
    ratings = list(stats['rating_distribution'].keys())
    counts = list(stats['rating_distribution'].values())
    
    ax1.bar(ratings, counts, color='skyblue')
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Count')
    ax1.set_title('Rating Distribution')
    ax1.set_xticks(ratings)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        ax1.text(ratings[i], count + 0.5, str(count), ha='center')
    
    # Plot user feedback summary
    user_stats = {}
    for user_id in feedback_manager.feedback_data['user_id'].unique():
        user_pref = feedback_manager.get_user_preferences(user_id)
        user_stats[user_id] = user_pref['average_rating']
    
    if user_stats:
        user_ids = list(user_stats.keys())
        avg_ratings = list(user_stats.values())
        
        ax2.bar(user_ids, avg_ratings, color='lightgreen')
        ax2.set_xlabel('User ID')
        ax2.set_ylabel('Average Rating')
        ax2.set_title('Average Rating by User')
        ax2.set_ylim(0, 5.5)
        
        # Add rating labels on bars
        for i, rating in enumerate(avg_ratings):
            ax2.text(user_ids[i], rating + 0.1, f"{rating:.2f}", ha='center')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'feedback_statistics.png')
    print(f"\nFeedback statistics visualization saved to {output_dir / 'feedback_statistics.png'}")
    
    # Show plot
    plt.show()

def compare_results(original_results, adjusted_results):
    """
    Compare original and feedback-adjusted results.
    
    Args:
        original_results: Original recommendation results
        adjusted_results: Feedback-adjusted recommendation results
    """
    print("\nComparing original and feedback-adjusted results:")
    print(f"{'Document ID':<20} {'Original Score':<15} {'Adjusted Score':<15} {'Change':<10}")
    print("-" * 60)
    
    # Create mapping of document IDs to results
    original_map = {r['id']: r for r in original_results}
    adjusted_map = {r['id']: r for r in adjusted_results}
    
    # Get all document IDs
    all_doc_ids = sorted(set(list(original_map.keys()) + list(adjusted_map.keys())))
    
    for doc_id in all_doc_ids[:10]:  # Show top 10 for brevity
        orig_score = original_map.get(doc_id, {}).get('score', 0)
        adj_score = adjusted_map.get(doc_id, {}).get('score', 0)
        change = adj_score - orig_score
        change_pct = (change / orig_score * 100) if orig_score > 0 else 0
        
        print(f"{doc_id:<20} {orig_score:<15.4f} {adj_score:<15.4f} {change_pct:+.2f}%")
    
    # Check for rank changes
    original_order = [r['id'] for r in original_results[:10]]
    adjusted_order = [r['id'] for r in adjusted_results[:10]]
    
    if original_order != adjusted_order:
        print("\nRanking changes detected in top 10 results!")
        print(f"{'Original Rank':<15} {'Document ID':<20} {'New Rank':<15}")
        print("-" * 50)
        
        for i, doc_id in enumerate(original_order):
            if doc_id in adjusted_order:
                new_rank = adjusted_order.index(doc_id)
                if i != new_rank:
                    print(f"{i+1:<15} {doc_id:<20} {new_rank+1:<15}")

def main():
    # Check if Pinecone API key is available
    api_key = os.environ.get('PINECONE_API_KEY')
    if not api_key:
        print("Pinecone API key not found. Please set the PINECONE_API_KEY environment variable.")
        print("For this example, we'll use simulated data instead.")
        use_simulated_data = True
    else:
        use_simulated_data = False
    
    # Initialize feedback manager
    feedback_store_path = Path('data/feedback/feedback_store.csv')
    feedback_manager = FeedbackManager(feedback_store_path)
    
    if use_simulated_data:
        # Load simulated recommendation results
        with open('data/examples/simulated_recommendations.json', 'r') as f:
            recommendations = json.load(f)
    else:
        # Initialize recommender
        recommender = PineconeRecommender(
            api_key=api_key,
            index_name='eu-legal-docs'
        )
        
        # Get recommendations
        query = "renewable energy policy in the European Union"
        recommendations = recommender.get_recommendations(
            query=query,
            limit=20
        )
    
    print(f"Retrieved {len(recommendations)} recommendations")
    
    # Simulate feedback from multiple users
    user_feedback = {
        'user1': {
            'CELEX:32018L2001': 5,  # Renewable Energy Directive
            'CELEX:52021PC0557': 4,  # Proposal to amend Renewable Energy Directive
            'CELEX:32009L0028': 3,   # Previous Renewable Energy Directive
            'CELEX:32018R1999': 4    # Governance Regulation
        },
        'user2': {
            'CELEX:32018L2001': 4,
            'CELEX:52021PC0557': 5,
            'CELEX:32009L0028': 4,
            'CELEX:32018R1999': 3
        },
        'user3': {
            'CELEX:32018L2001': 3,
            'CELEX:52021PC0557': 2,
            'CELEX:32009L0028': 5,
            'CELEX:32018R1999': 4
        }
    }
    
    # Add feedback from each user
    for user_id, ratings in user_feedback.items():
        simulate_user_feedback(feedback_manager, user_id, recommendations, ratings)
    
    # Get feedback statistics
    stats = feedback_manager.get_feedback_statistics()
    print("\nFeedback Statistics:")
    print(f"Total feedback count: {stats['total_feedback_count']}")
    print(f"Number of users: {stats['user_count']}")
    print(f"Number of documents: {stats['document_count']}")
    print(f"Average rating: {stats['average_rating']:.2f}")
    
    # Apply feedback to adjust recommendations
    print("\nApplying feedback to adjust recommendations...")
    adjusted_results = feedback_manager.apply_feedback_to_results(
        recommendations,
        boost_factor=0.3
    )
    
    # Compare original and adjusted results
    compare_results(recommendations, adjusted_results)
    
    # Visualize feedback statistics
    visualize_feedback_statistics(feedback_manager)
    
    print("\nFeedback integration example completed successfully!")

if __name__ == "__main__":
    main()
