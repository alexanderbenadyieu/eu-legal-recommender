#!/usr/bin/env python3
import sys
from pathlib import Path as _Path
# Ensure `recommender` root is on PYTHONPATH for imports
sys.path.insert(0, str(_Path(__file__).parent.parent))

# Import and apply dictionary patching solution
# This must be imported and applied before any other imports
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from evaluation.patch_all_dictionaries import safe_merge_dicts, main as apply_dict_patches
# Apply patches to prevent dictionary addition errors
apply_dict_patches()

"""
Diversity evaluation: Compute diversity metrics for different weight configurations.
"""
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from scipy.spatial.distance import cosine, pdist, squareform
import pandas as pd

# Load environment variables from .env (override existing)
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env', override=True)

from src.models.pinecone_recommender import PineconeRecommender
from src.models.personalized_recommender import PersonalizedRecommender
from src.utils.weight_optimizer import WeightOptimizer


def frange(start, stop, step):
    vals = []
    x = start
    while x <= stop:
        vals.append(round(x, 3))
        x += step
    return vals


def generate_personalization_lists(step=0.1):
    experts = frange(0.0, 1.0, step)
    histor = frange(0.0, 1.0, step)
    triples = []
    for e in experts:
        for h in histor:
            c = round(1.0 - e - h, 3)
            if 0.0 <= c <= 1.0:
                triples.append((e, h, c))
    return (sorted({e for e, h, c in triples}),
            sorted({h for e, h, c in triples}),
            sorted({c for e, h, c in triples}))


def calculate_content_diversity(doc_ids, recommender, k=5):
    """
    Calculate content diversity based on document embeddings.
    
    Args:
        doc_ids: List of document IDs in the recommendation list
        recommender: Recommender instance to retrieve document embeddings
        k: Number of top documents to consider
        
    Returns:
        Average pairwise cosine distance between document embeddings
    """
    if len(doc_ids) <= 1:
        return 0.0
    
    # Limit to top-k documents
    doc_ids = doc_ids[:k]
    
    # Get document embeddings
    embeddings = []
    for doc_id in doc_ids:
        try:
            doc = recommender.get_document_by_id(doc_id)
            if doc and 'embedding' in doc:
                embeddings.append(doc['embedding'])
            else:
                logging.warning(f"No embedding found for document {doc_id}")
                # Use a random vector as fallback
                embeddings.append(np.random.random(768))  # Assuming 768-dim embeddings
        except Exception as e:
            logging.error(f"Error retrieving document {doc_id}: {str(e)}")
            embeddings.append(np.random.random(768))
    
    # Calculate pairwise distances
    if len(embeddings) <= 1:
        return 0.0
        
    embeddings = np.array(embeddings)
    distances = pdist(embeddings, metric='cosine')
    return np.mean(distances)


def calculate_categorical_diversity(doc_ids, recommender, k=5):
    """
    Calculate categorical diversity based on document metadata.
    
    Args:
        doc_ids: List of document IDs in the recommendation list
        recommender: Recommender instance to retrieve document metadata
        k: Number of top documents to consider
        
    Returns:
        Dictionary with diversity scores for different categorical features
    """
    if len(doc_ids) <= 1:
        return {
            'form': 0.0, 
            'type': 0.0, 
            'subject_matter': 0.0, 
            'overall': 0.0,
            'document_types': {},
            'subject_matters': {}
        }
    
    # Limit to top-k documents
    doc_ids = doc_ids[:k]
    
    # Get document categories
    categories = {
        'form': [],
        'type': [],
        'subject_matter': []
    }
    
    # Store actual values for detailed analysis
    doc_types = []
    subject_matters = []
    
    for doc_id in doc_ids:
        try:
            doc = recommender.get_document_by_id(doc_id)
            if doc and 'metadata' in doc:
                metadata = doc['metadata']
                
                # Extract categories
                if 'form' in metadata:
                    form = metadata['form']
                    categories['form'].append(form)
                
                if 'document_type' in metadata:
                    doc_type = metadata['document_type']
                    # Handle both string and list formats
                    if isinstance(doc_type, list):
                        categories['type'].extend(doc_type)
                        doc_types.extend(doc_type)
                    else:
                        categories['type'].append(doc_type)
                        doc_types.append(doc_type)
                
                if 'subject_matter' in metadata:
                    subject = metadata['subject_matter']
                    # Handle both string and list formats
                    if isinstance(subject, list):
                        categories['subject_matter'].extend(subject)
                        subject_matters.extend(subject)
                    else:
                        categories['subject_matter'].append(subject)
                        subject_matters.append(subject)
        except Exception as e:
            logging.error(f"Error retrieving document {doc_id}: {str(e)}")
            for cat in categories:
                categories[cat].append('unknown')
    
    # Calculate diversity for each category (unique proportion)
    diversity = {}
    for category, values in categories.items():
        if not values:
            diversity[category] = 0.0
        else:
            unique_ratio = len(set(values)) / len(values)
            diversity[category] = unique_ratio
    
    # Calculate overall diversity (average of all categories)
    if not any(categories.values()):
        diversity['overall'] = 0.0
    else:
        valid_diversities = [div for cat, div in diversity.items() if cat != 'overall' and div > 0]
        diversity['overall'] = sum(valid_diversities) / len(valid_diversities) if valid_diversities else 0.0
    
    # Add detailed category distributions
    from collections import Counter
    diversity['document_types'] = dict(Counter(doc_types))
    diversity['subject_matters'] = dict(Counter(subject_matters))
    
    return diversity


def create_test_data_from_profiles(profiles_dir, output_path):
    """Create comprehensive test data from client profiles.
    
    This function generates test data for weight optimization by extracting all components
    from client profiles, including historical documents, expert profiles, and categorical
    preferences. The historical documents are used as ground truth for evaluation.
    """
    test_data = {}
    profiles = list(profiles_dir.glob("*_client.json"))
    
    print(f"Creating comprehensive test data from {len(profiles)} profiles...")
    for profile_path in profiles:
        client = profile_path.stem
        print(f"Processing {client}...")
        
        try:
            with open(profile_path) as f:
                profile_data = json.load(f)
            
            # Check for different profile structures
            # Structure 1: {"profile": {...}} (standard structure)
            # Structure 2: {"client_name": {...}} (test data structure)
            
            if "profile" in profile_data:
                # Standard profile structure
                profile = profile_data["profile"]
            elif client in profile_data:
                # Test data might have client name as the key
                profile = profile_data[client]
                if "profile_components" in profile:
                    # Extract from profile_components if available
                    profile = profile["profile_components"]
            else:
                # Assume profile data is directly in the root (fallback)
                profile = profile_data
            
            # Extract all profile components for comprehensive testing
            historical_docs = []
            expert_profile = {}
            categorical_prefs = {}
            
            # Try to get historical documents
            if "historical_documents" in profile:
                historical_docs = profile["historical_documents"]
            elif "profile_components" in profile and "historical_documents" in profile["profile_components"]:
                historical_docs = profile["profile_components"]["historical_documents"]
            
            # Try to get expert profile
            if "expert_profile" in profile:
                expert_profile = profile["expert_profile"]
            elif "profile_components" in profile and "expert_profile" in profile["profile_components"]:
                expert_profile = profile["profile_components"]["expert_profile"]
            
            # Try to get categorical preferences
            if "categorical_preferences" in profile:
                categorical_prefs = profile["categorical_preferences"]
            elif "profile_components" in profile and "categorical_preferences" in profile["profile_components"]:
                categorical_prefs = profile["profile_components"]["categorical_preferences"]
            
            # Create test entry with all components
            test_data[client] = {
                "query": "",  # Empty query for personalized recommendations
                "profile_components": {
                    "historical_documents": historical_docs,
                    "expert_profile": expert_profile,
                    "categorical_preferences": categorical_prefs
                }
            }
            
        except Exception as e:
            print(f"Error processing {client}: {str(e)}")
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Saved test data to {output_path}")
    return test_data


def analyze_coverage_metrics(all_results):
    """
    Analyze coverage and overlap across recommendations for different clients.
    
    Args:
        all_results: Dictionary with results for all clients
        
    Returns:
        Dictionary with coverage metrics
    """
    from collections import Counter, defaultdict
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    
    # Extract all recommendations
    all_recommendations = {}
    for client, results in all_results.items():
        # Get recommendations from the combined profile results
        if "combined" in results:
            recs = results["combined"].get("recommendations", [])
            all_recommendations[client] = recs
    
    if not all_recommendations:
        return {"error": "No recommendations found in results"}
    
    # Calculate unique document counts
    all_docs = []
    for client, recs in all_recommendations.items():
        all_docs.extend(recs)
    
    total_recs = len(all_docs)
    unique_docs = set(all_docs)
    unique_count = len(unique_docs)
    
    # Calculate recommendation overlap
    client_ids = list(all_recommendations.keys())
    jaccard_matrix = np.zeros((len(client_ids), len(client_ids)))
    
    for i, client_i in enumerate(client_ids):
        for j, client_j in enumerate(client_ids):
            if i == j:
                jaccard_matrix[i][j] = 1.0
            else:
                set_i = set(all_recommendations[client_i])
                set_j = set(all_recommendations[client_j])
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                jaccard_matrix[i][j] = intersection / union if union > 0 else 0.0
    
    # Calculate average overlap
    mask = np.ones(jaccard_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)  # Exclude diagonal elements
    avg_overlap = jaccard_matrix[mask].mean() if mask.any() else 0.0
    
    # Extract document types and subject matters
    doc_types = Counter()
    subject_matters = Counter()
    
    for client, results in all_results.items():
        if "combined" in results and "categorical_diversity" in results["combined"]:
            cat_div = results["combined"]["categorical_diversity"]
            
            # Add document types
            if "document_types" in cat_div:
                for dt, count in cat_div["document_types"].items():
                    doc_types[dt] += count
                    
            # Add subject matters
            if "subject_matters" in cat_div:
                for sm, count in cat_div["subject_matters"].items():
                    subject_matters[sm] += count
    
    # Calculate industry coverage
    industry_coverage = defaultdict(set)
    client_industries = {}
    
    for client in all_recommendations.keys():
        # Extract industry from client ID (e.g., 'pharma_01' -> 'pharma')
        parts = client.split('_')
        if len(parts) >= 1:
            industry = parts[0]
            client_industries[client] = industry
            industry_coverage[industry].update(all_recommendations[client])
    
    industry_metrics = {}
    for industry, docs in industry_coverage.items():
        client_count = sum(1 for c, i in client_industries.items() if i == industry)
        industry_metrics[industry] = {
            "client_count": client_count,
            "unique_documents": len(docs),
            "documents_per_client": len(docs) / client_count if client_count > 0 else 0
        }
    
    # Compile all metrics
    coverage_metrics = {
        "total_recommendations": total_recs,
        "unique_documents": unique_count,
        "uniqueness_ratio": unique_count / total_recs if total_recs > 0 else 0,
        "average_overlap": avg_overlap,
        "document_types": dict(doc_types.most_common()),
        "document_type_count": len(doc_types),
        "subject_matters": dict(subject_matters.most_common()),
        "subject_matter_count": len(subject_matters),
        "industry_coverage": industry_metrics
    }
    
    return coverage_metrics


def main():
    """Run diversity evaluation for all client profiles."""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    profiles_dir = base_dir / "profiles" / "fake_clients"
    results_file = base_dir / "evaluation" / "diversity_results.json"
    coverage_file = base_dir / "evaluation" / "coverage_metrics.json"
    
    # Make directories if they don't exist
    base_dir.mkdir(exist_ok=True)
    (base_dir / "evaluation").mkdir(exist_ok=True)
    
    # Load Pinecone API key from .env or environment
    env_path = Path(__file__).parent.parent / '.env'
    api_key = None
    if env_path.exists():
        for l in env_path.read_text().splitlines():
            if l.startswith('PINECONE_API_KEY'):
                api_key = l.split('=', 1)[1].strip()
                break
    if not api_key:
        api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env or environment")
    
    index_name = os.getenv("PINECONE_INDEX_NAME", "eu-legal-documents-legal-bert")
    print(f"Using Pinecone index: {index_name}")
    
    # Project paths
    base_dir = Path(__file__).parent.parent
    profiles_dir = base_dir / "profiles"
    test_data_file = base_dir / "evaluation" / "test_data.json"
    results_file = base_dir / "evaluation" / "diversity_results.json"
    
    # Make directories if they don't exist
    base_dir.mkdir(exist_ok=True)
    (base_dir / "evaluation").mkdir(exist_ok=True)
    
    # Create test data if not present
    if not test_data_file.exists():
        test_data = create_test_data_from_profiles(profiles_dir, test_data_file)
    else:
        with open(test_data_file, 'r') as f:
            test_data = json.load(f)
    
    if not test_data:
        print("Error: No test data available.")
        return
    
    # Weight params to test
    text_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    summary_weights = text_weights.copy()
    expert_ws, hist_ws, cat_ws = generate_personalization_lists(step=0.25)
    
    # Diversity results for all clients
    all_results = {}
    
    # Process each client
    for client, data in test_data.items():
        print(f"\nProcessing client: {client}")
        
        # Get client profile components
        client_test_file = base_dir / "evaluation" / f"{client}_test_data.json"
        with open(client_test_file, 'w') as f:
            json.dump({client: data}, f, indent=2)
        
        # Extract profile components
        components = data.get("profile_components", {})
        historical_docs = components.get("historical_documents", [])
        expert_desc = components.get("expert_profile", {}).get("description", "")
        categorical_prefs = components.get("categorical_preferences", {})
        
        client_results = {}
        
        # 1. Test with FULL PROFILE
        print(f"Testing with full profile...")
        
        # Track results for different weight configurations
        sim_diversity = []
        emb_diversity = []
        pers_diversity = []
        
        # 1.1 Text vs Categorical Weights
        print(f"Testing text vs categorical weights...")
        
        # Initialize recommender with client profile
        personal = PersonalizedRecommender(api_key=api_key, index_name=index_name)
        personal.create_expert_profile(client, expert_desc)
        personal.set_categorical_preferences(client, categorical_prefs)
        
        # Add each historical document
        for doc_id in historical_docs:
            personal.add_historical_document(client, doc_id)
        
        for tw in tqdm(text_weights, desc="Text weight"):
            cw = round(1.0 - tw, 3)
            
            # Set weights for this test
            personal.set_similarity_weights(text_weight=tw, categorical_weight=cw)
            
            # Get recommendations
            try:
                # Note: The patched API requires user_id and expects top_k instead of limit
                recs = personal.get_personalized_recommendations(
                    user_id=client,
                    query_text="",  # Empty string triggers profile-based recommendations
                    top_k=5
                )
                rec_ids = [doc['id'] for doc in recs] if recs else []
            except Exception as e:
                logging.error(f"Error getting recommendations: {str(e)}")
                rec_ids = []
            
            # Calculate diversity
            content_div = calculate_content_diversity(rec_ids, personal, k=5)
            cat_div = calculate_categorical_diversity(rec_ids, personal, k=5)
            
            # Record results
            sim_diversity.append({
                "weights": {
                    "text_weight": tw,
                    "categorical_weight": cw
                },
                "content_diversity": content_div,
                "categorical_diversity": cat_div,
                "recommendations": rec_ids[:5] if rec_ids else []
            })
        
        # 1.2 Summary vs Keyword Weights
        print(f"Testing summary vs keyword weights...")
        
        for sw in tqdm(summary_weights, desc="Summary weight"):
            kw = round(1.0 - sw, 3)
            
            # Create recommender with this client profile
            personal = PersonalizedRecommender(api_key=api_key, index_name=index_name)
            personal.create_expert_profile(client, expert_desc)
            personal.set_categorical_preferences(client, categorical_prefs)
            
            # Add each historical document
            for doc_id in historical_docs:
                personal.add_historical_document(client, doc_id)
            
            # Set weights for this test
            # Note: There is no embedding weights, using similarity weights instead
            personal.set_similarity_weights(text_weight=sw, categorical_weight=kw)
            
            # Get recommendations
            try:
                # Note: The patched API requires user_id and expects top_k instead of limit
                recs = personal.get_personalized_recommendations(
                    user_id=client,
                    query_text="",  # Empty string triggers profile-based recommendations
                    top_k=5
                )
                rec_ids = [doc['id'] for doc in recs] if recs else []
            except Exception as e:
                logging.error(f"Error getting recommendations: {str(e)}")
                rec_ids = []
            
            # Calculate diversity
            content_div = calculate_content_diversity(rec_ids, personal, k=5)
            cat_div = calculate_categorical_diversity(rec_ids, personal, k=5)
            
            # Record results
            emb_diversity.append({
                "weights": {
                    "summary_weight": sw,
                    "keyword_weight": kw
                },
                "content_diversity": content_div,
                "categorical_diversity": cat_div,
                "recommendations": rec_ids[:5] if rec_ids else []
            })
        
        # 1.3 Personalization Component Weights
        print(f"Testing personalization weights...")
        
        for ew in tqdm(expert_ws, desc="Expert weight"):
            for hw in hist_ws:
                cw = round(1.0 - ew - hw, 3)
                if cw < 0 or cw > 1.0:
                    continue
                
                # Create recommender with this client profile
                personal = PersonalizedRecommender(api_key=api_key, index_name=index_name)
                personal.create_expert_profile(client, expert_desc)
                personal.set_categorical_preferences(client, categorical_prefs)
                
                # Add each historical document
                for doc_id in historical_docs:
                    personal.add_historical_document(client, doc_id)
                
                # Set weights for this test
                personal.set_profile_component_weights(
                    expert_weight=ew,
                    historical_weight=hw,
                    categorical_preference_weight=cw
                )
                
                # Get recommendations
                try:
                    recs = personal.get_personalized_recommendations(
                        user_id=client,
                        query_text="",  # Empty string triggers profile-based recommendations
                        top_k=5
                    )
                    rec_ids = [doc['id'] for doc in recs] if recs else []
                except Exception as e:
                    logging.error(f"Error getting recommendations: {str(e)}")
                    rec_ids = []
                
                # Calculate diversity
                content_div = calculate_content_diversity(rec_ids, personal, k=5)
                cat_div = calculate_categorical_diversity(rec_ids, personal, k=5)
                
                # Record results
                pers_diversity.append({
                    "weights": {
                        "expert_weight": ew,
                        "historical_weight": hw,
                        "categorical_weight": cw
                    },
                    "content_diversity": content_div,
                    "categorical_diversity": cat_div,
                    "recommendations": rec_ids[:5] if rec_ids else []
                })
        
        # Save results for this client
        client_results["full_profile"] = {
            "similarity": sim_diversity,
            "embedding": emb_diversity,
            "personalization": pers_diversity
        }
        
        # 2. Test with EXPERT PROFILE ONLY
        print(f"Testing with expert profile only...")
        personal = PersonalizedRecommender(api_key=api_key, index_name=index_name)
        personal.create_expert_profile(client, expert_desc)
        
        # Get recommendations
        try:
            recs = personal.get_personalized_recommendations(
                user_id=client,
                query_text="",  # Empty string triggers profile-based recommendations
                top_k=5
            )
            rec_ids = [doc['id'] for doc in recs] if recs else []
        except Exception as e:
            logging.error(f"Error getting recommendations: {str(e)}")
            rec_ids = []
        
        # Calculate diversity
        content_div = calculate_content_diversity(rec_ids, personal, k=5)
        cat_div = calculate_categorical_diversity(rec_ids, personal, k=5)
        
        # Save expert profile results
        client_results["expert_profile"] = {
            "content_diversity": content_div,
            "categorical_diversity": cat_div,
            "recommendations": rec_ids[:5] if rec_ids else []
        }
        
        # 3. Test with HISTORICAL DOCUMENTS ONLY
        print(f"Testing with historical documents only...")
        personal = PersonalizedRecommender(api_key=api_key, index_name=index_name)
        
        # Add each historical document
        for doc_id in historical_docs:
            personal.add_historical_document(client, doc_id)
        
        # Get recommendations
        try:
            recs = personal.get_personalized_recommendations(
                user_id=client,
                query_text="",  # Empty string triggers profile-based recommendations
                top_k=5
            )
            rec_ids = [doc['id'] for doc in recs] if recs else []
        except Exception as e:
            logging.error(f"Error getting recommendations: {str(e)}")
            rec_ids = []
        
        # Calculate diversity
        content_div = calculate_content_diversity(rec_ids, personal, k=5)
        cat_div = calculate_categorical_diversity(rec_ids, personal, k=5)
        
        # Save historical documents results
        client_results["historical_documents"] = {
            "content_diversity": content_div,
            "categorical_diversity": cat_div,
            "recommendations": rec_ids[:5] if rec_ids else []
        }
        
        # 4. Test with CATEGORICAL PREFERENCES ONLY
        print(f"Testing with categorical preferences only...")
        
        personal = PersonalizedRecommender(api_key=api_key, index_name=index_name)
        personal.set_categorical_preferences(client, categorical_prefs)
        
        # Get recommendations
        try:
            recs = personal.get_personalized_recommendations(
                user_id=client,
                query_text="",  # Empty string triggers profile-based recommendations
                top_k=5
            )
            rec_ids = [doc['id'] for doc in recs] if recs else []
        except Exception as e:
            logging.error(f"Error getting recommendations: {str(e)}")
            rec_ids = []
        
        # Calculate diversity
        content_div = calculate_content_diversity(rec_ids, personal, k=5)
        cat_div = calculate_categorical_diversity(rec_ids, personal, k=5)
        
        # Save categorical preferences results
        client_results["categorical_preferences"] = {
            "content_diversity": content_div,
            "categorical_diversity": cat_div,
            "recommendations": rec_ids[:5] if rec_ids else []
        }
        
        # Store results for this client
        all_results[client] = client_results
        
        # Save client-specific results
        client_results_file = base_dir / "evaluation" / f"{client}_diversity_results.json"
        with open(client_results_file, 'w') as f:
            json.dump(client_results, f, indent=2)
        print(f"Saved diversity results for {client} to {client_results_file}")
    
    # Save all results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved all diversity results to {results_file}")
    
    # Calculate and save coverage metrics
    print("\nCalculating recommendation coverage metrics...")
    coverage_metrics = analyze_coverage_metrics(all_results)
    
    with open(coverage_file, 'w') as f:
        json.dump(coverage_metrics, f, indent=2)
    
    # Print summary of coverage metrics
    print("\n" + "="*80)
    print("EU LEGAL RECOMMENDER COVERAGE ANALYSIS")
    print("="*80 + "\n")
    
    print(f"Total recommendations: {coverage_metrics['total_recommendations']}")
    print(f"Unique documents: {coverage_metrics['unique_documents']}")
    print(f"Uniqueness ratio: {coverage_metrics['uniqueness_ratio']:.2f} ({coverage_metrics['uniqueness_ratio']*100:.1f}%)")
    print(f"Average overlap between clients: {coverage_metrics['average_overlap']:.2f} ({coverage_metrics['average_overlap']*100:.1f}%)\n")
    
    print("DOCUMENT TYPE DISTRIBUTION:")
    for doc_type, count in sorted(coverage_metrics['document_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"- {doc_type}: {count} occurrences")
    print(f"Total unique document types: {coverage_metrics['document_type_count']}\n")
    
    print("SUBJECT MATTER DISTRIBUTION:")
    for subject, count in sorted(coverage_metrics['subject_matters'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"- {subject}: {count} occurrences")
    print(f"Total unique subject matters: {coverage_metrics['subject_matter_count']}\n")
    
    print("INDUSTRY COVERAGE:")
    for industry, data in sorted(coverage_metrics['industry_coverage'].items()):
        print(f"- {industry}: {data['unique_documents']} unique documents for {data['client_count']} clients")
    
    print(f"\nDetailed coverage metrics saved to {coverage_file}")
    print("="*80)


if __name__ == "__main__":
    main()
