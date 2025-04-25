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
Run weight optimization specifically for the bottling company client only.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
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


def main():
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
    base = Path(__file__).parent.resolve()
    profiles_dir = base.parent / "profiles"
    test_data_file = base / "test_data.json"
    output_file = base / "bottling_optimization_results.json"
    
    # Parameter sweeps
    text_weights = frange(0.0, 1.0, 0.1)
    summary_weights = frange(0.0, 1.0, 0.1)
    expert_ws, hist_ws, cat_ws = generate_personalization_lists()
    
    # Store results
    results = {}
    
    # Only run for bottling company client
    profile_path = profiles_dir / "bottling_company_client.json"
    client = "bottling_company_client"
    print(f"\nOptimizing weights for {client}...")
    
    # Load profile data
    with open(profile_path) as f:
        profile_data = json.load(f)
    
    # Extract profile components
    try:
        if "profile" in profile_data:
            # Standard structure
            profile = profile_data["profile"]
            expert_desc = profile["expert_profile"]["description"]
            cat_prefs = profile["categorical_preferences"]
            hist_docs = profile["historical_documents"]
        else:
            # Alternative structure
            print("Using alternative profile structure")
            expert_desc = profile_data["expert_profile"]["description"]
            cat_prefs = profile_data["categorical_preferences"]
            hist_docs = profile_data["historical_documents"]
    except KeyError as e:
        print(f"Error extracting profile components: {str(e)}, trying alternate format...")
        try:
            # Load from test data if available
            with open(test_data_file) as f:
                test_data_all = json.load(f)
            if client in test_data_all:
                comp = test_data_all[client]["profile_components"]
                expert_desc = comp["expert_profile"]["description"]
                cat_prefs = comp["categorical_preferences"]
                hist_docs = comp["historical_documents"]
                print("Successfully loaded profile components from test data")
            else:
                print(f"Could not find profile components for {client}, skipping...")
                return
        except Exception as e2:
            print(f"Failed to load alternate format: {str(e2)}, skipping...")
            return

    # Client-specific test data with historical documents as ground truth
    try:
        with open(test_data_file) as f:
            test_data_all = json.load(f)
            
        client_test_data = {}
        if client in test_data_all:
            client_test_data[client] = test_data_all[client]
            client_test_file = base / f"{client}_test_data.json"
            with open(client_test_file, 'w') as f:
                json.dump(client_test_data, f, indent=2)
            print(f"Created client-specific test file: {client_test_file}")
        else:
            print(f"Warning: No test data for {client}, skipping...")
            return
    except Exception as e:
        print(f"Error preparing client test data: {str(e)}, skipping...")
        return
        
    # Creating empty dicts to store results
    client_results = {
        "full_profile": {},
        "expert_only": {},
        "categorical_only": {},
        "historical_only": {}
    }
    
    # 1. Test with FULL PROFILE
    print(f"Testing with full profile...")
    personal = PersonalizedRecommender(api_key=api_key, index_name=index_name)
    personal.create_expert_profile(client, expert_desc)
    personal.set_categorical_preferences(client, cat_prefs)
    for doc_id in hist_docs:
        try:
            personal.add_historical_document(client, doc_id)
        except Exception as e:
            print(f"Warning: skipping historical doc {doc_id}: {str(e)}")
            
    opt = WeightOptimizer(personal, test_data_path=str(client_test_file))
    
    # Similarity weights (text vs categorical)
    sim_params = {
        "text_weight": text_weights,
        "categorical_weight": [round(1 - tw, 3) for tw in text_weights]
    }
    opt.metric = "precision@10"
    sim_prec = opt.optimize(sim_params, k_values=[10], metrics=["precision"])
    opt.metric = "ndcg@10"
    sim_ndcg = opt.optimize(sim_params, k_values=[10], metrics=["ndcg"])
    
    # Embedding weights (summary vs keyword)
    emb_params = {
        "summary_weight": summary_weights,
        "keyword_weight": [round(1 - sw, 3) for sw in summary_weights]
    }
    opt.metric = "precision@10"
    emb_prec = opt.optimize(emb_params, k_values=[10], metrics=["precision"])
    opt.metric = "ndcg@10"
    emb_ndcg = opt.optimize(emb_params, k_values=[10], metrics=["ndcg"])
    
    # Personalization weights (expert vs historical vs categorical)
    pers_params = {
        "expert_weight": expert_ws,
        "historical_weight": hist_ws,
        "categorical_weight": cat_ws
    }
    opt.metric = "precision@10"
    pers_prec = opt.optimize(pers_params, k_values=[10], metrics=["precision"])
    opt.metric = "ndcg@10"
    pers_ndcg = opt.optimize(pers_params, k_values=[10], metrics=["ndcg"])
    
    client_results["full_profile"] = {
        "similarity": {"precision@10": sim_prec, "ndcg@10": sim_ndcg},
        "embedding": {"precision@10": emb_prec, "ndcg@10": emb_ndcg},
        "personalization": {"precision@10": pers_prec, "ndcg@10": pers_ndcg}
    }
    
    # 2. Test with EXPERT PROFILE ONLY
    print(f"Testing with expert profile only...")
    personal = PersonalizedRecommender(api_key=api_key, index_name=index_name)
    personal.create_expert_profile(client, expert_desc)
    
    opt = WeightOptimizer(personal, test_data_path=str(client_test_file))
    
    # Only need personalization weights here
    pers_params = {
        "expert_weight": [1.0],  # Only expert profile
        "historical_weight": [0.0],
        "categorical_weight": [0.0]
    }
    opt.metric = "precision@10"
    pers_prec = opt.optimize(pers_params, k_values=[10], metrics=["precision"])
    opt.metric = "ndcg@10"
    pers_ndcg = opt.optimize(pers_params, k_values=[10], metrics=["ndcg"])
    
    client_results["expert_only"] = {
        "personalization": {"precision@10": pers_prec, "ndcg@10": pers_ndcg}
    }
    
    # 3. Test with CATEGORICAL PREFERENCES ONLY
    print(f"Testing with categorical preferences only...")
    personal = PersonalizedRecommender(api_key=api_key, index_name=index_name)
    personal.set_categorical_preferences(client, cat_prefs)
    
    opt = WeightOptimizer(personal, test_data_path=str(client_test_file))
    
    pers_params = {
        "expert_weight": [0.0],
        "historical_weight": [0.0],
        "categorical_weight": [1.0]  # Only categorical prefs
    }
    opt.metric = "precision@10"
    pers_prec = opt.optimize(pers_params, k_values=[10], metrics=["precision"])
    opt.metric = "ndcg@10"
    pers_ndcg = opt.optimize(pers_params, k_values=[10], metrics=["ndcg"])
    
    client_results["categorical_only"] = {
        "personalization": {"precision@10": pers_prec, "ndcg@10": pers_ndcg}
    }
    
    # 4. Test with HISTORICAL DOCUMENTS ONLY
    print(f"Testing with historical documents only...")
    personal = PersonalizedRecommender(api_key=api_key, index_name=index_name)
    
    # Try to find alternative document IDs if needed
    success_count = 0
    alt_hist_docs = []
    
    # First, try exact document IDs
    for doc_id in hist_docs:
        try:
            personal.add_historical_document(client, doc_id)
            success_count += 1
            print(f"Added historical document: {doc_id}")
        except Exception as e:
            print(f"Warning: couldn't add historical doc {doc_id}: {str(e)}")
            # Try alternative formats that might work
            alt_formats = [
                doc_id.replace('L', '').replace('R', ''),  # Remove L/R type indicators
                doc_id.replace('32', '3'),                 # Try shorter format
                'celex:' + doc_id                         # Try with prefix
            ]
            alt_added = False
            
            # Try each alternative format
            for alt_id in alt_formats:
                try:
                    personal.add_historical_document(client, alt_id)
                    success_count += 1
                    alt_hist_docs.append(alt_id)
                    print(f"Added alternative document ID: {alt_id} (for {doc_id})")
                    alt_added = True
                    break
                except Exception:
                    pass  # Try next alternative
                    
            if not alt_added:
                print(f"Failed to add {doc_id} with any ID format")
    
    opt = WeightOptimizer(personal, test_data_path=str(client_test_file))
    
    pers_params = {
        "expert_weight": [0.0],
        "historical_weight": [1.0],  # Only historical docs
        "categorical_weight": [0.0]
    }
    opt.metric = "precision@10"
    pers_prec = opt.optimize(pers_params, k_values=[10], metrics=["precision"])
    opt.metric = "ndcg@10"
    pers_ndcg = opt.optimize(pers_params, k_values=[10], metrics=["ndcg"])
    
    client_results["historical_only"] = {
        "personalization": {"precision@10": pers_prec, "ndcg@10": pers_ndcg}
    }
    
    # Save results
    results[client] = client_results
    
    # Write results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results for {client} to {output_file}")
    
    print(f"\nCompleted all weight optimizations for bottling company client. Results saved to {output_file}")


if __name__ == '__main__':
    main()
