# Evaluation Plan for EU Legal Recommender

## 1. Overview
We will evaluate recommendation quality for two client profiles (`renewable_energy_client` and `bottling_company_client`) using Precision@10 and NDCG@10. Ground-truth is their `historical_documents` arrays.

## 2. Test Data Preparation
- Read each profile JSON in `recommender/profiles/`
- Extract `historical_documents` → relevant_docs list
- Assign relevance_scores = 1.0 for all
- Output combined `evaluation/test_data.json`

## 3. Parameter Grids
1. **Similarity**
   - Sweep `text_weight` in [0.0,0.1,…,1.0]
   - `categorical_weight = 1 - text_weight`
2. **Embedding**
   - Sweep `summary_weight` in [0.0,0.1,…,1.0]
   - `keyword_weight = 1 - summary_weight`
3. **Personalization**
   - All triples `(expert_weight, historical_weight, categorical_weight)` ∈ {0.0,0.1,…,1.0}, summing to 1.0

## 4. Evaluation Approaches

### A) WeightOptimizer Grid‑search
- Use `src/utils/weight_optimizer.py`
- For each client:
  ```python
  pine = PineconeRecommender(...)  
  personal = PersonalizedRecommender(pine)
  opt = WeightOptimizer(personal, test_data_path="evaluation/test_data.json")
  sim_res = opt.optimize({"text_weight":[...]}, metric="precision@10")
  emb_res = opt.optimize({"summary_weight":[...]}, metric="precision@10")
  pers_res = opt.optimize({
      "expert_weight":[...],
      "historical_weight":[...],
      "categorical_weight":[...]
    }, metric="precision@10")
  # repeat for ndcg@10
  ```
- Save best_weights & best_score per sweep

### B) run_benchmarks Loop
- Programmatically override `config.SIMILARITY`, `WeightConfig.embedding`, and `USER_PROFILE` weights
- Call `scripts/run_benchmarks.py --profile-path profiles/<client>.json --output-file evaluation/results_<client>_<sweep>.json`
- Collect performance & personalized results

## 5. Reporting
- For each client & each sweep, tabulate:
  - Best Precision@10 & weight config
  - Best NDCG@10 & weight config
- Compare across clients: which weight regimes generalize from narrow → broad profiles

## 6. Next Steps
- Implement **Test Data** script
- Implement **Option A** orchestrator
- Implement **Option B** orchestrator
- Run experiments and analyze
