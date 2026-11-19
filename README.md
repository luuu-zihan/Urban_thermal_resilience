# Urban thermal resilience code and data

This compact package contains selected analysis code and city-level result tables associated with the manuscript.

Contents

- code/01_heatwave_extraction/extract_heatwaves.py identifies heatwave events.
- code/02_thermal_resilience_metrics/ contains the thermal-resistance and thermal-recovery workflows.
- code/03_environmental_associations/test_machine_learning_models.py contains the nested city-grouped machine-learning workflow.
- code/03_environmental_associations/shap_analysis.py summarizes out-of-fold SHAP values by feature.
- data/city_thermal_resilience_metrics/ contains one city-level table for each metric.
- data/shap_results/ contains the two core SHAP feature-importance tables.

The package is intentionally compact. It does not include the full daily satellite and climate inputs, the complete event-level model snapshots, or the plotting files required to rerun every manuscript figure. The two metric scripts share the same calculation logic and are provided as separate public entry points for readability.
