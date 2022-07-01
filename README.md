## Manipulating SHAP via Adversarial Data Perturbations (Student Abstract)
 
This repository contains data and code for the article: 

H. Baniecki, P. Biecek. **Manipulating SHAP via Adversarial Data Perturbations (Student Abstract)**. In: *AAAI Conference on Artificial Intelligence (AAAI)*, 36(11):12907-12908, 2022. URL: https://ojs.aaai.org/index.php/AAAI/article/view/21590.

> **2022-06-12 Update.** Values of Kendall tau *distance* reported in the article come from the `scipy.stats.kendalltau()` function, which in fact computes the Kendall tau *coefficient* (see the SciPy GitHub issue on "Kendall tau distance" https://github.com/scipy/scipy/issues/7089). Knowing that the (normalized) distance equals `(1 - coefficient) / 2`, the actual distance values equal 0.20 and 0.07, respectively. We updated `export_and_table.ipynb` to account for this error, which doesn't change the conclusion.

Python version: 3.9.2

- `alg` directory with the algorithm's code implementation
- `data` directory with the datasets
- `results` directory with the pickled metadata and logs
- `scenario_heart.ipynb` recreates the `heart` analysis
- `scenario_apartment.ipynb` recreates the `apartment` analysis
- `export_and_table.ipynb` converts the `.p` result files and computes Kendall tau
- `figures_and_table.R` creates Figures and computes the remaining distances
