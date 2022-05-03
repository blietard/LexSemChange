# <center style='font-family:serif'> LexSemChange </center>

---
## Requirements
**Python packages :**

- numpy
- scipy
- pandas
- matplotlib
- mangoes

**Miscellaneous requirements :**

The path to a complete data folder from SemEval2020 Task 1 should be specified at the begining of the main file.

**File organisation**
- `results_visual.ipynb` : notebook with visualisation of results (influence of parameters)
- `analysis.ipynb` : notebook with analysis of results (nearest neighbors, false negative, etc.)
- `static.ipynb` : notebook with methods to construct static representations matrices and create results (scores) for them.  Also includes comparison to reference code.
- `testOP.py` : script to produce a visual test of well alignment with Orthogonal Procrustes.
- `static_exp_pipe.py` : script that embed a full pipeline to produce many static representations matrices and scores for varying parameters. Used for exploratory study.
- `results/` : folder in which results are stored as `tab` separated CSV.
- `results_img/` : folder to store whatever resulting image being produced.
- `tools/` : folder that contains useful modules:
    - `analysis` contains classes and function used for model's results analysis.
    - `count_based` contains functions to compute static representations (counts, PPMI, SVD).
    - `pipelines` contains pipeline functions that calls multiple functions at once from `count_based`.
    - `readers` contains classes used to read datasets.
    - `utils` contains OP alignment and misc. useful functions.

