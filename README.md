# CLGAM

This repository contains the code to generate the results given in our paper.

## Our dependencies

We exported our environment to `environment.yaml`.  To create a conda environment that mirrors ours, run:

```
conda env create -f environment.yml
```

## Our Model

Our model is encoded in `proximal_gradient_descent.py`.  Here's a minimal example of how to run it:

```{python}
from proximal_gradient_descent import warm_start

# Get data
X = some matrix/tensor

# Initialize
L_init = [np.eye(d) for d in dims]

# Choose which and how many regularization parameters you want to use
glassoregs = np.logspace(0.2, -2, 50) # this was a good range for (50, 50) synthetic data, but may need changing
frobreg = 0

# Run our model
Lss, _ = warm_start(X, L_init, glassoregs=glassoregs, dont_warm_start=True, frobreg=frobreg)

# If you specify 'sample axes', the model will treat them as independent, so if you have an n-by-m matrix
# and sample_axes={0}, we will fit an LGAM model (due to independent samples) with n samples and m features.
Lss_lgam, _ = warm_start(X, L_init, glassoregs=glassoregs, sample_axes={0}, dont_warm_start=True, frobreg=frobreg)
```

The output will be a list of tuples containing the estimates for each axis.  For example, a 50x100x200 tensor would result in a list of tuples of the form (50x50, 100x100, 200x200) lower triangular matrices encoding the DAG.  Each element in the list corresponds to one regularization value in `glassoregs`, in order.

`glassoregs` may also be passed in as a list of tuples, if one wants to regularize each axis with different strengths.  If `glassoregs` is a list of floats, it will apply uniformly
to all axes.

## Main Paper

In most cases, when an experiment was run we jotted down the runtime of the experiment.  We often had multiple experiments going at once on the same personal computer (in particular `synthetic-experiments.ipynb` and `effect-of-ordering.ipynb` were run simultaneously) so these times will be slower than if everything is run sequentially.  Of course, the exact runtimes will be different from computer-to-computer.  The runtimes are high for the two aforementioned notebooks about synthetic experiments, as we cast a fairly fine grid of L1 penalties; if this is prohibitive, consider changing lines of the form `glassoregs = np.logspace(A, B, C)` to have a smaller value of C.

* Table 1: The code to generate this is located in `synthetic-experiments.ipynb`.  Each cell contains an experiment leading to one of the rows in the table; all resultant values were rounded to three decimal places when included in the paper.
* Table 2: The code to generate this is located in `runtime.ipynb`.
* Table 3: The code to generate this is located in `effect-of-ordering.ipynb`.  Each cell contains an experiment leading to one of the rows in the table; all resultant values were rounded to three decimal places when included in the paper.
* Figure 1: The code to generate this is located in `krum_experiment_pseudotime.ipynb`.
* Figure 2: The code to generate this is located in `krum_experiment_pseudotime.ipynb`.
* Figure 3: The code to generate this is located in `krum_experiment_pseudotime.ipynb`.
* Figure 4: The code to generate this is located in `coil_experiment.ipynb`.
* Figure 5: The code to generate this is located in `coil_experiment.ipynb`.

For the tables, the code block will output text of this form:

```
*************** Problem type ***************
        CLGAM:
            Max MCC: ?
            AUCPR: ?
        LGAM:
            Max MCC: ?
            AUCPR: ?
        GmGM:
            Max MCC: ?
            AUCPR: ?
```


## Supplementary Materials

* Figure A: The code to generate this is located in `krum_experiment_pseudotime.ipynb`.  Warning: it takes a long time to run over the whole grid search.
* Figure B: The code to generate this is located in `krum_experiment_no_domain.ipynb`.  Warning: it takes a long time to run over the whole grid search.