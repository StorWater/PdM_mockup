Stor water: predictive maintenance mockup (pdm)
==============================



Project Organization
------------


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── storpdm                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module      
    │   ├── make_dataset.py
    │   ├── build_features.py
    │   ├── predict_model.py
    │   ├── train_model.py
    │   └── visualize.py <- Scripts to create exploratory and results oriented visualizations


## Setup Docker container

TODO


## Setup of jupyter notebooks

This is the configuration of the extensions I am using:

```
set NODE_OPTIONS=--max-old-space-size=4096
jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.1 --no-build
jupyter labextension install jupyterlab-plotly@1.4.0 --no-build
jupyter labextension install plotlywidget@1.4.0 --no-build
jupyter labextension install @jupyterlab/toc --no-build
jupyter labextension install @ryantam626/jupyterlab_code_formatter --no-build
pip install jupyterlab_code_formatter
jupyter serverextension enable --py jupyterlab_code_formatter --no-build
jupyter lab build
```


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
