import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted


class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError(
                "Some estimators are missing parameters: %s" % missing_params
            )
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.best_model = None
        self.best_score = None

    def fit(self, X, y, cv=3, n_jobs=2, verbose=1, scoring=None):
        """Perform GridSearchCV and saves results

        # TODO docu

        Parameters
        ----------
        X : [type]
            [description]
        y : [type]
            [description]
        cv : int, optional
            [description], by default 3
        n_jobs : int, optional
            [description], by default 3
        verbose : int, optional
            [description], by default 1
        scoring : [type], optional
            [description], by default None

        """
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            gs = GridSearchCV(
                estimator=self.models[key],
                param_grid=self.params[key],
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                scoring=scoring,
                refit=True,
                return_train_score=True,
            )
            gs.fit(X, y)
            self.grid_searches[key] = gs

        # Save best model
        _, _ = self.best_model_(set_attr=True)

    def best_model_(self, set_attr=True):
        """Returns the best model

        Parameters
        ----------
        set_attr : bool
            Sets the attributes best_model and best_score. Default: True

        Returns
        -------
        best_model : model
            Best-performing model
        best_score
            Score of the best performing model
        """

        for i, k in enumerate(self.grid_searches):
            # Initialize
            if i == 0:
                best_model = self.grid_searches[k].best_estimator_
                best_score = self.grid_searches[k].best_score_
            # Update if better metrics are found
            else:
                if self.grid_searches[k].best_score_ > best_score:
                    best_model = self.grid_searches[k].best_estimator_
                    best_score = self.grid_searches[k].best_score_

        if set_attr:
            self.best_model = best_model
            self.best_score = best_score

        return best_model, best_score

    def score_summary(self, sort_by="mean_score"):
        """Returns the outcome of the grid search organized by score

        Parameters
        ----------
        sort_by : str, optional
            [description], by default "mean_score"

        Returns
        -------
        pd.DataFrame
            Score summary table
        """

        list_df = []
        for k in self.grid_searches:
            df_score = pd.DataFrame(self.grid_searches[k].cv_results_)
            df_score["model"] = k
            list_df.append(df_score)

        df = pd.concat(list_df).sort_values("mean_test_score", ascending=False)

        return df
