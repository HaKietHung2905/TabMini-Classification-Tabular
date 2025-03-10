import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from sklearn.utils import check_X_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.base import BaseEstimator, ClassifierMixin


class XGBoost(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        time_limit: int = 3600,
        device="cuda",
        seed: int = 42,
        kwargs: dict = {},
        small_dataset: bool = False,
    ):
        self.time_limit = time_limit
        self.device = device
        self.seed = seed
        self.kwargs = kwargs
        self.result_df = None

        self.param_grid = {
            "n_estimators": [10, 20, 30] if small_dataset else [50, 100, 200],
            "max_depth": [3, 5] if small_dataset else [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1] if small_dataset else [0.01, 0.1, 0.3],
        }

    def fit(self, X, y, X_test, y_test) -> "XGBoost":
        X_train, y_train = check_X_y(X, y, accept_sparse=True)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        results = []
        best_f1 = -1
        best_model = None

        param_combinations = [
            dict(zip(self.param_grid.keys(), v))
            for v in np.array(np.meshgrid(*self.param_grid.values())).T.reshape(
                -1, len(self.param_grid.keys())
            )
        ]

        for param in param_combinations:
            param["n_estimators"] = int(param["n_estimators"])
            param["max_depth"] = int(param["max_depth"])

            current_model = XGBClassifier(
                **param,
                objective="binary:logistic",
                eval_metric="auc",
                use_label_encoder=False,
                random_state=self.seed
            )
            current_model.fit(X_train, y_train)

            # make predictions
            y_pred = current_model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="binary")
            acc = accuracy_score(y_test, y_pred)

            results.append({**param, "accuracy": acc, "f1_score": f1})

            # Update best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = current_model

        self.result_df = pd.DataFrame(results).sort_values(
            by="f1_score", ascending=False
        )
        self.model = best_model
        self.best_params_ = best_model.get_params() if best_model else None



    def save_results(self, filename):
        if self.result_df is not None:
            self.result_df.to_csv(filename, index=False)
