import optuna
import json
import joblib
import dvc.api

from dvc.api import make_checkpoint
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

PARAMS = dvc.api.params_show()


def objective(trial):

    X, y = make_classification(n_features=10, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    C = trial.suggest_float("C", 1e-7, 10.0, log=True)

    clf = LogisticRegression(C=C)
    clf.fit(X_train, y_train)

    metrics = {
        "C": C,
        "optuna_score": clf.score(X_test, y_test),
    }

    # save model, metrics (similarly for plots or anything else)
    with open(PARAMS["paths"]["metrics"], "w") as f:
        json.dump(metrics, f)

    # we could of course use MLEM here instead
    joblib.dump(clf, PARAMS["paths"]["trial_model"], compress=1)

    # if make_checkpoint allowed for named checkpoints, it could be useful (pair them with trials)
    make_checkpoint()
    return metrics["optuna_score"]


if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    joblib.dump(study, PARAMS["paths"]["study"], compress=1)
