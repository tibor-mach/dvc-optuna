import optuna
import joblib
import dvc.api
import time


from dvclive.optuna import DVCLiveCallback
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

    time.sleep(1)
    return metrics["optuna_score"]


if __name__ == "__main__":

    dvclive_callback = DVCLiveCallback()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, callbacks=[dvclive_callback])
    joblib.dump(study, PARAMS["paths"]["study"], compress=1)
