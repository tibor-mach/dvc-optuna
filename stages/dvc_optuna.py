import optuna
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

    # save
    joblib.dump(clf, PARAMS["paths"]["model"], compress=1)
    make_checkpoint()
    return clf.score(X_test, y_test)


if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
