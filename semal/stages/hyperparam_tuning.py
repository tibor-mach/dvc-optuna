import optuna
import pandas as pd

from semal.modules.tuning import deserialize_optuna_distributions


def hyperparam_tuning(
    df_train,
    model_unfitted,
    hyperparam_space: dict,
    search_params: dict,
    target_column: str = "target",
):
    """
    Perform a hyperparameter search with optuna using cross-validation on the
    train set and return the search results.

    Parameters
    ----------
    df_train : pandas.DataFrame
        training set for cross-validation search
    model_unfitted : sklearn.pipeline.Pipeline
        Model pipeline with unfitted features pipeline and estimator
    hyperparam_space : dict of optuna distributions
        A dictionary with names of hyperparameters for tuning and optuna distributions
        (see ``deserialize_optuna_distributions``)
    search_params : dict
        additional hyperparameter search parameters,
        see https://optuna.readthedocs.io/en/latest/reference/generated/optuna.integration.OptunaSearchCV.html

    Returns
    -------
    optuna.integration.OptunaSearchCV
        Results of the hyperparameter search,
        see https://optuna.readthedocs.io/en/latest/reference/generated/optuna.integration.OptunaSearchCV.html
    """

    optuna_search_results = optuna.integration.OptunaSearchCV(
        estimator=model_unfitted,
        param_distributions=hyperparam_space,
        **search_params,
    )

    optuna_search_results.fit(df_train, df_train[target_column])

    return optuna_search_results


if __name__ == "__main__":
    import logging
    import joblib
    import dvc.api

    # init
    # TODO? use os.path.splitext(os.path.basename(__file__)[0]) instead of "assemble_model"
    params = dvc.api.params_show(stages=["hyperparam_tuning"])
    estimator_used = params["estimator_used"]

    if params["hyperparam_tuning"]:

        hyperparam_space = deserialize_optuna_distributions(
            params[estimator_used]["hyperparam_space"]
        )

        # load
        df_train = pd.read_csv(params["paths"]["data_train"])
        model_unfitted = joblib.load(params["paths"]["model_unfitted"])

        # process

        hypersearch_result = hyperparam_tuning(
            df_train=df_train,
            model_unfitted=model_unfitted,
            hyperparam_space=hyperparam_space,
            search_params=params[estimator_used]["search_params"],
            target_column=params["target_column"],
        )

        # save
        joblib.dump(
            hypersearch_result, params["paths"]["hypersearch_result"], compress=1
        )
    else:
        logging.warning(
            "Optimize_hyperparameters is set to False, skipping optimization."
        )
