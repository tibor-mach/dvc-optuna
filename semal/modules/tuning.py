import json

from optuna.distributions import json_to_distribution


def deserialize_optuna_distributions(
    hyperparam_space_dict: dict, names_prefix: str = "estimator__"
):
    """
    Converts a dictionary of hyperparameters and optuna distribution parameters
    to a keyword arguments dictionary used in the ``param_distributions``
    parameter of ``optuna.integration.OptunaSearchCV.``

    Parameters
    ----------
    hyperparam_space_dict : dict
        A dictionary of hyperparameter names as keys and a dictionary
        corresponding to serialized optuna distributions as values
    names_prefix : str, optional
        prefix added to the names of parameters for use with scikit-learn
        pipelines, by default "estimator__"

    Returns
    -------
    dict
        A dictionary of hyperparameter names with the added dunder prefix
        convention as keys and optuna distributions as values.

    Reference
    ---------
    `optuna distributions docs<https://optuna.readthedocs.io/en/latest/reference/distributions.html>`_
    """

    distribution_kwargs = {
        names_prefix
        + hyperparam_name: json_to_distribution(json.dumps(distribution_dict))
        for hyperparam_name, distribution_dict in hyperparam_space_dict.items()
    }

    return distribution_kwargs
