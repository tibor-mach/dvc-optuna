import warnings
import yaml

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def transformer_from_dict(transformer_dict, all_transformers=None):
    """
    Creates an instance of a transformer object from a configuration given by a dictionary
    with transformer class name and parameters.

    Parameters
    ----------
    transformer_dict : dict
        A dictionary with keys "transformer" denoting the class name of the transformer
        and "params" which is a dictionary of its parameters.
    all_transformers : dict, optional
        A dictionary of ``{name: class}`` where ``name`` is a class name and ``class``
        is the actual type of the class. If not provided, the default dictionary of all
        transformers from scikit-learn is used.
        See the `scikit-learn docs <https://scikit-learn.org/stable/modules/generated/sklearn.utils.all_estimators.html#sklearn.utils.all_estimators>`_
        for mode detail.
    Returns
    -------
    sklearn.base.TransformerMixin
        Transformer instance given by ``transformer_dict``
    """
    if all_transformers is None:
        # use a dictionary of default sklearn transformers
        from sklearn.utils import all_estimators

        all_transformers = dict(all_estimators(type_filter="transformer"))
        # specific scikit-learn keywords
        if transformer_dict["transformer"] in {"passthrough", "drop"}:
            return transformer_dict["transformer"]
        else:
            return all_transformers[transformer_dict["transformer"]](
                # empty dict default in case there are no custom parameters
                **transformer_dict.get("params", {})
            )


def pipeline_from_dict(steps_dict, all_transformers=None):
    """
    Creates a scikit-learn Pipeline object from a dictionary of pipeline steps.

    .. warning::
        For better compatibility with `dvc params <https://dvc.org/doc/command-reference/params>`_
        the steps are given by a dictionary instead of a list. However, this means that the
        order of keys matters and Python>=3.7 is required for this functionality to work
        reliably (previous versions do not preserve the order of dictionaries).


    Parameters
    ----------
    steps_dict : dict
        A dictionary ``{step_name: step_config}`` where ``step_name`` is a name of a
        pipeline step and ``step_config`` is a dictionary containing the configuration
        of the step. See the documentation of ``transformer_from_dict`` for more details.
    all_transformers: dict, optional
        A dictionary of transformer names and classes. See the documentation of
        ``transformer_from_dict`` for more details.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A Pipeline chaining all the steps given by ``steps_dict``.
    """

    pipe_steps = [
        (step_name, transformer_from_dict(step_config, all_transformers))
        for step_name, step_config in steps_dict.items()
    ]

    return Pipeline(pipe_steps)


def col_transformer_from_dict(
    config_dict,
    all_transformers=None,
    **column_transformer_kwargs,
):
    """
    Creates an instance of a scikit-learn ``ColumnTransformer`` given by ``config_dict``.

    Parameters
    ----------
    config_dict : dict
        A dictionary of the form

        .. code-block:: python

            {pipeline_name: {
                "steps": steps_dict,
                "features_in": features_list,
                }
            }


        where  ``pipeline_name`` is a name of each pipeline in the ``ColumnTransformer``,
        ``steps_dicts`` contains the configuration of the steps in the pipeline and
        ``features_list`` is the list of features to be processed by the pipeline.
        See the documentation of ``pipeline_from_dict`` for more details.

    all_transformers: dict, optional
        A dictionary of transformer names and classes which is used to create instances
        of transformers in the steps of each pipeline in the resulting
        ``ColumnTransformer``. See the documentation of ``transformer_from_dict``
        for more details.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        A column transformer given by ``config_dict``.
    """
    column_pipelines = []
    for pipe_name in config_dict:
        if len(config_dict[pipe_name]["features_in"]) > 0:
            column_pipelines.append(
                (
                    pipe_name,
                    pipeline_from_dict(
                        config_dict[pipe_name]["steps"], all_transformers
                    ),
                    config_dict[pipe_name]["features_in"],
                )
            )
        else:
            warnings.warn(
                f"Pipeline {pipe_name} has no input features, skipping.", UserWarning
            )

    return ColumnTransformer(transformers=column_pipelines, **column_transformer_kwargs)


def deserialize_col_transformer(
    # TODO: Change this to a "yaml wrapper" of col_transformer_from_dict
    config_path: str,
    all_transformers: dict = None,
    **column_transformer_kwargs,
):
    """
    Deserializes the configuration of transformers (typically used in a preprocessing
    pipeline) into a sklearn.preprocessing.ColumnTransformer instance. The configuration
    has to be provided in yaml format where each transformer has to be declared using
    the following syntax:

    .. code-block:: yaml

        <custom_name_of_transformer>: steps:
            <name_of_step_1>:
                transformer: <TransformerClassName> params:
                    <transformer_param_1>: ... <transformer_param_2>: ...
            <custom_name_of_step_2>:
                ...
        # list of named input columns features_in: [
                <input_feature_1>, <input_feature_2>, ...
        ]


    The <TransformerName> has to match the keys from ``all_transformers``. A warning is
    raised if the ``features_in`` list is empty for some of the transformers.


    Parameters
    ----------
    params_transformers : str
        path to the yaml configuration file
    all_transformers: dict
        A dictionary with transformer names as keys and their classes as values. This
        dictionary is used to look up the appropriate transformer classes by name. If it
        is not provided, the default dictionary of all scikit-learn transformers is used
        instead.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        The column transformer given by the configuration file
    """

    config_dict = yaml.safe_load(open(config_path, "r"))
    return col_transformer_from_dict(
        config_dict, all_transformers, **column_transformer_kwargs
    )
