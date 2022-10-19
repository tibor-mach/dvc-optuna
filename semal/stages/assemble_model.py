if __name__ == "__main__":
    import joblib
    import yaml

    import dvc.api
    from sklearn.pipeline import Pipeline
    from semal.modules.preprocessing import deserialize_col_transformer
    from semal.utils.utils import deserialize_class

    # init
    # TODO? use os.path.splitext(os.path.basename(__file__)[0]) instead of "assemble_model"
    params = dvc.api.params_show(stages=["assemble_model"])

    # process
    preprocessing = deserialize_col_transformer(
        params["paths"]["params_preprocessing"],
        **params["assemble_model"]["column_transformer_kwargs"],
    )

    params_estimator = yaml.safe_load(open(params["paths"]["params_estimators"]))
    params_estimator = params_estimator[params["assemble_model"]["estimator_used"]]
    estimator = deserialize_class(params_estimator)

    model_pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("estimator", estimator),
        ]
    )

    # save
    joblib.dump(model_pipeline, params["paths"]["model_unfitted"], compress=1)
