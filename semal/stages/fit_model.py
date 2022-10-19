import pandas as pd

# import mlem.api as mlemapi

if __name__ == "__main__":
    import joblib
    import dvc.api

    # init
    # TODO? use os.path.splitext(os.path.basename(__file__)[0]) instead of "assemble_model"
    params = dvc.api.params_show(stages=["fit_model"])
    paths = params["paths"]
    df_train = pd.read_csv(paths["data_train"])

    # TODO: Currently, this is fairly trivial. This should be a place for refitting on all data, classifier calibration etc.
    if params["hyperparam_tuning"]:
        # the result of optimization is automatically refitted using the best parameters
        # so we can simply load the result
        model_fitted = joblib.load(paths["hypersearch_result"]).best_estimator_
    else:
        # TODO: the following will fail unless target is called "target"
        model_fitted = joblib.load(paths["model_unfitted"]).fit(
            X=df_train, y=df_train["target"]
        )

    # save

    joblib.dump(model_fitted, paths["model_fitted"], compress=1)

    # mlemapi.save(
    #     obj=model_fitted,
    #     path=paths["model_fitted"],
    #     sample_data=df_train,
    # )
