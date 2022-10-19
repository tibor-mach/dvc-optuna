if __name__ == "__main__":
    # TODO: make this stage more modular
    import dvc.api
    import pandas as pd
    from semal.modules.split import train_test_split
    from semal.utils.utils import deserialize_class
    from sklearn.preprocessing import LabelEncoder

    # TODO? use os.path.splitext(os.path.basename(__file__)[0]) instead of "train_test_split"
    params = dvc.api.params_show(stages=["train_test_split"])

    # load
    with dvc.api.open(**params["data"]) as data:
        df = pd.read_csv(data)

    # TODO This is out of scope for the project use and should be done prior to modelling.
    # remove rows with missing targets
    df = df.dropna(subset=params["target_column"]).reset_index(drop=True)

    # TODO This is out of scope for the project use and should be done prior to modelling.
    # encode targets (for non-sklearn models)
    label_encoder = LabelEncoder()
    df[params["target_column"]] = label_encoder.fit_transform(
        df[params["target_column"]]
    )

    # process
    splitter = deserialize_class(params["splitter"])
    df_train, df_test = train_test_split(df, splitter, **params["train_test_split"])

    # save
    df_train.to_csv(params["paths"]["data_train"], index=False)
    df_test.to_csv(params["paths"]["data_test"], index=False)
