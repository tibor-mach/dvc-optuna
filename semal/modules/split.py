from sklearn.model_selection import GroupShuffleSplit


def train_test_split(
    df,
    splitter,
    target_column: str = "target",
    group_column: str = None,
):
    """
    Splits a pandas dataframe to train and test using a chosen splitter.
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with all data
    splitter : sklearn.model_selection.BaseShuffleSplit
        An instance of a class which inherits from the sklearn BaseShuffleSplit class
    group_column : str, optional
        group to be used by group splitters,
        required when ``splitter`` is "GroupShuffleSplit", by default None
    target_column : str, optional
        name of the column with the target values
    test_size : float, optional
        portion of the data used for the test dataset, by default 0.3
    random_state : int, optional
        random state used in the splitter (if applicable), by default 42
    Returns
    -------
    a tuple of 2 pandas Dataframes
        The train and test datraframes
    Raises
    ------
    TypeError
        When using a group splitter but not specifying the group_column
    """

    if group_column is None and isinstance(splitter, GroupShuffleSplit):
        raise TypeError("For group splitters, group_column is a required parameter!")

    X = df.drop(columns=target_column)
    y = df[target_column]
    train_indices, test_indices = next(splitter.split(X, y, groups=X[group_column]))

    df_train = df.loc[train_indices]
    df_test = df.loc[test_indices]

    return df_train, df_test
