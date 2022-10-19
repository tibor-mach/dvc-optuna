import sklearn
import pkgutil
import inspect
import importlib
from operator import itemgetter


def all_useful_classes():
    # TODO Pick a better name...
    # TODO Expand to cover also user-defined modules
    # TODO reintroduce filtering so users can use this to see what is available at each stage
    """

    Get a list of all (nonabstract) classes which inherit from BaseEstimator
    or BaseShuffleSplit from sklearn.

    Classes that are defined in test-modules are not included.

    .. note::
        This function is still very experimental and will almost certainly be changed later.

    Returns
    -------
    useful_classes : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.
    """
    # lazy import to avoid circular imports from sklearn.base
    from sklearn.utils._testing import ignore_warnings
    from sklearn.base import BaseEstimator
    from sklearn.model_selection._split import BaseShuffleSplit

    def is_abstract(c):
        if not (hasattr(c, "__abstractmethods__")):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    modules_to_ignore = {
        "tests",
        "externals",
        "setup",
        "conftest",
        "enable_hist_gradient_boosting",
    }
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=sklearn.__path__, prefix="sklearn."
        ):
            mod_parts = modname.split(".")
            if any(part in modules_to_ignore for part in mod_parts) or "._" in modname:
                continue
            module = importlib.import_module(modname)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, est_cls) for name, est_cls in classes if not name.startswith("_")
            ]
            all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [
        c
        for c in all_classes
        if (issubclass(c[1], BaseEstimator) and c[0] != "BaseEstimator")
    ]
    splitters = [
        c
        for c in all_classes
        if (issubclass(c[1], BaseShuffleSplit) and c[0] != "BaseShuffleSplit")
    ]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]
    splitters = [c for c in splitters if not is_abstract(c[1])]

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators + splitters), key=itemgetter(0))


def deserialize_class(class_config: dict):
    """
    Imports and creates an instance of a class with given parameters.

    Parameters
    ----------
    class_config : dict
        a dictionary with keys "class_module" and "class_name" used for module importing
        and "params" containing a dictionary of keyword arguments of the class.

    Returns
    -------
    Class instance
        An instance of the class with parameters given by "params"
    """

    # import the chosen estimator class
    Class_ = getattr(
        importlib.import_module(class_config["class_module"]),
        class_config["class_name"],
    )

    return Class_(**class_config["params"])
