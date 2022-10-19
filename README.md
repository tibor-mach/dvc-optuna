## How to try the demonstration

1. call `make venv` to set up the virtual environment (and activate it by `source .venv/bin/activate`)
2. call `dvc exp run`
3. call `dvc exp show` to observe the checkpointed models for each trial

With the current setup, `dvc repro` won't work. It no likey checkpoints.

This is just a simple demonstration, no metrics or plots are logged in the checkpoints but there is
nothing difficult about that.

Note that it is common to retrain the best model(s) coming from hyperparameter optimization on both
train and test data (and doing a final evaluation on a validation dataset) and in case of cross-validation
it probably makes little sense to checkpoint individual folds (but rather a model refitted on all folds for each trial).

Again, there should be nothing complicated about that though.

## Notes

- Recording Optuna trials with DVC easy - just use dvc.api.make_checkpoint()
- This will also work with pruning (an option to choose whether to checkpoint pruned trials)
- in case of multi-objective optimization, it might make sense to just fit the pareto front. But unless storage
is an issue, the simplest way is to prune checkpointed trials once the study is over (to keep only those on the pareto front)
- it would be good to allow having make_checkpoint() with a parameter to have it ignored by dvc repro or some other way
of making sure that we can keep it in code and keep it in dvc.yaml while keeping the option to do simple dvc repro runs.
- named checkpoints would be a nice addition (could be paired with trial numbers)
- using optuna study callbacks and DVCLive might be a good way to track the process in real time
- adding optuna visualisations to plots automatically?
- hyperparameter space defined in params.yaml? (using optuna de-serialization)
- integration of DVC and optuna CLI?
- probably not a good idea to attempt a one-size fits all integration which restricts the user (Optuna has a lot of options)...on the other hand some optuna integrations kind of work like that already (e.g. `OptunaSearchCV`), always a compromise though.
