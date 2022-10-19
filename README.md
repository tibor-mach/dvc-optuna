# Notes

- Recording Optuna trials with DVC easy - just use dvc.api.make_checkpoint()
- This will also work with pruning (an option to choose whether to checkpoint pruned trials)
- in case of multi-objective optimization, it might make sense to just fit the pareto front. But unless storage
is an issue, the simplest way is to prune checkpointed trials once the study is over (to keep only those on the pareto front)
- named checkpoints would be a nice addition (could be paired with trial numbers)
- using optuna study callbacks and DVCLive might be a good way to track the process in real time
- adding optuna visualisations to plots automatically?
- hyperparameter space defined in params.yaml? (using optuna de-serialization)
- integration of DVC and optuna CLI?
- probably not a good idea to attempt a one-size fits all integration which restricts the user (Optuna has a lot of options)...on the other hand some optuna integrations kind of work like that already (e.g. `OptunaSearchCV`), always a compromise though.
