# Serialized Machine Learning

A "cookie cutter"-style project which provides templates to common machine
learning tasks which are serialized into yaml configuration files. Based largely
on tools by Iterative.

Currently limited to tabular data and scikit-learn compatible models. See the TODOs below
for other currently considered features.

The aim is to provide an alternative to AutoML frameworks which are very inflexible
and almost always come with a vendor lock-in. This way, templace can be used like AutoML
if desired, but they can be broken down and modified easily as well...or some parts can be
replaced, not forcing the user to use same framework for the entire  ML workflow.

It could also serve as a basis for a collection of ML recipes,
which are fully open source, ready to production and using gitops practices.
# TODO phase 1
### (Basic structure)
- [ ]
    Add a mechanism to add custom transformers/estimators to `all_transformers`. This
    should probably work just like `sklearn.utils.all_estimators` but going through all
    estimators including custom ones (even ones imported from other packages like [scikit-lego](https://scikit-lego.readthedocs.io/en/latest/))
- [ ]
    Consolidate paramrs/config locations...the main object (for the tabular scikit setup) is an estimator.
    All its parameters (including the hyperparameter space) should be defined in one place. The params.yamls
    should be organized based on objects they cover

- [ ]
    A terraform AWS sandbox to set up all infra for DVC and remote repro/experiments
    (a private container registry)
- [ ]
    Set up infra and scripts for standardizing local devcontainers with remote
    containers used for remote training (with CML in CICD or directly with TPI for prototyping)
- [ ]
    Add CML (repro, PR, reporting) with the standardized containers
- [ ]
    Add an easy option to run an experiment remotely via TPI (probably not for
    individual stages, see arguments below...)

## TPI Remote experiments/repro sketch workflow
1.
    Make changes to the pipeline and/or parameters
2.
    Use TPI to copy and install all that is needed to run the experiment remotely
    (copy `.dvc/cache/runs` to prevent redoing existing experiments ?)
3.  Setting up the remote script:
    Reproduce the pipeline remotely with the `--no-commit` option. The option is not really
    necessary but it demonstrates intent and should speed things up slightly. Then,
    in the root directory of your repo run

    ```dvc ls --dvc-only . -R | zip -@ results/data.zip```

    in the remote.
4.
    Back in the local environment, once everything finishes run `terraform destroy`.
    The `results` directory now contains the results of the pipeline run, but they still
    need to be added to `dvc` locally and (optionally) commited. So, unpack everything to the root,
    overwriting the outputs, the dvc.lock file and adding a record to `.dvc/cache/runs`.
5.
    Now our entire project directory should be in the same state as if we ran the pipeline locally
    with `dvc repro --no-commit`. Run `dvc commit` to commit all changes.

### Running individual stages remotely (later development thoughs...probably not worth it)
    Reasoning: Optimization of workloads for each stage...However, typically there is
    not going to be that much diversity in computational needs between stages to justify
    the extra complexity => this is likely premature optimization.

    Anyway, in case it turns out to be useful `.dvc/cache/runs` seem to track stage runs separately,
    maybe a good start for the separation of stages? Would need something remote to orchestrate
    and gather results from stages. Maybe a "driver" EC2/VM instance which would spin new EC2s
    for each stage, gather results and send them back? Not very elegant though, too many moving pieces ...

# TODO phase 2
### (More modularity)
- [ ] Rework train-test splitting to be properly modular
- [ ] Rework hyperparameter optimization to be more general (and up to date with Optuna 3.0)
- [ ] Rework evaluation to be more general and serialized (does evaluation also benefit from being serialized???)

# TODOs phase 3
### (CICD)
- [ ]
    Upgrade the simple data registry to a MLEM+GTO+DVC Data+model registry and modify
    the pipelines to work with that
- [ ]
    Finish deployment of serverless infra for deployment for AWS (Azure already done)
- [ ]
    Set up CICD to work with the registry
- [ ]
    Once all of the above is done, write it out in the docs for all repositories
    (link several sphinx docs of the various repos together), publish (the relevant bits)
    on PyPi, the docs on readthedocs and maybe write a medium blogpost or something.

# TODOs phase 4
### (Tasks other than regression/classification and frameworks other than scikit-learn)
- [ ] Recommender engine?
- [ ] Clustering?
- [ ] [Darts](https://github.com/unit8co/darts) (forecasting library)
- [ ] Hugging Face
- [ ] Fast.ai
- [ ] Pytorch geometric
- [ ] Pytorch (maybe too general???)
