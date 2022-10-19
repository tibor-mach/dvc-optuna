Serialized ML docs
==================

.. warning::

   The documentation is not finished yet (for any version...also the project is not even in an alfa stage).
   However, the basic idea has been laid out along with a :doc:`list of to dos <todos>`.


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

Scope
-----
(Batch) data ETLs, data cleaning, standardization, exploration feature engineering or feature stores
are currently out of scope of the project. The idea is to take it up from there
(but some integrations of training data preparation pipelines and models in version tracking via DVC and GTO
are planned)

Also, provisioning of cloud infrastructure used in the ML workflow is included
(in a separate repo, currently just `Azure functions infra <https://github.com/tibor-mach/azure-functions-ml-infra>`_). Those are mainly:

* storage(s) for DVC remotes
* infra for serverless (later perhaps also kubernetes) deployment testing - currently done for Azure
* A container registry for models and development/training containers
*
   (this would be separate from the rest of the infra) Setting up TPI to allow "local remote"
   dvc pipeline experimentation (i.e. without GitHub/Gitlab where...there CML is better)


Main concepts in this repo
--------------------------

*
   ML experimentation can be done in a "declarative" way and the code itself does not
   have to be modified at all for most common ML tasks.
*
   Since the configuration yaml files are the single source of truth for the entire setup
   of a specific ML project, it is very easy to keep track of all of it using DVC while
   keeping the code `DRY <https://en.wikipedia.org/wiki/Don%27t_repeat_yourself>`_.
*
   Individual steps (scripts) of a ML pipeline are modular and
   can be swapped and replaced in the DVC pipeline by other steps with the same inputs
   and outputs. This comes mostly for free with DVC pipelines, provided that the template
   steps are separated enough. E.g. assembling a model and hyperparameter optimization should
   not happen in the same step, calibration of a classifier should be separated from both etc.


.. toctree::
   :hidden:

   Home page <self>
   Overview <overview/index>
   Getting started <tutorial/index>
   How-to guides <howto/index>
   API reference <api>
   To do list <todos>

First steps
-----------
Go to :doc:`tutorial/index` to get everything ready to start working.

You can then have a look at the :doc:`Overview <overview/index>` to see the main concepts
in detail.

How the documentation is organized
----------------------------------
The documentation is split into the following broad areas:

*
   :doc:`Getting started <tutorial/index>` where you can learn the basics of working with
   the project.
*
   :doc:`Overview <overview/index>` which you should read to understand the
   high-level concepts behind how each tool is used and how it all fits together.
*
   :doc:`How-to guides <howto/index>` where you will find recipes for specific common
   problems and use/cases. These guides are more advanced than the tutorial and it is
   expected that you have a basic idea about the project.
*  :doc:`API reference <api>` where you can find an automatically generated
   documentation of modules and functions included in the repository.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
