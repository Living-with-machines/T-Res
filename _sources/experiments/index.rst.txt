Experiments and evaluation
==========================

Follow these steps to reproduce the experiments in our paper.

1. Obtain the external resources
--------------------------------

Follow the instructions in the ":doc:`resources`" page in the documentation
to obtain the resources required for running the experiments.

2. Preparing the data
-------------------------

To create the datasets that we use in the experiments presented in the paper,
run the following command from the ``./experiments/`` folder:

.. code-block:: bash

    $ python ./prepare_data.py

This script takes care of downloading the LwM and HIPE datasets and format them
as needed in the experiments.

3. Running the experiments
--------------------------

To run the experiments, run the following script from the ``./experiments/``
folder:

.. code-block:: bash

    $ python ./toponym_resolution.py

This script does runs for all different scenarios reported in the experiments in
the paper.

4. Evaluate
-----------

To evaluate the different approaches and obtain a table with results such as the
one provided in the paper, go to the ``./evaluation/`` directory. There, you
should clone the `HIPE scorer <https://github.com/hipe-eval/HIPE-scorer>`_. We
are using the code version at commit ``50dff4e``, and have added the line
``return eval_stats`` at the end of the ``get_results()`` function. From
``./evaluation/``, run the following script to obtain the results in latex
format:

.. code-block:: bash

    $ python display_results.py
