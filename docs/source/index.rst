========================================================================
T-Res: A Toponym Resolution Pipeline for Digitised Historical Newspapers
========================================================================

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://github.com/Living-with-machines/T-Res/blob/master/LICENSE
   :alt: License

T-Res is an end-to-end pipeline for toponym resolution for digitised historical
newspapers. Given an input text (a sentence or a text), T-Res identifies the
places that are mentioned in it, links them to their corresponding Wikidata
IDs, and provides their geographic coordinates. T-Res has been designed to
tackle common problems of working with digitised historical newspapers.

The pipeline has three main components:

#. **The Recogniser** performs named entity recognition.
#. **The Ranker** performs candidate selection and ranking.
#. **The Linker** performs entity linking and resolution.

The three components are used in combination in the **Pipeline** class.

We also provide the code to deploy T-Res as an API, and show how to use it.
Each of these elements are described in this documentation.

.. toctree::
   :maxdepth: 2
   :caption: Table of contents:

   getting-started/index
   reference/index
   t-res-api/index
   other-files/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
