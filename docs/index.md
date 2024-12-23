# T-Res: A Toponym Resolution Pipeline for Digitised Historical Newspapers

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Living-with-machines/T-Res/blob/master/LICENSE)

T-Res is an end-to-end pipeline for toponym resolution for digitised historical newspapers. Given an input text, T-Res identifies the places that are mentioned in it, links them to their corresponding Wikidata IDs, and provides their geographic coordinates. T-Res has been designed to tackle common problems of working with digitised historical newspapers.

The pipeline has three main components:

1.  **The Recogniser** performs named entity recognition.
2.  **The Ranker** performs candidate selection and ranking.
3.  **The Linker** performs entity linking and resolution.

The three components are used in combination in the **Pipeline** class.

We also provide the code to deploy T-Res as an API, and show how to use it. Each of these elements are described in this documentation.

<div class="grid cards" markdown>

-   [Getting started](getting-started/index.md)
-   [Reference](reference/index.md)
-   [T Res Api](t-res-api/index.md)
-   [Experiments and evaluation](experiments/index.md)

</div>

## Indices and tables

-   [Genindex](genindex-broken.md)
-   [Modindex](modindex-broken.md)
-   [Search](search-broken.md)
