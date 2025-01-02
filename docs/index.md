# T-Res: A Toponym Resolution Pipeline for Digitised Historical Newspapers

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Living-with-machines/T-Res/blob/master/LICENSE)

T-Res is an end-to-end pipeline for toponym resolution for digitised historical newspapers. Given an input text, T-Res identifies the places that are mentioned in it, links them to their corresponding Wikidata IDs, and provides their geographic coordinates. T-Res has been designed to tackle common problems of working with digitised historical newspapers.

The pipeline has three main components:

1.  **The Recogniser** performs named entity recognition.
2.  **The Ranker** performs candidate selection and ranking.
3.  **The Linker** performs entity linking and resolution.

These three components are used in combination in the **Pipeline** class.

We also provide the code to deploy T-Res as an HTTP API, and show how to use it. Each of these elements are described in this documentation.

<div class="grid cards" markdown>

-   :material-cog-outline:{ .lg .middle } __Installation & Setup__

    ---

    Install T-Res and get up and running

    [:octicons-arrow-right-24: Getting started](getting-started/index.md)

-   :material-text-box-outline:{ .lg .middle } __Reference__

    ---

    Complete reference for the T-Res codebase

    [:octicons-arrow-right-24: Reference](reference/index.md)

-   :material-swap-vertical:{ .lg .middle } __HTTP API__

    ---

    Deploy & use T-Res via an HTTP API

    [:octicons-arrow-right-24: Customization](t-res-api/index.md)

-   :material-flask-outline:{ .lg .middle } __Experiments__

    ---

    Reproduce experimental benchmarks

    [:octicons-arrow-right-24: Experiments](experiments/index.md)

</div>