=======================
Using the T-Res API
=======================

If you deploy the T-Res API according to the steps in the previous section, it should now be available on your server as a HTTP API (be sure to expose the correct ports).
Swagger documentation will automatically be deployed to port XXXX.

The following example shows how to query the API via curl to resolve the toponyms in a single sentence:
.. code-block:: bash
    curl ...

See the `app/api_usage.ipynb` notebook for more examples of how to use the API's various endpoints via Python.