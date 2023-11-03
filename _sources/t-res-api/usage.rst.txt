=======================
Using the T-Res API
=======================

If you deploy the T-Res API according to the steps in the previous section, 
it should now be available on your server as a HTTP API 
(be sure to expose the correct ports - by default, the app is deployed to port 8000). 
Automatically generated, interactive documentation (created by `Swagger`) is available at the ``/docs`` endpoint.

The following example shows how to query the API via curl to resolve the toponyms in a single sentence:

.. code-block:: bash
    
    curl -X GET http://20.0.184.45:8000/v2/t-res_deezy_reldisamb-wpubl-wmtops/toponym_resolution \
    -H "Content-Type: application/json" \
    -d '{"text": "Harvey, from London;Thomas and Elizabeth, Barnett.", "place": "Manchester", "place_wqid": "Q18125"}'

See the ``app/api_usage.ipynb`` notebook for more examples of how to use the API's various endpoints via Python.