====================
Deploying the T-Res API
====================

T-Res can also be deployed as a `FastAPI <https://fastapi.tiangolo.com>`_ via `Docker <https://www.docker.com>`_, 
allowing remote users to access your T-Res pipeline instead of their own local installation.

The API consists of the following files:

* ``app/app_template.py``
* ``app/configs/<CONFIG_NAME>.py``
* ``app/template.Dockerfile``
* ``docker-compose.yml``

Example configuration files are provided in this repository, which can be adapted to fit your needs.

.. toctree::
   :maxdepth: 2
   :caption: Table of contents:

   installation
   usage