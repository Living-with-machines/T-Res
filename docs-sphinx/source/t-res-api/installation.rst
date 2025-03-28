=======================
Deploying the T-Res API
=======================

The T-Res API can be deployed either as a standalone docker container, 
or via docker compose to deploy multiple configurations of the pipeline simultaneously behind 
a reverse-proxy (`traefik <https://traefik.io/traefik/>`_).

Docker and Docker Compose should be installed on your server according to 
the `official installation guide <https://docs.docker.com/engine/install/ubuntu/>`_ 
before proceeding with the following steps to build and deploy the containers.

..
    A bash script `builder.sh` has been included in the repository to conveniently (re-)deploy the example API:
    .. code-block:: bash
        ./builder.sh 


1. Building the container
--------------

To build a docker image for the app using the default configuration provided (``t-res_deezy_reldisamb-wpubl-wmtops.py``), 
run the following bash commands from the root of the repository:  

.. code-block:: bash

    export CONTAINER_NAME=t-res_deezy_reldisamb-wpubl-wmtops
    sudo -E docker build -f app/template.Dockerfile --no-cache --build-arg APP_NAME=${CONTAINER_NAME} -t ${CONTAINER_NAME}_image .

2. Deploying the container
--------------

The docker image built in step 1 can then be deployed by running the following command, providing the required 
resources are available according to the :ref: `Resources and directory structure <../getting-started/resources.html>`_ section.

.. code-block:: bash

    sudo docker run -p 80:80 -it \
     -v ${HOME}/T-Res/resources/:/app/resources/ \
     -v ${HOME}/T-Res/geoparser/:/app/geoparser/ \
     -v ${HOME}/T-Res/utils/:/app/utils/ \
     -v ${HOME}/T-Res/preprocessing/:/app/preprocessing/ \
     -v ${HOME}/T-Res/experiments/:/app/experiments/ \
     -v ${HOME}/T-Res/app/configs/:/app/configs/ \
     ${CONTAINER_NAME}_image:latest


3. Deploying multiple containers via Docker Compose
--------------
To deploy the example configuration behind a traefik load-balancing server:

.. code-block:: bash

    HOST_URL=<YOUR_HOST_URL> sudo -E docker-compose up -d

4. Configuring your deployment
--------------

1. Add your T-Res pipeline configuration file to the ``app/config`` directory. This file should instantiate the ``Recogniser``, ``Linker``, and ``Ranker`` to be used in your pipeline and store them in a dictionary called ``CONFIG``, which is then imported and used by the app.
2. Optionally, you can add or edit endpoints or app behaviour in the ``app/app_template.py`` file
3. Build your docker container as in step 1, setting the ``CONTAINER_NAME`` environment variable to your new configuration's name
4. Add a section to the docker-compose.yml, updating the service name, image and labels as follows:

    .. code-block:: yaml
    
        <YOUR_CONFIG_NAME>:
            image: <YOUR_CONFIG_NAME>_image:latest
            restart: always
            expose:
            - 80
            volumes:
            - ${HOME}/T-Res/resources/:/app/resources/
            - ${HOME}/T-Res/geoparser/:/app/geoparser/
            - ${HOME}/T-Res/utils/:/app/utils/
            - ${HOME}/T-Res/preprocessing/:/app/preprocessing/
            - ${HOME}/T-Res/experiments/:/app/experiments/
            labels:
            - traefik.enable=true
            - traefik.http.services.<YOUR_CONFIG_NAME>.loadbalancer.server.port=80
            - traefik.http.routers.<YOUR_CONFIG_NAME>_router.service=<YOUR_CONFIG_NAME>
            - traefik.http.routers.<YOUR_CONFIG_NAME>_router.rule=Host(`<YOUR_HOST_URL>`, `0.0.0.0`) && PathPrefix(`/v2/t-res_<YOUR_CONFIG_NAME>`)
            - traefik.http.middlewares.test-stripprefix-rwop.stripprefix.prefixes=/v2/t-res_<YOUR_CONFIG_NAME>
            - traefik.http.routers.<YOUR_CONFIG_NAME>_router.middlewares=test-stripprefix-rwop
            command: ["poetry", "run", "uvicorn", "app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--root-path", "/v2/t-res_deezy_reldisamb-wpubl-wmtops"]

