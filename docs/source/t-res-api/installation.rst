=======================
Deploying the T-Res API
=======================

The T-Res API can be deployed either as a standalone docker container, 
or via docker compose to deploy multiple configurations of the pipeline simultaneously behind a reverse-proxy (`traefik <https://traefik.io/traefik/>`).

Docker and Docker Compose should be installed on your server according to the `official installation guide <https://docs.docker.com/engine/install/ubuntu/>` 
before proceeding with the following steps to build and deploy the containers.

A bash script `builder.sh` has been included in the repository to conveniently (re-)deploy the example API:
.. code-block:: bash
    ./builder.sh 


1. Building the container
--------------

.. code-block:: bash
    export CONTAINER_NAME=t-res_deezy_reldisamb-wpubl-wmtops
    sudo -E docker build -f app/template.Dockerfile --no-cache --build-arg APP_NAME=${CONTAINER_NAME} -t ${CONTAINER_NAME}_image .


2. Deploying the container
--------------

.. code-block:: bash
    sudo docker run -p 80:80 -it \
     -v ${HOME}/toponym-resolution/resources/:/app/resources/ \
     -v ${HOME}/toponym-resolution/geoparser/:/app/geoparser/ \
     -v ${HOME}/toponym-resolution/utils/:/app/utils/ \
     -v ${HOME}/toponym-resolution/preprocessing/:/app/preprocessing/ \
     -v ${HOME}/toponym-resolution/experiments/:/app/experiments/ \
     -v ${HOME}/nltk_data:/root/nltk_data/ \
     -v ${HOME}/toponym-resolution/app/configs/:/app/configs/ \
     ${CONTAINER_NAME}_image:latest


3. Deploying multiple containers via Docker Compose
--------------
To deploy the example configuration behind a traefik load-balancing server:
.. code-block:: bash
    HOST_URL=20.0.184.45 sudo -E docker-compose up -d

4. Configuring your deployment
--------------

1. Add your T-Res pipeline configuration to the `app/config` directory (see `Adding a new pipeline configuration`)
2. Optionally, you can add or edit endpoints or app behaviour in the `app/app_template.py` file
3. Build your docker container as in step 1, setting the CONTAINER_NAME environment variable to your new configuration's name
4. Add a section to the docker-compose.yml, updating the service name, image and labels as follows:
    .. code-block:: yaml
        <YOUR_CONFIG_NAME>:
            image: <YOUR_CONFIG_NAME>_image:latest
            restart: always
            expose:
            - 80
            volumes:
            - ${HOME}/toponym-resolution/resources/:/app/resources/
            - ${HOME}/toponym-resolution/geoparser/:/app/geoparser/
            - ${HOME}/toponym-resolution/utils/:/app/utils/
            - /${HOME}/toponym-resolution/preprocessing/:/app/preprocessing/
            - ${HOME}/toponym-resolution/experiments/:/app/experiments/
            - ${HOME}/nltk_data:/root/nltk_data/
            labels:
            - traefik.enable=true
            - traefik.http.services.<YOUR_CONFIG_NAME>.loadbalancer.server.port=80
            - traefik.http.routers.<YOUR_CONFIG_NAME>_router.service=<YOUR_CONFIG_NAME>
            - traefik.http.routers.<YOUR_CONFIG_NAME>_router.rule=Host(`<YOUR_HOST_URL>`, `0.0.0.0`) && PathPrefix(`/v2/t-res_<YOUR_CONFIG_NAME>`)
            - "traefik.http.middlewares.test-stripprefix-rwop.stripprefix.prefixes=/v2/t-res_<YOUR_CONFIG_NAME>"
            - traefik.http.routers.<YOUR_CONFIG_NAME>_router.middlewares=test-stripprefix-rwop
            command: ["poetry", "run", "uvicorn", "app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]

