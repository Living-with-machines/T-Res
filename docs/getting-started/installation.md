# Installing T-Res

If you want to work directly on the codebase, we suggest to install T-Res following these instructions (which have been tested Linux (ubuntu 20.04)).

## Update the system

First, you need to make sure the system is up to date and all essential libraries are installed.

``` bash
$ sudo apt update
$ sudo apt install build-essential curl libbz2-dev libffi-dev \
  liblzma-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
  libssl-dev libxml2-dev libxmlsec1-dev llvm make tk-dev wget \
  xz-utils zlib1g-dev
```

## Install pyenv

Then you need to install pyenv, which we use to manage virtual environments:

``` bash
$ curl https://pyenv.run | bash
```

And also to make sure paths are properly exported:

``` bash
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
$ echo -e 'if command -v pyenv 1>/dev/null 2>&1; \
  then\n eval "$(pyenv init --path)"\nfi' >> ~/.bashrc
```

Then you can restart your bash session, to make sure all changes are updated:

``` bash
$ source ~/.bashrc
```

And then you run the following commands to update ``pyenv`` and create the needed environemnt.

``` bash
$ pyenv update
$ pyenv install 3.9.7
$ pyenv global 3.9.7
```

## Install poetry

To manage dipendencies across libraries, we use Poetry. To install it, do the following:

``` bash
$ curl -sSL https://install.python-poetry.org | python3 -
$ echo 'export PATH=$PATH:$HOME/.poetry/bin' >> ~/.bashrc
```

## Project Installation

You can now clone the repo and ``cd`` into it:

``` bash
$ git clone git@github.com:Living-with-machines/T-Res.git
$ cd T-Res
```

Explicitly tell poetry to use the python version defined above:

``` bash
$ poetry env use python
```

Install all dependencies using `poetry`:

``` bash
$ poetry update
$ poetry install
```

Create a kernel:

``` bash
$ poetry run ipython kernel install --user --name=<KERNEL_NAME>
```

## How to use poetry

To activate the environment:

``` bash
$ poetry shell
```

Now you can run a script as usual, for instance :

``` bash
$ python experiments/toponym_resolution.py
```

To add a package:

``` bash
$ poetry add [package name]
```

To run the Python unit tests:

``` bash
$ poetry run pytest
```

To run unit and integration tests, some of which depend on the [T-Res resources](../getting-started/resources.html):

``` bash
$ poetry run pytest tests --no-skip
```

If you want to use Jupyter notebook, run it as usual, and then select the created kernel in "Kernel" > "Change kernel".

``` bash
$ jupyter notebook
```

## Pre-commit hoooks

In order to guarantee style consistency across our codebase we use a few basic pre-commit hooks.

To use them, first run:

``` bash
$ poetry run pre-commit install --install-hooks
```

To run the hooks on all files, you can do:

``` bash
$ poetry run pre-commit run --all-files
```
