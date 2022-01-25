# Toponym Resolution

## Environment installation

The setup relies on the integration of `pyenv` and `poetry`. The following are the commands you should run if you are setting this up on Ubuntu (to know more, see [these guidelines](https://www.adaltas.com/en/2021/06/09/pyrepo-project-initialization/)). To install them on a different OS, you can follow [these guidelines](https://github.com/pyenv/pyenv#installation) for `pyenv` (but remember to first of all install the prerequisites listed in the link!) and then [these guidelines](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions) for `poetry`.

If you haven't already `pyenv` and `poetry` installed, first you need to ensure the following packages are installed:

```
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl
```

Then you can install `pyenv` with the `pyenv-installer`:

```
curl https://pyenv.run | bash
```
Then to properly configure pyenv for use on the system, you need:

```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init --path)"\nfi' >> ~/.bashrc
```

Restart the terminal and check that `pyenv` is correctly installed by typing:
```
pyenv
```

Install Python 3.9.7 and set it as the global Python version:

```
pyenv install 3.9.7
pyenv global 3.9.7
```

Now you can install `poetry` the following way:

```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

## Project Installation

You can now clone the repo and `cd` into it:

```
git clone https://github.com/Living-with-machines/toponym-resolution.git
cd toponym-resolution
```

Install all dependencies using `poetry`:

```
poetry install
```

### How to use poetry

To activate the environment:

```
poetry shell
```

Now you can run a script as usual:

```
python processing.py
```

Add a package:

```
poetry add [package name]
```

Run the Python tests:

```
poetry run pytest
```


