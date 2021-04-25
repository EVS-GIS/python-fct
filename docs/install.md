# Getting started

## Platform requirements

- Python 3.x (currently tested with Python 3.8)
- a C/C++ compiler in order to compile Cython extensions

## Install from GitHub

Clone source code from GitHub :

```
git clone --recursive https://github.com/tramebleue/fct-cli
```

Create and activate a python virtual environment.
We recommend not to create the python virtual environment inside the `fct-cli` directory. For example, you can create a `pyenv` directory in your home directory to store all your python virtual environments.

You can give your new virtual environment whatever name you want.
For example, we use `fct`.

It will create a new `fct` directory :

```
mkdir pyenv
cd pyenv
python3 -m venv fct
. fct/bin/activate
```

From directory `fct-cli`,
install required dependencis using `pip` :

```
python -m pip install -r requirements/fct.txt
```

Build extensions and install modules :

```
python -m pip install .
```

If you would like to install in developement mode (links to source code rather than copy python modules to your site-packages folder), you can use the `-e` flag :

```
python -m pip install -e .
```