# Getting started

## Platform requirements

- Python 3.x (currently tested with Python 3.8)
- a C/C++ compiler in order to compile Cython extensions

## Install from GitHub

Clone source code from GitHub :

```
git clone --recursive https://github.com/tramebleue/fct-cli
```

(Recommended)
Create and activate a python virtual environment.
You can give it whatever name you want, we use `python3`.
It will create a new `python3` (or whatever name you picked) directory :

```
python3 -m venv python3
. python3/bin/activate
```

From directory `fct-cli`,
install required dependencis using `pip` :

```
python -m pip install -r requirements.txt
```

Build extensions and install modules :

```
python -m pip install .
```

If you would like to install in developement mode (links to source code rather than copy python modules to your site-packages folder), you can use the `-e` flag :

```
python -m pip install -e .
```