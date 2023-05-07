# Development

## Install PanQEC from the repository for development

Clone the repo and enter the repo
```
git clone https://github.com/ehua7365/panqec.git
cd panqec
```

Within a Python 3.8+ environment, install the library and its dependencies for development mode.
```
pip install -r requirements/dev.txt
pip install -e .
```

## Run the tests

After you've installed the dependences,
you can run the tests to make sure everything has been correctly installed.

```
pytest
```
If all the tests pass everything should be working as expected.

There are also some more comprehensive tests which are very slow to run.
To also run those, use
```
pytest --runslow
```

## Check the code Style

Just follow PEP 8, but to actually test that the code is compliant, run the
linter using
```
make lint
```
It will run `flake8` and print out a list of non-compliances.
It will also run `mypy` to do type checking.

## Build the documentation

Within the root directory of the repository, run
```
make doc
```
You can then read it by running the command
```
python -m http.server 9000 --directory docs/_build/html
```
and going to `localhost:9000` in your browser.
Alternatively, you can just open `docs/_build/html/index.html` with your
browser.

## Submit PR
When everything is ready, just submit the PR on the github repo.
