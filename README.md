<div align="center">

<h1>CEARTIN Machine Project</h1>

</div>

This serves as the main Git repository for a Filipino NLP project with RoBERTa.

## Setting up

### Installing dependencies

This project uses [`uv`](https://docs.astral.sh/uv/) to manage packages.

1. Create a Python 3.12 virtual environment with `uv venv --python 3.12`.
2. Run `uv sync` to install dependencies.

### Set up `pre-commit`

1. `pre-commit` should have been installed as a development dependency. Check
   with `pre-commit --version`.
2. Install the hook scripts with `pre-commit install`.
3. Run `pre-commit run --all-files` to run the pre-commit hooks on all files.
