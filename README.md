## Prerequisites
### Python3, poetry, and a virtual environment
To avoid environment-related issues, the entire set up guide should be ran in a single project-specific environment. A complete guide on setting up a Python virtual environment can be found on the [Python Docs](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv).

We use poetry to manage the Python packages and dependencies, so ensure that you have [poetry installed](https://python-poetry.org/docs/#installation).
## Local Setup

1. Create a new python virtual environment in this folder (E.g. `python3 -m venv env`)

2. Activate the virtual environment (E.g. `source env/bin/activate`)

3. Install python packages using Poetry

```bash
poetry install
```

To install any additional packages:

```bash
poetry add <package-name>
```

### Docker Setup
Do follow the docker installation instructions listed in the lectures, preferably through CLI.

If attempting to setup the docker image, the above poetry installation is not required.

To build the docker image:
```bash
make build
```

## Running Docker
To run the docker image after building (make build), use:

```bash
docker run -d -p 5000:5000 gatech_project_166
```

and then going through localhost:5000/graph should work.

To stop the docker image:

```bash
docker ps
docker stop {container_id}
```

## Viewing the Project
To view the project, go to localhost:5000/graph