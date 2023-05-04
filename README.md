# Latency in ML models

> A demo on how LLMs performance is impacted by the use of GPUs.

Based off example from [`modal-example`](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/stable_lm)


## Get started

The project was developed using [PDM](https://pdm.fming.dev/latest/). It's the recommended way to get the project setup. See below for how to install PDM.

**Make sure you have Python 3.9 installed.**

```console
git clone git@github.com:murilo-cunha/latency.git
cd latency/
pdm install
```

PDM will take care of installing the dependencies and virtual environments.

Alternatively, we've exported a `requirements.txt` file for convenience. Using Python's built in `venv`:

```console
git clone git@github.com:murilo-cunha/latency.git
cd latency/
python --version  # make sure you are using python 3.9
python -m venv .venv
source .venv/bin/activate  # alternatively, `.venv\Scripts\activate.bat` if you're on Windows
pip install -r requirements.txt
```

### Running the demos

If you have set up using PDM:

```console
pdm run local # local execution
pdm run remote # cloud/remote execution
```

If using `venv`:

```console
source .venv/bin/activate  # alternatively, `.venv\Scripts\activate.bat` if you're on Windows
python scripts/local.py  # local execution
modal run scripts/remote.py  # cloud/remote execution
```

### Installing PDM

Easiest way is to install using `pipx`. See their [docs](https://pdm.fming.dev/latest/#recommended-installation-method) for other installation methods. See `pipx` [docs](https://pypa.github.io/pipx/installation/) for more info about `pipx` and how to install it.

```console
pipx install pdm==2.5.3
```


## Project structure

```console
.
├── .gitignore
├── .models/
├── .pre-commit-config.yaml
├── README.md
├── common/
│   ├── __init__.py
│   └── utils.py
├── pdm.lock
├── pyproject.toml
└── scripts/
    ├── local.py
    └── remote.py
```

Directories:

- `.models/` - an (empty) directory to hold LLM models
- `common/` - a local python package with the common building blocks used both for local and remote execution
- `scripts/` - entrypoint scripts; demos are ran from these scripts

Files:

- `.gitignore` - list of patterns to not be committed to the repo
- `.pre-commit-config.yaml` - list of pre-commit hooks
- `README.md` - general project information
- `pdm.lock` - PDM's lockfile with dependency tree and versions
- `pyproject.toml` - definition of project; includes dependencies, configuration and scripts

## Modal

The demo uses [Modal](https://modal.com/) for GPU use. Modal is a simple cloud service for creating serverless applications.

Signing up to Modal is as easy as linking your GitHub account. After a few hours you should receive a notification about you account being created. Once the account is created, create a token using

```console
model token new
```

### Why Modal?

Modal is free to sign up, not requiring any credit card. Once signed up, it offers a free tier of $30/month. The current month's usage is transparently shown in under [`Settings` > `Usage and Billing`](https://modal.com/settings/usage).

It's also a Python-only API to define your dependencies, including container dependencies, replacing the need for `Dockerfiles`, etc. It offers different serverless applications, including functions, schedulers or REST APIs. Check our their [docs](https://modal.com/docs/guide) and [examples](https://modal.com/examples) to see more.

## What else?

If you've ran the demo and would like to get your hands dirty, I'd recommend:

- [ ]: Go over Modal's [starting examples](https://modal.com/docs/guide/ex/hello_world) to better understand how it works
- [ ]: Modify the cloud demo to create a FastAPI REST endpoint to have a **true** experience on "latency in model serving" (via REST API)
  - See the original [example](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/stable_lm) for more info
  - See the [web endpoints documentation](https://modal.com/docs/guide/webhooks)

You should be able to call your REST application running:

```bash
curl $MODEL_APP_ENDPOINT \    # specify your endpoint here
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Generate a list of 20 great names for sentient cheesecakes that teach SQL",
    "stream": false,
    "max_tokens": 64
  }'
```
