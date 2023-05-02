import base64
from pathlib import Path
import modal
import tomli

from common.utils import (
    build_models,
    StabilityLM,
    CompletionRequest,
    Q_STYLE,
    Q_END,
    INSTRUCTIONS,
)


requirements_txt_path = Path(__file__).resolve().parents[1] / "pyproject.toml"

requirements = (
    (
        tomli.loads(requirements_txt_path.read_text())
        .get("project", {})
        .get("optional-dependencies", {})
        .get("modal-app", [])
    )
    if requirements_txt_path.is_file()
    else []
)
requirements_data = base64.b64encode("\n".join(requirements).encode("utf-8")).decode(
    "utf-8"
)


image = (
    modal.Image.conda()
    .apt_install("git", "software-properties-common", "wget")
    .conda_install(
        "cudatoolkit-dev=11.7",
        "pytorch-cuda=11.7",
        "rust=1.69.0",
        channels=["nvidia", "pytorch", "conda-forge"],
    )
    .env(
        {
            "HF_HOME": "/root",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "SAFETENSORS_FAST_GPU": "1",
            "BITSANDBYTES_NOWELCOME": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
        }
    )
    .run_commands(
        f"echo '{requirements_data}' | base64 --decode > /root/requirements-modal.txt",
        "pip install -r /root/requirements-modal.txt",
        gpu="any",
    )
    .run_function(
        build_models,
        gpu=None,
        timeout=3600,
    )
)


stub = modal.Stub(
    name="example-stability-lm",
    image=image,
    secrets=[modal.Secret({"REPO_ID": "stabilityai/stablelm-tuned-alpha-7b"})],
)


@stub.cls(
    gpu="A10G",
    # mounts=modal.create_package_mounts(["common"]),
)
class ModalStabilityLM(StabilityLM):
    @modal.method()
    def generate(self, *args, **kwargs):
        return super().generate(*args, **kwargs)


@stub.local_entrypoint()
def main():
    # `pdm run modal run scripts/remote.py
    instruction_requests = [
        CompletionRequest(prompt=q, max_tokens=128) for q in INSTRUCTIONS
    ]

    print("Running distributed example on cloud:\n")
    for q, a in zip(
        INSTRUCTIONS, list(ModalStabilityLM().generate.map(instruction_requests))
    ):
        print(f"{Q_STYLE}{q}{Q_END}\n{a}\n\n")
