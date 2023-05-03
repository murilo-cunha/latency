"""Run Stable LM locally."""
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from common.utils import (
    INSTRUCTIONS,
    Q_END,
    Q_STYLE,
    CompletionRequest,
    StabilityLM,
    build_models,
)


def main() -> None:
    """Entrypoint for running inference locally."""
    i = 0
    instruction_requests = [
        CompletionRequest(prompt=q, max_tokens=128) for q in INSTRUCTIONS
    ]

    model_dir = Path(".models/")
    torch_dtype = (
        torch.bfloat16
    )  # https://huggingface.co/databricks/dolly-v2-12b/discussions/8#64377e51369f6f907f5ceca8

    with TemporaryDirectory() as offload_dir:
        if model_dir / "model.safetensors" not in list(model_dir.rglob("*")):
            # clear dir contents and repopulate
            for file in model_dir.rglob("*"):
                if file.name not in (".gitkeep", ".gitignore"):
                    file.unlink()
            print(
                "Building models at",
                build_models(
                    repo_id="stabilityai/stablelm-tuned-alpha-3b",
                    local_dir=model_dir,
                    offload_dir=offload_dir,
                    torch_dtype=torch_dtype,
                ),
            )
        else:  # use downloaded files
            print(f"Reusing models from `{model_dir}/`")

    print("Running example non-streaming completions locally:\n")
    with TemporaryDirectory() as offload_dir, StabilityLM(
        model_url=model_dir,
        torch_dtype=torch_dtype,
        offload_dir=offload_dir,
    ) as stability_lm:
        print(
            f"{Q_STYLE}{INSTRUCTIONS[i]}{Q_END}\n{stability_lm.generate(instruction_requests[i])}\n\n",
        )


if __name__ == "__main__":
    main()
