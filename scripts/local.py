import os
from pathlib import Path
from common.utils import (
    INSTRUCTIONS,
    Q_END,
    Q_STYLE,
    CompletionRequest,
    StabilityLM,
    build_models,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from tempfile import TemporaryDirectory
import torch
from transformers import AutoTokenizer, TextIteratorStreamer, pipeline


def main():
    instruction_requests = [
        CompletionRequest(prompt=q, max_tokens=128) for q in INSTRUCTIONS
    ]
    i = 0

    MODEL_DIR = Path(".models/")
    TORCH_DTYPE = (
        torch.bfloat16
    )  # https://huggingface.co/databricks/dolly-v2-12b/discussions/8#64377e51369f6f907f5ceca8
    offload_dir = ".offload/"

    if MODEL_DIR / "model.safetensors" not in list(MODEL_DIR.rglob("*")):
        # clear dir contents
        for file in MODEL_DIR.rglob("*"):
            if file.name not in (".gitkeep", ".gitignore"):
                os.remove(file)
        # repopulate dir
        with TemporaryDirectory() as offload_dir:
            print(
                "Building models at",
                build_models(
                    repo_id="stabilityai/stablelm-tuned-alpha-3b",
                    local_dir=MODEL_DIR,
                    offload_dir=offload_dir,
                    torch_dtype=TORCH_DTYPE,
                ),
            )
    else:  # use downloaded files
        print(f"Reusing models from `{MODEL_DIR}/`")

    print("Running example non-streaming completions locally:\n")
    with StabilityLM(
        model_url=MODEL_DIR, torch_dtype=TORCH_DTYPE, offload_dir=offload_dir
    ) as stability_lm:
        print(
            f"{Q_STYLE}{INSTRUCTIONS[i]}{Q_END}\n{stability_lm.generate(instruction_requests[i])}\n\n"
        )


if __name__ == "__main__":
    main()
