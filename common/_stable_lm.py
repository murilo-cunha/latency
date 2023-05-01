import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Union
import tomli

from dataclasses import asdict, dataclass
from typing import Annotated, Literal

import modal

requirements_txt_path = Path(__file__).resolve().parents[1] / "pyproject.toml"

if requirements_txt_path.is_file():
    requirements = (
        tomli.loads(requirements_txt_path.read_text())
        .get("project", {})
        .get("optional-dependencies", {})
        .get("modal-app", {})
    )
    requirements_data = base64.b64encode(
        "\n".join(requirements).encode("utf-8")
    ).decode("utf-8")
else:
    requirements_data = ""


def build_models():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = snapshot_download(
        "stabilityai/stablelm-tuned-alpha-7b",
        ignore_patterns=["*.md"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.save_pretrained(model_path, safe_serialization=True, max_shard_size="24GB")
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.save_pretrained(model_path)

    for p in Path(model_path).rglob("*.bin"):
        p.unlink()


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


@dataclass
class CompletionRequest:
    prompt: Annotated[str, "The prompt for text completion"]
    model: Annotated[
        Literal["stabilityai/stablelm-tuned-alpha-7b"],
        "The model to use for text completion",
    ] = "stabilityai/stablelm-tuned-alpha-7b"
    temperature: Annotated[
        float,
        "Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic.",
    ] = 0.8
    max_tokens: Annotated[
        int, "Maximum number of new tokens to generate for text completion."
    ] = 16
    top_p: Annotated[
        float,
        "Probability threshold for the decoder to use in sampling next most likely token.",
    ] = 0.9
    stream: Annotated[
        bool, "Whether to stream the generated text or return it all at once."
    ] = False
    stop: Annotated[Union[str, Sequence[str]], "Any additional stop words."] = ()
    top_k: Annotated[
        int,
        "Limits the set of tokens to consider for next token generation to the top k.",
    ] = 40
    do_sample: Annotated[
        bool, "Whether to use sampling or greedy decoding for text completion."
    ] = True


@stub.cls(gpu="A10G")
class StabilityLM:
    def __init__(
        self,
        model_url: str = "stabilityai/stablelm-tuned-alpha-7b",
        decode_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_url = model_url
        self.decode_kwargs = decode_kwargs or {}
        self.stop_tokens = [
            "<|USER|>",
            "<|ASSISTANT|>",
            "<|SYSTEM|>",
            "<|padding|>",
            "<|endoftext|>",
        ]
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def __enter__(self):
        """
        Container-lifeycle method for model setup.
        """
        import torch
        from transformers import AutoTokenizer, TextIteratorStreamer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(self.model_url, local_files_only=True)
        self.stop_ids = tokenizer.convert_tokens_to_ids(self.stop_tokens)
        self.streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, **self.decode_kwargs
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model_url,
            tokenizer=tokenizer,
            streamer=self.streamer,
            torch_dtype=torch.float16,
            device_map="auto",
            model_kwargs={"local_files_only": True},
        )
        self.generator.model = torch.compile(self.generator.model)

    def __exit__(self, *args, **kwargs):
        ...

    def get_config(self, completion_request: CompletionRequest) -> Dict[str, Any]:
        return dict(
            pad_token_id=self.generator.tokenizer.eos_token_id,
            eos_token_id=list(
                set(
                    self.generator.tokenizer.convert_tokens_to_ids(
                        self.generator.tokenizer.tokenize(
                            "".join(completion_request.stop)
                        )
                    )
                    + self.stop_ids
                )
            ),
            max_new_tokens=completion_request.max_tokens,
            **{
                k: v
                for k, v in asdict(completion_request).items()
                if k not in ("prompt", "model", "stop", "max_tokens", "stream")
            },
        )

    def generate_completion(
        self, completion_request: CompletionRequest
    ) -> Generator[str, None, None]:
        import re
        from threading import Thread

        from transformers import GenerationConfig

        text = format_prompt(completion_request.prompt)
        gen_config = GenerationConfig(**self.get_config(completion_request))
        stop_words = self.generator.tokenizer.convert_ids_to_tokens(
            gen_config.eos_token_id
        )
        stop_words_pattern = re.compile("|".join(map(re.escape, stop_words)))
        thread = Thread(
            target=self.generator.__call__,
            kwargs=dict(text_inputs=text, generation_config=gen_config),
        )
        thread.start()
        for new_text in self.streamer:
            if new_text.strip():
                yield stop_words_pattern.sub("", new_text)
        thread.join()

    @modal.method()
    def generate(self, completion_request: CompletionRequest) -> str:
        return "".join(self.generate_completion(completion_request))


def format_prompt(instruction: str) -> str:
    return f"<|USER|>{instruction}<|ASSISTANT|>"


@stub.local_entrypoint()
def main():
    q_style, q_end = "\033[1m", "\033[0m"
    instructions = [
        "Generate a list of the 10 most beautiful cities in the world.",
        "How can I tell apart female and male red cardinals?",
    ]
    instruction_requests = [
        CompletionRequest(prompt=q, max_tokens=128) for q in instructions
    ]
    print("Running example non-streaming completions:\n")
    for q, a in zip(
        instructions, list(StabilityLM().generate.map(instruction_requests))
    ):
        print(f"{q_style}{q}{q_end}\n{a}\n\n")
    # build_models()
    # i = 0
    # with StabilityLM() as stability_lm:
    #     print(
    #         f"{q_style}{instructions[i]}{q_end}\n{stability_lm.generate(instruction_requests[i])}\n\n"
    #     )


# """
# # TODO:
# - test local script
# - check that we don't need pydantic
# - replace pydantic with dataclass
# - remove pydantic and fastapi deps form pyproject.toml
# - check that local and remote scripts are working with utils
# - add docstrings and documentation (ruff lint)
# - add scripts to pdm (lint, hooks)
# - add pre-commit hooks and make sure they are compliant
# - add README documentation
# - add CICD for linting and tagging
# """
