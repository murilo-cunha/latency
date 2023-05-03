"""Common utility functions and classes for both remote and local runs."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    import torch


Q_STYLE, Q_END = "\033[1m", "\033[0m"
INSTRUCTIONS = [
    "Generate a list of the 10 most beautiful cities in the world.",
    "How can I tell apart female and male red cardinals?",
]


@dataclass
class CompletionRequest:
    """A completion request with reasonable defaults."""

    prompt: Annotated[str, "The prompt for text completion"]
    model: Annotated[
        Literal["stabilityai/stablelm-tuned-alpha-7b"],
        "The model to use for text completion",
    ] = "stabilityai/stablelm-tuned-alpha-7b"
    temperature: Annotated[
        float,
        "Adjusts randomness of outputs, >1 is random and 0 is deterministic.",
    ] = 0.8
    max_tokens: Annotated[
        int,
        "Maximum number of new tokens to generate for text completion.",
    ] = 16
    top_p: Annotated[
        float,
        "Decoder probability threshold in sampling next most likely token.",
    ] = 0.9
    stream: Annotated[
        bool,
        "Whether to stream the generated text or return it all at once.",
    ] = False
    stop: Annotated[str | Sequence[str], "Any additional stop words."] = ()
    top_k: Annotated[
        int,
        "Limits the set of tokens to consider for next token generation to the top k.",
    ] = 40
    do_sample: Annotated[
        bool,
        "Whether to use sampling or greedy decoding for text completion.",
    ] = True


class StabilityLM:
    """Stable LM model container - load weights and inference."""

    def __init__(
        self: StabilityLM,
        model_url: str = "stabilityai/stablelm-tuned-alpha-7b",
        decode_kwargs: dict[str, Any] | None = None,
        torch_dtype: torch.dtype | None = None,
        offload_dir: str = ".tmp",
    ) -> None:
        """Load tokens and setup Huggingface for offline use."""
        self.model_url = model_url
        self.decode_kwargs = decode_kwargs or {}
        self.torch_dtype = torch_dtype
        self.offload_dir = offload_dir
        self.stop_tokens = [
            "<|USER|>",
            "<|ASSISTANT|>",
            "<|SYSTEM|>",
            "<|padding|>",
            "<|endoftext|>",
        ]
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def __enter__(self: StabilityLM) -> StabilityLM:
        """Container-lifeycle method for model setup."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TextIteratorStreamer,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_url, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_url,
            torch_dtype=self.torch_dtype or torch.float16,
            device_map="auto",
            local_files_only=True,
            offload_folder=self.offload_dir,
        )

        self.stop_ids = tokenizer.convert_tokens_to_ids(self.stop_tokens)
        self.streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            **self.decode_kwargs,
        )
        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            streamer=self.streamer,
            torch_dtype=self.torch_dtype or torch.float16,
            device_map="auto",
            model_kwargs={"local_files_only": True},
        )
        self.generator.model = torch.compile(self.generator.model)
        return self

    def __exit__(
        self: StabilityLM,
        *_args: Any,  # noqa: ANN401
        **_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Allow using class as a context manager."""
        ...

    def get_config(
        self: StabilityLM,
        completion_request: CompletionRequest,
    ) -> dict[str, Any]:
        """Get the LLM configuration - token IDs, etc."""
        return dict(
            pad_token_id=self.generator.tokenizer.eos_token_id,
            eos_token_id=list(
                set(
                    self.generator.tokenizer.convert_tokens_to_ids(
                        self.generator.tokenizer.tokenize(
                            "".join(completion_request.stop),
                        ),
                    )
                    + self.stop_ids,
                ),
            ),
            max_new_tokens=completion_request.max_tokens,
            **{
                k: v
                for k, v in asdict(completion_request).items()
                if k not in ("prompt", "model", "stop", "max_tokens", "stream")
            },
        )

    def generate_completion(
        self: StabilityLM,
        completion_request: CompletionRequest,
    ) -> Generator[str, None, None]:
        """Generate string completion with LLM."""
        import re
        from threading import Thread

        from transformers import GenerationConfig

        text = self._fmt_prompt(completion_request.prompt)
        gen_config = GenerationConfig(**self.get_config(completion_request))
        stop_words = self.generator.tokenizer.convert_ids_to_tokens(
            gen_config.eos_token_id,
        )
        stop_words_pattern = re.compile("|".join(map(re.escape, stop_words)))
        thread = Thread(
            target=self.generator.__call__,
            kwargs={"text_inputs": text, "generation_config": gen_config},
        )
        thread.start()
        for new_text in self.streamer:
            if new_text.strip():
                yield stop_words_pattern.sub("", new_text)
        thread.join()

    def generate(self: StabilityLM, completion_request: CompletionRequest) -> str:
        """Generate completion as a string."""
        return "".join(self.generate_completion(completion_request))

    @staticmethod
    def _fmt_prompt(instruction: str) -> str:
        """Format string with promt tokens."""
        return f"<|USER|>{instruction}<|ASSISTANT|>"


def build_models(
    offload_dir: Path | None = None,
    save_model_max_shard_size: str | None = "24GB",
    torch_dtype: torch.dtype | None = None,
    **download_kwargs: Any,  # noqa: ANN401
) -> str:
    """Download artifacts from repo, build and save model."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    download_kwargs["ignore_patterns"] = download_kwargs.get("ignore_patterns") or [
        "*.md",
    ]
    download_kwargs["repo_id"] = (
        download_kwargs.get("repo_id") or "stabilityai/stablelm-tuned-alpha-7b"
    )
    model_path = snapshot_download(**download_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype or torch.float16,  # if `None`
        device_map="auto",
        local_files_only=True,
        offload_folder=str(offload_dir),
    )
    model.save_pretrained(
        model_path,
        safe_serialization=True,
        max_shard_size=save_model_max_shard_size,
    )
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.save_pretrained(model_path)

    for p in Path(model_path).rglob("*.bin"):
        p.unlink()
    return model_path
