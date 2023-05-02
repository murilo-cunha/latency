from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights


def main(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        offload_folder=".offload/"
    )
    model.save_pretrained(model_path, safe_serialization=True, max_shard_size="10GB")
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.save_pretrained(model_path)

    for p in Path(model_path).rglob("*.bin"):
        p.unlink()


def conf(model_dir):
    config = AutoConfig.from_pretrained(model_dir)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    device_map = infer_auto_device_map(model, dtype="float16")
    from pprint import pprint

    pprint(device_map)


if __name__ == "__main__":
    main(Path(".models/"))
