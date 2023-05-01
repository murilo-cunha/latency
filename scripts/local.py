from pathlib import Path
from common.utils import (
    INSTRUCTIONS,
    Q_END,
    Q_STYLE,
    CompletionRequest,
    StabilityLM,
    build_models,
)


def main():
    instruction_requests = [
        CompletionRequest(prompt=q, max_tokens=128) for q in INSTRUCTIONS
    ]
    i = 0
    
    models_dir = Path(".models/")
    
    if next(models_dir.iterdir(), None) is None:  # empty
        print("Building models at", build_models(local_dir=models_dir))
    else:
        print(f"Reusing models from {models_dir}")
    
    print("Running example non-streaming completions:\n")
    with StabilityLM() as stability_lm:
        print(
            f"{Q_STYLE}{INSTRUCTIONS[i]}{Q_END}\n{stability_lm.generate(instruction_requests[i])}\n\n"
        )
