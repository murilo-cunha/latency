from common.utils import (
    INSTRUCTION_REQUESTS,
    INSTRUCTIONS,
    Q_END,
    Q_STYLE,
    CompletionRequest,
    StabilityLM,
    stub,
)


@stub.local_entrypoint()
def main():
    i = 0

    print("Running example non-streaming completions:\n")
    print(
        f"{Q_STYLE}{INSTRUCTIONS[i]}{Q_END}\n{StabilityLM().generate(INSTRUCTION_REQUESTS[i])}\n\n"
    )

    print("Running example streaming completion:\n")
    for part in StabilityLM().generate_stream(
        CompletionRequest(
            prompt="Generate a list of ten sure-to-be unicorn AI startup names.",
            max_tokens=128,
            stream=True,
        )
    ):
        print(part, end="", flush=True)
