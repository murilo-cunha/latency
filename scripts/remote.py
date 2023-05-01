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
    print("Running example non-streaming completions:\n")
    for q, a in zip(
        INSTRUCTIONS, list(StabilityLM().generate.map(INSTRUCTION_REQUESTS))
    ):
        print(f"{Q_STYLE}{q}{Q_END}\n{a}\n\n")

    print("Running example streaming completion:\n")
    for part in StabilityLM().generate_stream.call(
        CompletionRequest(
            prompt="Generate a list of ten sure-to-be unicorn AI startup names.",
            max_tokens=128,
            stream=True,
        )
    ):
        print(part, end="", flush=True)
