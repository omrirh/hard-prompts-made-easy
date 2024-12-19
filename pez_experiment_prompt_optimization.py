from optim_utils import *
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

GSM8K_EXAMPLE_PROMPT = """
Amaya scored 20 marks fewer in Maths than she scored in Arts.
She also got 10 marks more in Social Studies than she got in Music.
If she scored 70 in Music and scored 1/10 less in Maths,
what's the total number of marks she scored in all the subjects?
"""
HOTPOTQA_EXAMPLE_PROMPT = """
"""
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def main():
    optim_args = {
        "prompt_len": len(GSM8K_EXAMPLE_PROMPT),
        "iter": 3000,
        "lr": 0.1,
        "weight_decay": 0.1,
        "prompt_bs": 1,
        "loss_weight": 1.0,
        "print_step": 100,
        "batch_size": 1
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    optimized_example = optimize_prompt(
        model=model,
        tokenizer=tokenizer,
        args=optim_args,
        device=torch.device("cuda"),
        target_prompts=[GSM8K_EXAMPLE_PROMPT]
    )

    print(f"Original Prompt:"
          f"----------------"
          f"{GSM8K_EXAMPLE_PROMPT}\n\n")

    print(f"Optimized Prompt:"
          f"-----------------"
          f"{optimized_example}")


if __name__ == "__main__":
    main()
