import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optim_utils import optimize_prompt

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
        "prompt_len": 16,
        "iter": 500, # 3000, --> quick experimential run
        "lr": 0.01,  # For stabilizing loss values diverge to NaN
        "weight_decay": 0.01,
        "prompt_bs": 1,
        "loss_weight": 1.0,
        "print_step": 100,
        "batch_size": 1
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )

    optimized_example = optimize_prompt(
        model=model,
        tokenizer=tokenizer,
        args=optim_args,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        target_prompts=[GSM8K_EXAMPLE_PROMPT]
    )

    print(f"Original Prompt:\n----------------\n{GSM8K_EXAMPLE_PROMPT}\n")
    print(f"Optimized Prompt:\n-----------------\n{optimized_example}")

if __name__ == "__main__":
    main()
