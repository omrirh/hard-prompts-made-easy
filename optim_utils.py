import random
import numpy as np
import copy
import json
from typing import Any, Mapping

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
import warnings

# Suppress warnings for beta APIs in torchvision
warnings.filterwarnings(
    "ignore",
    message=".*torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta.*")


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def decode_ids(input_ids, tokenizer: PreTrainedTokenizer, by_token=False):
    input_ids = input_ids.detach().cpu().numpy()

    texts = []

    if by_token:
        for input_ids_i in input_ids:
            curr_text = []
            for tmp in input_ids_i:
                curr_text.append(tokenizer.decode([tmp]))

            texts.append('|'.join(curr_text))
    else:
        for input_ids_i in input_ids:
            texts.append(tokenizer.decode(input_ids_i))

    return texts


def get_target_feature(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device, target_prompts=None):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure EOS token exists
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({'eos_token': '</s>'})
    eos_token_id = tokenizer.eos_token_id

    # Append EOS token explicitly to prompts
    target_prompts = [prompt + tokenizer.eos_token for prompt in target_prompts]

    tokenized_inputs = tokenizer(
        target_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=False  # Avoid automatic addition
    ).to(device)

    input_ids = tokenized_inputs['input_ids']
    model = model.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        last_hidden_state = outputs.hidden_states[-1]

    eos_embeddings = []
    for i, ids in enumerate(input_ids):
        eos_positions = (ids == eos_token_id).nonzero(as_tuple=True)
        if len(eos_positions[0]) == 0:
            raise ValueError("EOS token not found in input_ids. Check tokenizer and inputs.")
        eos_position = eos_positions[0][-1].item()
        eos_embeddings.append(last_hidden_state[i, eos_position])

    all_target_features = torch.stack(eos_embeddings)
    return all_target_features


def nn_project(curr_embeds, embedding_layer):
    with torch.no_grad():
        # Ensure embeddings are in float32 for stable normalization
        curr_embeds = curr_embeds.float()
        bsz, seq_len, emb_dim = curr_embeds.shape

        # Normalize embeddings
        curr_embeds = curr_embeds.reshape((-1, emb_dim))
        curr_embeds = torch.nn.functional.normalize(curr_embeds, p=2, dim=-1)

        # Normalize embedding matrix
        embedding_matrix = embedding_layer.weight.float()
        embedding_matrix = torch.nn.functional.normalize(embedding_matrix, p=2, dim=-1)

        # Compute cosine similarity manually
        similarity = torch.matmul(curr_embeds, embedding_matrix.T)

        # Find nearest neighbors
        nn_indices = torch.argmax(similarity, dim=-1)

        # Reshape indices back to original batch size
        nn_indices = nn_indices.reshape((bsz, seq_len))

        # Map to embeddings
        projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices


def optimize_prompt_loop(model, tokenizer, prompt_embeds, all_target_features, args, tokenized_prompts):
    opt_iters = args['iter']
    lr = 0.001  # Updated learning rate to 0.001
    weight_decay = args['weight_decay']
    print_step = args['print_step']

    # Ensure prompt_embeds is a leaf tensor
    prompt_embeds = prompt_embeds.clone().detach().to('cuda:0').requires_grad_(True)

    input_optimizer = torch.optim.AdamW([prompt_embeds], lr=lr, weight_decay=weight_decay)

    best_sim = -1000 * args['loss_weight']
    best_text = ""

    for step in range(opt_iters):
        # Project embeddings to nearest neighbors
        projected_embeds, nn_indices = nn_project(prompt_embeds, model.get_input_embeddings())

        # Update prompt_embeds with projected embeddings
        prompt_embeds = projected_embeds.clone().detach().requires_grad_(True)

        # Compute outputs
        padded_embeds = prompt_embeds.reshape(1, -1, projected_embeds.shape[-1]).to('cuda:0')
        outputs = model(inputs_embeds=padded_embeds, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[-1]

        # Extract EOS embeddings
        eos_positions = [(ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[-1][-1].item() for ids in
                         tokenized_prompts['input_ids']]
        eos_positions = torch.tensor(eos_positions, device='cuda:0')
        prompt_features = hidden_states[:, eos_positions, :]

        # Compute cosine similarity
        prompt_features = torch.nn.functional.normalize(prompt_features, p=2, dim=-1)
        all_target_features = torch.nn.functional.normalize(all_target_features, p=2, dim=-1)
        cosim_scores = torch.nn.functional.cosine_similarity(prompt_features, all_target_features, dim=-1)

        # Loss computation
        loss = 1 - cosim_scores.mean() * args['loss_weight']

        # Debug logs
        print(f"Step {step}: Loss {loss.item():.4f}")

        # Gradient update
        input_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(prompt_embeds, max_norm=0.1)

        # Optimizer step
        input_optimizer.step()

        # Track best embeddings and prompt text
        if loss.item() < best_sim:
            best_sim = loss.item()
            best_text = decode_ids(nn_indices, tokenizer)[0]

        print(f"Best Prompt: {best_text}")

    print(f"Final Prompt: {best_text}")

    return best_text


def optimize_prompt(model, tokenizer, args, device, target_prompts=None):
    model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_prompts = [prompt + tokenizer.eos_token for prompt in target_prompts]

    tokenized_prompts = tokenizer(
        target_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=False
    ).to(device)

    token_embedding = model.get_input_embeddings()
    prompt_embeds = token_embedding(tokenized_prompts['input_ids']).clone().detach().to('cuda:0').requires_grad_(True)

    all_target_features = get_target_feature(model, tokenizer, device, target_prompts=target_prompts)
    learned_prompt = optimize_prompt_loop(model, tokenizer, prompt_embeds, all_target_features, args, tokenized_prompts)
    return learned_prompt
