import os
import math
import json
import random
import argparse

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from kvslimmer.cache import KVSlimmerCache
from kvslimmer.patch import enable_kvslimmer_attention


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default="Llama-3.1-8B-Instruct",
        choices=["gemma-1.1-2b", "gemma-1.1-7b", "Mistral-7B-Instruct-v0.3",
                 "Llama-3.1-8B-Instruct", "Qwen2-7B-Instruct"]
    )
    parser.add_argument('--method', type=str, default="kvslimmer", choices=["llm", "kvslimmer"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)


def build_chat(prompt, model_name, tokenizer):
    if "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "Llama-3" in model_name:
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif "Qwen" in model_name or "Mistral" in model_name:
        chat = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    elif "gemma" in model_name:
        chat = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device, method):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    model = model.eval()
    return model, tokenizer


def smooth_hessian_proxy_like_hk(
    new_proxy,
    old_proxy,
    delta_idx: int,
    alpha: float = 0.0,
):
    if new_proxy is None:
        return None
    if old_proxy is None or old_proxy == []:
        return new_proxy

    denom = max(1, int(delta_idx))

    smoothed = []
    for i in range(len(new_proxy)):
        new_hk = new_proxy[i][0]
        old_hk = old_proxy[i][0]

        old_len = old_hk.shape[2]
        hk_part1 = ((1.0 / denom) + alpha) * new_hk[:, :, :old_len, :] + \
                   (((denom - 1.0) / denom) - alpha) * old_hk
        hk_part2 = new_hk[:, :, old_len:, :]
        hk = torch.cat((hk_part1, hk_part2), dim=2)

        smoothed.append((hk, None))

    return smoothed


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len, model_name="llama"):
    generated_ids = [input_ids.item()]
    pred_token_idx = input_ids
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False
        ).strip()

        if "llama" in model_name.lower() and (
            pred_token_idx[0].item() == tokenizer.eos_token_id or pred_token_idx[0].item() == 128009
        ):
            break
        if "qwen" in model_name.lower() and (
            pred_token_idx[0].item() == tokenizer.eos_token_id or tokenizer.decode(pred_token_idx[0]) == "<|im_end|>"
        ):
            break
        if "mistral" in model_name.lower() and (
            pred_token_idx[0].item() == tokenizer.eos_token_id or tokenizer.decode(pred_token_idx[0]) == "[/INST]"
        ):
            break
    return generated_text


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def build_hessian_proxy_from_ratio(
    past_key_values,
    attns_past_only,        
    start_size: int,
    num_key_value_groups: int,
    eps: float = 1e-21,
):

    hessian_local = []

    for layer_idx, layer in enumerate(past_key_values):
        k = layer[0]  # (B, kvH, T, D)
        v = layer[1]  # (B, kvH, T, Dv)
        l = layer[2]  # (B, kvH, T)

        B, kvH, T, D = k.shape

        attn = attns_past_only[layer_idx]         # (B, attnH, Q, T)
        attn_pos = attn.sum(dim=-2)               # (B, attnH, T)

        rep = num_key_value_groups
        attn_pos = attn_pos.view(B, kvH, rep, T).mean(dim=2)  # (B, kvH, T)

        if T <= start_size:
            hk = torch.zeros_like(k)
            hessian_local.append((hk, None))
            continue

        a  = attn_pos[:, :, start_size:]          # (B, kvH, Tmid)
        vv = v[:, :, start_size:, :]              # (B, kvH, Tmid, Dv)

        ll = l[:, :, start_size:].to(a.dtype)     # (B, kvH, Tmid)
        a  = a * ll

        Tmid = a.shape[-1]
        if Tmid == 0:
            hk = torch.zeros_like(k)
            hessian_local.append((hk, None))
            continue

        alpha = a / (a.sum(dim=-1, keepdim=True) + eps)        # (B, kvH, Tmid)

        o_global = (alpha[..., None] * vv).sum(dim=2)          # (B, kvH, Dv)
        dev = vv - o_global[:, :, None, :]                     # (B, kvH, Tmid, Dv)

        d_scalar = dev.abs().sum(dim=-1)                       # (B, kvH, Tmid)

        h_mid = alpha * (1.0 - 2.0 * alpha) * d_scalar         # (B, kvH, Tmid)

        hk = torch.zeros_like(k)
        hk[:, :, start_size:, :] = h_mid[..., None].expand(B, kvH, Tmid, D)

        hessian_local.append((hk, None))

    return hessian_local


def get_pred(model, tokenizer, rank, world_size, data_all, max_gen, prompt_format,
             dataset, device, model_name, out_path, method):
    data = data_all[rank::world_size]
    k_seq_dim = v_seq_dim = 2
    recent_size = 2048

    if method == "kvslimmer":
        enable_kvslimmer_attention(model_name, model)

    for json_obj in tqdm(data, desc=f"Processing dataset {dataset} on rank {rank}"):
        if method == "kvslimmer":
            kv_cache = KVSlimmerCache(
                start_size=32,
                recent_size=recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
            )
        else:
            kv_cache = None

        past_key_values = None

        if dataset in ["lsht", "trec", "triviaqa", "samsum"] or dataset in ["lcc", "repobench-p"]:
            prompt = prompt_format.format(**json_obj)
        else:
            prompt = build_chat(prompt_format.format(**json_obj), model_name, tokenizer)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        input_window = 512
        outputs = None
        hessian_diagonal = None
        attns_global = None
        delta_idx = 0
        alpha_smooth = 0.0

        for idx in range(0, context_length - 1, input_window):
            if idx + input_window < context_length:
                input_ids = input.input_ids[:, idx: idx + input_window].to(device)
            elif idx > context_length:
                input_ids = input.input_ids[:, idx - input_window:].to(device)
            else:
                input_ids = input.input_ids[:, idx:].to(device)

            if kv_cache is not None and past_key_values is not None and method == "kvslimmer":
                num_key_value_groups = model.model.layers[0].self_attn.num_key_value_groups

                with torch.no_grad():
                    outputs_attn = model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=True
                    )

                attns = outputs_attn.attentions  # list[layer]
                attns = [attn[:, :, :, :-input_ids.shape[-1]] for attn in attns]  # past-only
                attns_global = attns

                hessian_diagonal_local = build_hessian_proxy_from_ratio(
                    past_key_values=past_key_values,
                    attns_past_only=attns_global,
                    start_size=kv_cache.start_size,
                    num_key_value_groups=num_key_value_groups,
                    eps=1e-21,
                )

                hessian_diagonal_local = smooth_hessian_proxy_like_hk(
                    new_proxy=hessian_diagonal_local,
                    old_proxy=hessian_diagonal,
                    delta_idx=delta_idx,
                    alpha=alpha_smooth,
                )
                hessian_diagonal = hessian_diagonal_local

                past_key_values, hessian_diagonal_local = kv_cache(
                    past_key_values,
                    attns_global,
                    hessian_diagonal_local
                )
                hessian_diagonal = hessian_diagonal_local

            with torch.no_grad():
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

            delta_idx += 1

        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        pred = greedy_generate(model, tokenizer, pred_token_idx, past_key_values, max_gen_len=max_gen)
        pred = post_process(pred, model_name)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"]
                },
                f,
                ensure_ascii=False
            )
            f.write('\n')

    if world_size > 1:
        dist.barrier()


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend='nccl')

    model2path = json.load(open("config/model2path.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["hotpotqa"]


    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    if local_rank == 0:
        os.makedirs("pred", exist_ok=True)
        os.makedirs("pred_e", exist_ok=True)

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, args.method)

    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            os.makedirs(f"pred_e/{model_name}", exist_ok=True)
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset("json", data_files=f"data/{dataset}.jsonl")['train']
            os.makedirs(f"pred/{args.method}/{model_name}", exist_ok=True)
            out_path = f"pred/{args.method}/{model_name}/{dataset}.jsonl"

        if local_rank == 0 and os.path.exists(out_path):
            os.remove(out_path)

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]

        get_pred(
            model, tokenizer, local_rank, world_size, data_all,
            max_gen, prompt_format, dataset, device, model_name, out_path, args.method
        )

    if world_size > 1:
        dist.destroy_process_group()



