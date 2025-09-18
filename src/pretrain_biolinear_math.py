from pathlib import Path
import argparse, os, math
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, get_cosine_schedule_with_warmup

# 你项目里的数据加载
from src.get_dataset_math import LogicDataset, load_augmented_json_grouped, collate_fn

# BioLinear patch
from src.patch_qwen_bio import replace_linear_with_biolinear, connection_cost

def flatten_logic_batch_to_lm(batch, device, ignore_index=-100):
    """
    将 LogicDataset 的 collate 结果展开为标准 LM 批：
      input_ids [N, L], attention_mask [N, L], labels [N, L]
    我们把 clean 和 corrupt 都当作训练样本（下一个 token 预测由 HF 内部 shift）。
    """
    xs, masks = [], []
    # batch: { logic: [ [g1_dict, g2_dict], ... ] }
    for _, pair_list in batch.items():
        for g1, g2 in pair_list:
            # g1、g2 的每个字段形状都是 [m, L]，直接拼 batch 维
            for ids, am in [
                (g1["clean_ids"],    g1["clean_mask"]),
                (g2["clean_ids"],    g2["clean_mask"]),
                (g1["corrupt_ids"],  g1["corrupt_mask"]),
                (g2["corrupt_ids"],  g2["corrupt_mask"]),
            ]:
                xs.append(ids)
                masks.append(am)

    input_ids = torch.cat(xs, dim=0).to(device, non_blocking=True)
    attn_mask = torch.cat(masks, dim=0).to(device, non_blocking=True)
    # labels 与 input_ids 同形状；把 padding 位置设为 ignore_index
    labels = input_ids.clone()
    labels[attn_mask == 0] = ignore_index
    return input_ids, attn_mask, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", type=Path, default=Path("data/corrupt/math.json"))
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--save_dir", type=str, default="./ckpts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_type", type=int, default=2)
    ap.add_argument("--lambda_cc", type=float, default=None)
    ap.add_argument("--weight_factor", type=float, default=None)
    ap.add_argument("--l0", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    n_heads = getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", 1))
    replace_linear_with_biolinear(model, n_heads=n_heads, l0=args.l0)
    target_dtype = next(model.parameters()).dtype  # 通常是 torch.bfloat16
    model.to(device).to(target_dtype)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # dataset
    rows = load_augmented_json_grouped(args.data_json)
    ds = LogicDataset(rows, tok, group_size=2, n_logic_per_item=2, max_length=args.max_len, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.steps)

    lamb = args.lambda_cc if args.lambda_cc is not None else (0.0 if args.train_type==1 else 1)
    weight_factor = args.weight_factor if args.weight_factor is not None else (1.0 if args.train_type in [3,5] else 0.0)


    dataset_tag = args.data_json.stem
    save_dir = Path(args.save_dir) / dataset_tag
    os.makedirs(save_dir, exist_ok=True)

    data_iter = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # 🔧 关键修改：把嵌套结构展平成标准 LM 批
        input_ids, attn_mask, labels = flatten_logic_batch_to_lm(batch, device)

        # outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        # ce = outputs.loss

        bias_penalize = not (step < int(0.75 * args.steps))
        cc = connection_cost(model, weight_factor=weight_factor, bias_penalize=bias_penalize)
        loss = lamb * cc # only try to improve cc 

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step == int(args.steps/4):
            lamb *= 10
        if step == int(3*args.steps/4):
            lamb *= 0.1

        if step % args.log_every == 0:
            # ppl = math.exp(min(20.0, float(ce.detach().cpu())))
            print(f"step {step} | CC {float(cc):.2e} | lamb {lamb:.2e}")
            # print(f"step {step} | CE {float(ce):.4f} | CC {float(cc):.2e} | lamb {lamb:.2e} | ppl {ppl:.2f}")

        if step % args.save_every == 0 and step > 0:
            save_path = os.path.join(save_dir, f"step_{step}_bio")
            model.save_pretrained(save_path)
            tok.save_pretrained(save_path)

    # model.save_pretrained(os.path.join(args.save_dir, "final_bio"))
    # tok.save_pretrained(os.path.join(args.save_dir, "final_bio"))

if __name__ == "__main__":
    main()