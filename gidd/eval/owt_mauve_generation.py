import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from gidd import GiddPipeline
from gidd.utils import parse_dtype, sample_categorical


sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
try:
    print("Importing AutoResume lib...", flush=True)
    from userlib.auto_resume import AutoResume

    AutoResume.init()
    print("Found AutoResume SDK!", flush=True)
except Exception:
    AutoResume = None
    print("Did not find AutoResume SDK!", flush=True)


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


def _barrier():
    if _is_distributed():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def _log(message: str, rank: int | None = None):
    prefix = f"RANK{rank}: " if rank is not None else ""
    print(f"{prefix}{message}", flush=True)


def _maybe_autoresume(args, rank: int, stage: str):
    if not args.enable_autoresume or AutoResume is None:
        return
    if AutoResume.termination_requested():
        _log(f"AutoResumeHook: requesting resume after {stage}", rank)
        AutoResume.request_resume({"resumed": True, "stage": stage, "rank": rank})
        raise SystemExit(0)


def _rank_sample_count(total: int, rank: int, world_size: int) -> int:
    base = total // world_size
    return base + int(rank < total % world_size)


def _rank_slice(total: int, rank: int, world_size: int) -> tuple[int, int]:
    base = total // world_size
    remainder = total % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + int(rank < remainder)
    return start, end


def _setup_distributed(args):
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.distributed and env_world_size > 1:
        # This script uses distributed only for host-side coordination and
        # object collectives; model work is embarrassingly parallel on each GPU.
        # Gloo is more robust here than NCCL because we broadcast Python lists
        # and synchronize filesystem progress, not CUDA tensors.
        backend = args.distributed_backend or "gloo"
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(minutes=args.distributed_timeout_minutes),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
            mauve_device_id = local_rank
        else:
            device = torch.device("cpu")
            mauve_device_id = -1
        _log(
            f"distributed initialized backend={backend} world_size={world_size} "
            f"local_rank={local_rank} device={device}",
            rank,
        )
        return rank, world_size, local_rank, device, mauve_device_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mauve_device_id = 0 if device.type == "cuda" else -1
    return 0, 1, 0, device, mauve_device_id


def _load_pipeline(args, device: torch.device, rank: int, world_size: int):
    def _load_this_rank():
        start_time = time.time()
        _log(f"from_pretrained start: {args.model_name}", rank)
        pipe = GiddPipeline.from_pretrained(
            args.model_name,
            trust_remote_code=args.trust_remote_code,
            compile_step=args.compile_step,
        )
        _log(
            f"from_pretrained complete in {time.time() - start_time:.1f}s; "
            f"moving model to {device}",
            rank,
        )
        pipe.to(device)
        pipe.eval()
        _log(f"model ready in {time.time() - start_time:.1f}s", rank)
        return pipe

    if not _is_distributed() or not args.serialized_model_load:
        return _load_this_rank()

    pipe = None
    for loading_rank in range(world_size):
        if rank == loading_rank:
            pipe = _load_this_rank()
        _barrier()
    if pipe is None:
        raise RuntimeError(f"Rank {rank} did not load a pipeline.")
    return pipe


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _jsonl_has_count(path: Path, count: int) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) == count


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f if _.strip())


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_output_dir(args, budget: int) -> Path:
    if args.output_dir is not None:
        return Path(hydra.utils.to_absolute_path(args.output_dir))
    output_root = Path(hydra.utils.to_absolute_path(args.output_root))
    return (
        output_root
        / f"model-{_model_slug(args.model_name)}"
        / f"num_samples-{args.num_samples}"
        / f"seed-{args.seed}"
        / f"budget-{budget}"
    )


def _resolve_reference_features_path(args) -> Path:
    if args.reference_features_path is not None:
        return Path(hydra.utils.to_absolute_path(args.reference_features_path))
    reference_cache_dir = Path(hydra.utils.to_absolute_path(args.reference_cache_dir))
    feature_model = args.mauve_featurize_model.replace("/", "__")
    file_name = (
        f"owt_refs_{feature_model}"
        f"_num_samples-{args.reference_num_samples}"
        f"_max_len-{args.mauve_max_text_length}.npy"
    )
    return reference_cache_dir / file_name


def _resolve_reference_texts_path(args) -> Path:
    if args.reference_texts_path is not None:
        return Path(hydra.utils.to_absolute_path(args.reference_texts_path))
    reference_cache_dir = Path(hydra.utils.to_absolute_path(args.reference_cache_dir))
    return reference_cache_dir / f"owt_refs_num_samples-{args.reference_num_samples}.jsonl"


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _acquire_reference_lock(features_path: Path, timeout_seconds: int) -> Path | None:
    lock_path = features_path.with_suffix(features_path.suffix + ".lock")
    start_time = time.time()
    while True:
        if features_path.exists():
            return None
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"pid={os.getpid()}\n")
            return lock_path
        except FileExistsError:
            if timeout_seconds > 0 and time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for reference feature lock {lock_path}")
            print(f"Waiting for reference features at {features_path}")
            time.sleep(30)


def _load_reference_texts(args) -> list[str]:
    texts_path = _resolve_reference_texts_path(args)
    if _jsonl_has_count(texts_path, args.reference_num_samples):
        return _read_jsonl(texts_path)

    print(
        f"Loading {args.reference_num_samples} OWT references "
        f"from {args.reference_dataset}:{args.reference_split}"
    )
    ds = load_dataset(
        args.reference_dataset,
        args.reference_subset,
        split=args.reference_split,
        trust_remote_code=args.reference_trust_remote_code,
    )
    if len(ds) < args.reference_num_samples:
        raise ValueError(
            f"Reference split only has {len(ds)} rows, "
            f"but reference_num_samples={args.reference_num_samples}."
        )
    texts = [row["text"] for row in ds.select(range(args.reference_num_samples))]
    _write_jsonl(texts_path, texts)
    return texts


def _append_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _count_jsonl_paths(paths: list[Path]) -> int:
    return sum(_jsonl_count(path) for path in paths)


def _merge_jsonl_prefix(output_path: Path, shard_paths: list[Path], total_count: int) -> list:
    merged = []
    for shard_path in shard_paths:
        if shard_path.exists():
            merged.extend(_read_jsonl(shard_path))
        if len(merged) >= total_count:
            break
    merged = merged[:total_count]
    if len(merged) != total_count:
        raise ValueError(f"Only found {len(merged)} records, expected {total_count}.")
    _write_jsonl(output_path, merged)
    return merged


def _generate_raw_samples(pipe, args, budget: int, shard_dir: Path, raw_path: Path, dtype: torch.dtype, rank: int, world_size: int) -> list[str] | None:
    if not args.overwrite and _jsonl_count(raw_path) >= args.num_samples:
        if rank == 0:
            _log(f"Reusing merged raw samples at {raw_path}", rank)
        _barrier()
        return _read_jsonl(raw_path)[: args.num_samples] if rank == 0 else None

    raw_shard_path = shard_dir / f"raw_samples_rank{rank:05d}.jsonl"
    if rank == 0:
        raw_shard_paths = sorted(shard_dir.glob("raw_samples_rank*.jsonl"))
        existing_total = _count_jsonl_paths(raw_shard_paths)
    else:
        existing_total = None
    if _is_distributed():
        existing_total_msg = [existing_total]
        dist.broadcast_object_list(existing_total_msg, src=0)
        existing_total = existing_total_msg[0]
    remaining_total = max(args.num_samples - existing_total, 0)
    local_remaining = _rank_sample_count(remaining_total, rank, world_size)

    if rank == 0:
        _log(f"Already generated {min(existing_total, args.num_samples)} raw samples.", rank)
        _log(f"Generating {remaining_total} remaining raw samples across {world_size} rank(s).", rank)

    with tqdm.tqdm(total=local_remaining, desc=f"Generating rank {rank}", dynamic_ncols=True, disable=(rank != 0)) as pbar:
        for start in range(0, local_remaining, args.batch_size):
            batch_size = min(args.batch_size, local_remaining - start)
            batch_texts = pipe.generate(
                num_samples=batch_size,
                num_inference_steps=budget,
                show_progress=False,
                dtype=dtype,
            )
            _append_jsonl(raw_shard_path, batch_texts)
            pbar.update(batch_size)
            _maybe_autoresume(args, rank, "raw_generation_batch")

    _barrier()
    if rank == 0:
        raw_shard_paths = sorted(shard_dir.glob("raw_samples_rank*.jsonl"))
        return _merge_jsonl_prefix(raw_path, raw_shard_paths, args.num_samples)
    return None


@torch.no_grad()
def _correction_step(model, tokenizer, z_t, t, temp, tokens_per_step: int):
    logits = model(z_t, t)
    logits[..., tokenizer.mask_token_id] = -1e6

    p_t = (logits / temp).softmax(-1)
    z_tm1 = sample_categorical(p_t)
    score = (z_tm1 != z_t) * p_t.gather(-1, z_tm1.unsqueeze(-1)).squeeze(-1)

    ids = torch.topk(score, tokens_per_step, dim=-1).indices
    z_tm1 = z_t.scatter(-1, ids, z_tm1.gather(-1, ids))
    acc = (z_tm1 == logits.argmax(-1)).float().mean(-1)
    return z_tm1, acc


@torch.no_grad()
def _self_correct_batch(pipe, texts, args, budget: int, dtype: torch.dtype):
    device = next(pipe.model.parameters()).device
    tokenizer = pipe.tokenizer
    max_length = pipe.config.max_seq_len
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )["input_ids"].to(device)

    max_acc = torch.zeros(tokens.shape[0], device=device)
    patience = torch.zeros(tokens.shape[0], dtype=torch.long, device=device)
    steps_taken = torch.zeros(tokens.shape[0], dtype=torch.long, device=device)
    finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=device)

    for _ in range(budget):
        active = (~finished).nonzero(as_tuple=False).flatten()
        if active.numel() == 0:
            break

        z_t = tokens.index_select(0, active)
        t = torch.full((active.numel(),), args.correction_t0, device=device)
        with torch.autocast(device.type, dtype=dtype):
            z_next, acc = _correction_step(
                pipe.model,
                tokenizer,
                z_t,
                t,
                args.correction_temperature,
                args.correction_tokens_per_step,
            )

        steps_taken[active] += 1
        if args.correction_early_stopping:
            improved = acc > max_acc.index_select(0, active)
            next_patience = patience.index_select(0, active) + (~improved).long()
            next_patience = torch.where(improved, torch.zeros_like(next_patience), next_patience)
            max_acc[active] = torch.maximum(max_acc.index_select(0, active), acc)
            patience[active] = next_patience

            changed = (z_t != z_next).any(-1)
            stop = (next_patience > args.correction_patience) | (~changed)
            apply_update = ~stop
            z_updated = torch.where(apply_update.unsqueeze(-1), z_next, z_t)
            finished[active[stop]] = True
        else:
            z_updated = z_next

        tokens[active] = z_updated

    corrected = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    return corrected, steps_taken.cpu().numpy().tolist()


def _read_correction_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict) and {"idx", "text", "nfes"} <= set(record):
                records.append(record)
    return records


def _merge_correction_records(shard_paths: list[Path], total_count: int) -> tuple[list[str], list[int]]:
    by_idx = {}
    for shard_path in shard_paths:
        for record in _read_correction_records(shard_path):
            idx = int(record["idx"])
            if 0 <= idx < total_count and idx not in by_idx:
                by_idx[idx] = record
    missing = [idx for idx in range(total_count) if idx not in by_idx]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(f"Missing {len(missing)} corrected samples; first missing indices: {preview}")
    ordered = [by_idx[idx] for idx in range(total_count)]
    return [record["text"] for record in ordered], [int(record["nfes"]) for record in ordered]


def _self_correct_samples(pipe, raw_texts: list[str], args, budget: int, shard_dir: Path, corrected_path: Path, nfes_path: Path, dtype: torch.dtype, rank: int, world_size: int):
    total_count = len(raw_texts)
    if not args.overwrite and _jsonl_has_count(corrected_path, total_count) and nfes_path.exists():
        if rank == 0:
            _log(f"Reusing merged corrected samples at {corrected_path}", rank)
        _barrier()
        return (_read_jsonl(corrected_path), _read_json(nfes_path)) if rank == 0 else (None, None)

    correction_shard_path = shard_dir / f"corrected_samples_rank{rank:05d}.jsonl"
    if rank == 0:
        correction_shard_paths = sorted(shard_dir.glob("corrected_samples_rank*.jsonl"))
        completed = {}
        for shard_path in correction_shard_paths:
            for record in _read_correction_records(shard_path):
                idx = int(record["idx"])
                if 0 <= idx < total_count and idx not in completed:
                    completed[idx] = record
        remaining_indices = [idx for idx in range(total_count) if idx not in completed]
        _log(f"Already corrected {len(completed)} samples.", rank)
        _log(f"Correcting {len(remaining_indices)} remaining samples across {world_size} rank(s).", rank)
    else:
        remaining_indices = None
    if _is_distributed():
        remaining_indices_msg = [remaining_indices]
        dist.broadcast_object_list(remaining_indices_msg, src=0)
        remaining_indices = remaining_indices_msg[0]
    start, end = _rank_slice(len(remaining_indices), rank, world_size)
    local_indices = remaining_indices[start:end]

    with tqdm.tqdm(total=len(local_indices), desc=f"Correcting rank {rank}", dynamic_ncols=True, disable=(rank != 0)) as pbar:
        for start in range(0, len(local_indices), args.batch_size):
            batch_indices = local_indices[start : start + args.batch_size]
            batch_texts = [raw_texts[idx] for idx in batch_indices]
            corrected, steps_taken = _self_correct_batch(pipe, batch_texts, args, budget, dtype)
            records = [
                {"idx": int(idx), "text": text, "nfes": int(nfes)}
                for idx, text, nfes in zip(batch_indices, corrected, steps_taken)
            ]
            _append_jsonl(correction_shard_path, records)
            pbar.update(len(batch_texts))
            _maybe_autoresume(args, rank, "self_correction_batch")

    _barrier()
    if rank == 0:
        correction_shard_paths = sorted(shard_dir.glob("corrected_samples_rank*.jsonl"))
        corrected_texts, correction_nfes = _merge_correction_records(correction_shard_paths, total_count)
        _write_jsonl(corrected_path, corrected_texts)
        _write_json(nfes_path, correction_nfes)
        return corrected_texts, correction_nfes
    return None, None


def _compute_reference_features(args, device_id: int, features_path: Path):
    if features_path.exists():
        features = np.load(features_path)
        if features.shape[0] < args.reference_num_samples:
            raise ValueError(
                f"Cached reference features at {features_path} contain {features.shape[0]} rows, "
                f"but reference_num_samples={args.reference_num_samples}."
            )
        return features[: args.reference_num_samples]

    lock_path = _acquire_reference_lock(features_path, args.reference_lock_timeout_seconds)
    if lock_path is None:
        features = np.load(features_path)
        return features[: args.reference_num_samples]

    try:
        if features_path.exists():
            features = np.load(features_path)
            return features[: args.reference_num_samples]

        from mauve.compute_mauve import get_features_from_input

        reference_texts = _load_reference_texts(args)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        features = get_features_from_input(
            features=None,
            tokenized_texts=None,
            texts=reference_texts,
            featurize_model_name=args.mauve_featurize_model,
            max_len=args.mauve_max_text_length,
            device_id=device_id,
            name="p",
            verbose=True,
            batch_size=args.mauve_batch_size,
            use_float64=args.mauve_use_float64,
        )
        tmp_path = features_path.with_suffix(features_path.suffix + f".tmp-{os.getpid()}.npy")
        np.save(tmp_path, features)
        os.replace(tmp_path, features_path)
        return features
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _compute_mauve(args, corrected_texts, features_path: Path, device_id: int):
    import mauve

    p_features = _compute_reference_features(args, device_id, features_path)
    p_features = p_features[: len(corrected_texts)]
    results = mauve.compute_mauve(
        p_text=None,
        p_features=p_features,
        q_text=corrected_texts,
        q_features=None,
        featurize_model_name=args.mauve_featurize_model,
        device_id=device_id,
        max_text_length=args.mauve_max_text_length,
        verbose=True,
        batch_size=args.mauve_batch_size,
        use_float64=args.mauve_use_float64,
    )
    return float(results.mauve)


def _compute_entropy(texts: list[str], tokenizer, max_length: int, entropies_path: Path, args, rank: int):
    entropies = []
    for text in tqdm.tqdm(texts, desc="Entropy", dynamic_ncols=True):
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )["input_ids"][0]
        counts = torch.unique(tokenized, return_counts=True, sorted=True)[1]
        entropies.append(torch.special.entr(counts.float() / counts.sum()).sum().item())
        _maybe_autoresume(args, rank, "entropy")
    _write_json(entropies_path, entropies)
    return {
        "entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "entropy_num_samples": len(entropies),
        "entropies_path": str(entropies_path),
    }


@torch.no_grad()
def _compute_generative_ppl_local(texts: list[str], args, device: torch.device, rank: int):
    if not texts:
        return {"nll_sum": 0.0, "acc_sum": 0.0, "token_count": 0}

    _log(f"Loading gen-PPL model {args.gen_ppl_model_name_or_path}", rank)
    tokenizer = AutoTokenizer.from_pretrained(args.gen_ppl_model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.gen_ppl_model_name_or_path,
        trust_remote_code=True,
    ).eval().to(device)

    total_nll = 0.0
    total_acc = 0.0
    total_tokens = 0
    for start in tqdm.trange(
        0,
        len(texts),
        args.gen_ppl_batch_size,
        desc=f"Gen PPL rank {rank}",
        dynamic_ncols=True,
        disable=(rank != 0),
    ):
        xs = texts[start : start + args.gen_ppl_batch_size]
        batch = tokenizer(
            xs,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=args.gen_ppl_max_length,
        ).to(device)
        attn_mask = batch["attention_mask"]
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=attn_mask,
            use_cache=False,
        ).logits[:, :-1]

        labels = batch["input_ids"][:, 1:]
        loss_mask = attn_mask[:, 1:]
        nll = F.cross_entropy(
            logits.flatten(0, 1),
            labels.flatten(0, 1),
            reduction="none",
        ).view_as(labels)
        acc = (logits.argmax(-1) == labels).float()

        total_nll += (nll * loss_mask).sum().item()
        total_acc += (acc * loss_mask).sum().item()
        total_tokens += int(loss_mask.sum().item())
        _maybe_autoresume(args, rank, "generative_ppl_batch")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "nll_sum": total_nll,
        "acc_sum": total_acc,
        "token_count": total_tokens,
    }


def _compute_generative_ppl(args, corrected_texts: list[str], device: torch.device, rank: int, world_size: int):
    local_stats = {"nll_sum": 0.0, "acc_sum": 0.0, "token_count": 0}
    if not args.skip_gen_ppl and args.gen_ppl_model_name_or_path is not None:
        start, end = _rank_slice(len(corrected_texts), rank, world_size)
        local_stats = _compute_generative_ppl_local(corrected_texts[start:end], args, device, rank)

    gathered_stats = [None for _ in range(world_size)]
    if _is_distributed():
        dist.all_gather_object(gathered_stats, local_stats)
    else:
        gathered_stats = [local_stats]

    if rank != 0:
        return None

    if args.skip_gen_ppl or args.gen_ppl_model_name_or_path is None:
        return {
            "generative_ppl": -1.0,
            "gen_ppl_avg_nll": -1.0,
            "gen_ppl_acc": -1.0,
            "gen_ppl_tokens": 0,
            "gen_ppl_model_name_or_path": args.gen_ppl_model_name_or_path,
        }

    total_nll = sum(float(stats["nll_sum"]) for stats in gathered_stats if stats is not None)
    total_acc = sum(float(stats["acc_sum"]) for stats in gathered_stats if stats is not None)
    total_tokens = sum(int(stats["token_count"]) for stats in gathered_stats if stats is not None)
    if total_tokens == 0:
        avg_nll = float("nan")
        generative_ppl = float("nan")
        acc = float("nan")
    else:
        avg_nll = total_nll / total_tokens
        generative_ppl = float(np.exp(avg_nll))
        acc = total_acc / total_tokens

    return {
        "generative_ppl": generative_ppl,
        "gen_ppl_avg_nll": avg_nll,
        "gen_ppl_acc": acc,
        "gen_ppl_tokens": total_tokens,
        "gen_ppl_model_name_or_path": args.gen_ppl_model_name_or_path,
    }


def _write_run_config(args, budget: int, output_dir: Path, reference_features_path: Path):
    config = OmegaConf.to_container(args, resolve=True)
    config["budget"] = budget
    config["output_dir_resolved"] = str(output_dir)
    config["reference_features_path_resolved"] = str(reference_features_path)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _run_budget(pipe, args, budget: int, dtype: torch.dtype, device: torch.device, device_id: int, rank: int, world_size: int):
    output_dir = _resolve_output_dir(args, budget)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_samples.jsonl"
    corrected_path = output_dir / "corrected_samples.jsonl"
    nfes_path = output_dir / "correction_nfes.json"
    entropies_path = output_dir / "entropies.json"
    metrics_path = output_dir / "metrics.json"
    reference_features_path = _resolve_reference_features_path(args)

    if args.overwrite and rank == 0:
        for path in [raw_path, corrected_path, nfes_path, entropies_path, metrics_path]:
            if path.exists():
                path.unlink()
        for path in list(shard_dir.glob("*_rank*.jsonl")) + list(shard_dir.glob("*_rank*.json")):
            path.unlink()
    _barrier()

    skip_budget = metrics_path.exists() and args.skip_existing_metrics and not args.overwrite
    if _is_distributed():
        skip_budget_msg = [skip_budget if rank == 0 else None]
        dist.broadcast_object_list(skip_budget_msg, src=0)
        skip_budget = skip_budget_msg[0]
    if skip_budget:
        if rank == 0:
            print(f"Reusing existing metrics at {metrics_path}")
        _barrier()
        return

    if rank == 0:
        _write_run_config(args, budget, output_dir, reference_features_path)

    raw_texts = _generate_raw_samples(
        pipe,
        args,
        budget,
        shard_dir,
        raw_path,
        dtype,
        rank,
        world_size,
    )

    raw_msg = [raw_texts if rank == 0 else None]
    if _is_distributed():
        dist.broadcast_object_list(raw_msg, src=0)
    raw_texts = raw_msg[0]

    corrected_texts, correction_nfes = _self_correct_samples(
        pipe,
        raw_texts,
        args,
        budget,
        shard_dir,
        corrected_path,
        nfes_path,
        dtype,
        rank,
        world_size,
    )

    if rank == 0 and correction_nfes is None:
        correction_nfes = [budget] * len(corrected_texts)

    if args.release_model_before_mauve:
        pipe.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _barrier()

    samples_msg = [corrected_texts if rank == 0 else None]
    if _is_distributed():
        dist.broadcast_object_list(samples_msg, src=0)
    corrected_texts = samples_msg[0]

    gen_ppl_metrics = _compute_generative_ppl(args, corrected_texts, device, rank, world_size)

    if rank != 0:
        _barrier()
        return

    entropy_metrics = {
        "entropy": -1.0,
        "entropy_num_samples": 0,
        "entropies_path": str(entropies_path),
    }
    if not args.skip_entropy:
        entropy_metrics = _compute_entropy(
            corrected_texts,
            pipe.tokenizer,
            pipe.config.max_seq_len,
            entropies_path,
            args,
            rank,
        )

    if args.skip_mauve:
        mauve_score = -1.0
    else:
        _maybe_autoresume(args, rank, "before_mauve")
        mauve_score = _compute_mauve(args, corrected_texts, reference_features_path, device_id)
        _maybe_autoresume(args, rank, "after_mauve")

    correction_nfes_arr = np.array(correction_nfes, dtype=np.float64)
    metrics = {
        "model_name": args.model_name,
        "budget": budget,
        "num_samples": len(corrected_texts),
        "mauve": mauve_score,
        **gen_ppl_metrics,
        **entropy_metrics,
        "raw_generation_nfes_per_sample": budget,
        "raw_generation_nfes_total": int(budget * len(corrected_texts)),
        "correction_nfes_avg_per_sample": float(correction_nfes_arr.mean()),
        "correction_nfes_min_per_sample": int(correction_nfes_arr.min()),
        "correction_nfes_max_per_sample": int(correction_nfes_arr.max()),
        "correction_nfes_total": int(correction_nfes_arr.sum()),
        "total_nfes_avg_per_sample": float(budget + correction_nfes_arr.mean()),
        "total_nfes_total": int(budget * len(corrected_texts) + correction_nfes_arr.sum()),
        "raw_samples_path": str(raw_path),
        "corrected_samples_path": str(corrected_path),
        "correction_nfes_path": str(nfes_path),
        "world_size": world_size,
        "shard_dir": str(shard_dir) if world_size > 1 else None,
        "reference_features_path": str(reference_features_path),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    _barrier()


@hydra.main(config_path="../configs", config_name="owt_mauve", version_base="1.1")
def main(args):
    rank, world_size, local_rank, device, device_id = _setup_distributed(args)
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)
    _set_seed(args.seed + rank)

    budgets = [int(args.budget)] if args.budget is not None else [int(x) for x in args.budgets]
    dtype = parse_dtype(args.dtype)

    pipe = _load_pipeline(args, device, rank, world_size)

    for budget in budgets:
        pipe.to(device)
        _set_seed(args.seed + budget * 1000 + rank)
        _run_budget(pipe, args, budget, dtype, device, device_id, rank, world_size)

    if _is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
