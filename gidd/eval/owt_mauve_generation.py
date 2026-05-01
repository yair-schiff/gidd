import datetime
import json
import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from datasets import load_dataset
from omegaconf import OmegaConf

from gidd import GiddPipeline
from gidd.utils import parse_dtype, sample_categorical


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
        dist.barrier()


def _rank_sample_count(total: int, rank: int, world_size: int) -> int:
    base = total // world_size
    return base + int(rank < total % world_size)


def _setup_distributed(args):
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.distributed and env_world_size > 1:
        backend = args.distributed_backend or ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(minutes=args.distributed_timeout_minutes),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            mauve_device_id = local_rank
        else:
            device = torch.device("cpu")
            mauve_device_id = -1
        return rank, world_size, local_rank, device, mauve_device_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mauve_device_id = 0 if device.type == "cuda" else -1
    return 0, 1, 0, device, mauve_device_id


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


def _generate_raw_samples(pipe, args, budget: int, raw_path: Path, dtype: torch.dtype, num_samples: int, rank: int) -> list[str]:
    if not args.overwrite and _jsonl_has_count(raw_path, num_samples):
        print(f"Reusing raw samples at {raw_path}")
        return _read_jsonl(raw_path)

    print(f"RANK{rank}: Generating {num_samples} raw samples with budget={budget}")
    raw_texts = []
    with tqdm.tqdm(total=num_samples, desc=f"Generating rank {rank}", dynamic_ncols=True, disable=(rank != 0)) as pbar:
        for start in range(0, num_samples, args.batch_size):
            batch_size = min(args.batch_size, num_samples - start)
            batch_texts = pipe.generate(
                num_samples=batch_size,
                num_inference_steps=budget,
                show_progress=False,
                dtype=dtype,
            )
            raw_texts.extend(batch_texts)
            pbar.update(batch_size)
    _write_jsonl(raw_path, raw_texts)
    return raw_texts


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


def _self_correct_samples(pipe, raw_texts, args, budget: int, corrected_path: Path, nfes_path: Path, dtype: torch.dtype, rank: int):
    expected_count = len(raw_texts)
    if not args.overwrite and _jsonl_has_count(corrected_path, expected_count):
        print(f"Reusing corrected samples at {corrected_path}")
        if nfes_path.exists():
            correction_nfes = _read_json(nfes_path)
        else:
            correction_nfes = [budget] * expected_count
            _write_json(nfes_path, correction_nfes)
        return _read_jsonl(corrected_path), correction_nfes

    print(f"RANK{rank}: Running self-correction with budget={budget}")
    corrected_texts = []
    correction_nfes = []
    with tqdm.tqdm(total=len(raw_texts), desc=f"Correcting rank {rank}", dynamic_ncols=True, disable=(rank != 0)) as pbar:
        for start in range(0, len(raw_texts), args.batch_size):
            batch_texts = raw_texts[start : start + args.batch_size]
            corrected, steps_taken = _self_correct_batch(pipe, batch_texts, args, budget, dtype)
            corrected_texts.extend(corrected)
            correction_nfes.extend(steps_taken)
            pbar.update(len(batch_texts))
    _write_jsonl(corrected_path, corrected_texts)
    _write_json(nfes_path, correction_nfes)
    return corrected_texts, correction_nfes


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


def _write_run_config(args, budget: int, output_dir: Path, reference_features_path: Path):
    config = OmegaConf.to_container(args, resolve=True)
    config["budget"] = budget
    config["output_dir_resolved"] = str(output_dir)
    config["reference_features_path_resolved"] = str(reference_features_path)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _merge_rank_jsonl(output_path: Path, shard_paths: list[Path], expected_counts: list[int], total_count: int) -> list[str]:
    merged = []
    for shard_path, expected_count in zip(shard_paths, expected_counts):
        if not _jsonl_has_count(shard_path, expected_count):
            raise ValueError(f"Shard {shard_path} does not contain {expected_count} records.")
        merged.extend(_read_jsonl(shard_path))
    merged = merged[:total_count]
    _write_jsonl(output_path, merged)
    return merged


def _merge_rank_json_arrays(output_path: Path, shard_paths: list[Path], expected_counts: list[int], total_count: int) -> list[int]:
    merged = []
    for shard_path, expected_count in zip(shard_paths, expected_counts):
        if not shard_path.exists():
            raise ValueError(f"Missing shard {shard_path}.")
        values = _read_json(shard_path)
        if len(values) != expected_count:
            raise ValueError(f"Shard {shard_path} contains {len(values)} records, expected {expected_count}.")
        merged.extend(values)
    merged = merged[:total_count]
    _write_json(output_path, merged)
    return merged


def _run_budget(pipe, args, budget: int, dtype: torch.dtype, device_id: int, rank: int, world_size: int):
    output_dir = _resolve_output_dir(args, budget)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_samples.jsonl"
    corrected_path = output_dir / "corrected_samples.jsonl"
    nfes_path = output_dir / "correction_nfes.json"
    metrics_path = output_dir / "metrics.json"
    reference_features_path = _resolve_reference_features_path(args)

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

    local_num_samples = _rank_sample_count(args.num_samples, rank, world_size)
    expected_counts = [_rank_sample_count(args.num_samples, r, world_size) for r in range(world_size)]
    raw_shard_paths = [shard_dir / f"raw_samples_rank{r:05d}.jsonl" for r in range(world_size)]
    corrected_shard_paths = [shard_dir / f"corrected_samples_rank{r:05d}.jsonl" for r in range(world_size)]
    nfes_shard_paths = [shard_dir / f"correction_nfes_rank{r:05d}.json" for r in range(world_size)]

    if rank == 0:
        _write_run_config(args, budget, output_dir, reference_features_path)

    raw_texts = _generate_raw_samples(
        pipe,
        args,
        budget,
        raw_shard_paths[rank] if world_size > 1 else raw_path,
        dtype,
        local_num_samples if world_size > 1 else args.num_samples,
        rank,
    )
    corrected_texts, correction_nfes = _self_correct_samples(
        pipe,
        raw_texts,
        args,
        budget,
        corrected_shard_paths[rank] if world_size > 1 else corrected_path,
        nfes_shard_paths[rank] if world_size > 1 else nfes_path,
        dtype,
        rank,
    )

    if correction_nfes is None:
        correction_nfes = [budget] * len(corrected_texts)

    if args.release_model_before_mauve:
        pipe.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _barrier()

    if rank != 0:
        _barrier()
        return

    if world_size > 1:
        raw_texts = _merge_rank_jsonl(raw_path, raw_shard_paths, expected_counts, args.num_samples)
        corrected_texts = _merge_rank_jsonl(corrected_path, corrected_shard_paths, expected_counts, args.num_samples)
        correction_nfes = _merge_rank_json_arrays(nfes_path, nfes_shard_paths, expected_counts, args.num_samples)

    if args.skip_mauve:
        mauve_score = -1.0
    else:
        mauve_score = _compute_mauve(args, corrected_texts, reference_features_path, device_id)

    correction_nfes_arr = np.array(correction_nfes, dtype=np.float64)
    metrics = {
        "model_name": args.model_name,
        "budget": budget,
        "num_samples": len(corrected_texts),
        "mauve": mauve_score,
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

    print(f"RANK{rank}: Loading pretrained GIDD model {args.model_name} on {device}")
    pipe = GiddPipeline.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        compile_step=args.compile_step,
    )
    pipe.to(device)
    pipe.eval()

    for budget in budgets:
        pipe.to(device)
        _set_seed(args.seed + budget * 1000 + rank)
        _run_budget(pipe, args, budget, dtype, device_id, rank, world_size)

    if _is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
