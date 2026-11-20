"""Phase 0 throughput benchmark against a running vLLM OpenAI-compatible server.

Measures generated tokens/sec at several concurrency levels using the server-reported
`usage.completion_tokens` (not a client-side tokenizer estimate). Writes one JSON file
with every per-request record so the summary numbers can be recomputed.

Usage:
    python scripts/bench_vllm.py --model Qwen/Qwen2.5-7B-Instruct --out results/phase0/bench.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import time
from pathlib import Path

from openai import AsyncOpenAI

# Fixed, varied prompts that elicit long answers, so max_tokens is normally reached.
TOPICS = [
    "the causes of the French Revolution", "how vaccines train the immune system",
    "the history of the Roman aqueducts", "how a transformer neural network works",
    "the life cycle of a star", "the economics of inflation",
    "plate tectonics and earthquakes", "the development of the printing press",
    "how public-key cryptography works", "the water cycle",
    "the biography of Marie Curie", "the rules and history of cricket",
    "photosynthesis in plants", "the Apollo program",
    "how the internet routes packets", "the Industrial Revolution in Britain",
]


def prompt(i: int) -> str:
    return f"Write a detailed, factual, multi-paragraph explanation of {TOPICS[i % len(TOPICS)]}. (Request {i}.)"


async def one(client: AsyncOpenAI, model: str, i: int, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    r = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt(i)}],
        max_tokens=max_tokens,
        temperature=0.0,
        seed=0,
    )
    dt = time.perf_counter() - t0
    return {
        "i": i,
        "latency_s": dt,
        "prompt_tokens": r.usage.prompt_tokens,
        "completion_tokens": r.usage.completion_tokens,
        "finish_reason": r.choices[0].finish_reason,
    }


async def run_level(client: AsyncOpenAI, model: str, concurrency: int, n: int, max_tokens: int) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(i: int) -> dict:
        async with sem:
            return await one(client, model, i, max_tokens)

    t0 = time.perf_counter()
    recs = await asyncio.gather(*(guarded(i) for i in range(n)))
    wall = time.perf_counter() - t0
    out_tok = sum(r["completion_tokens"] for r in recs)
    lat = sorted(r["latency_s"] for r in recs)
    return {
        "concurrency": concurrency,
        "n_requests": n,
        "max_tokens": max_tokens,
        "wall_s": wall,
        "completion_tokens_total": out_tok,
        "prompt_tokens_total": sum(r["prompt_tokens"] for r in recs),
        "throughput_out_tok_per_s": out_tok / wall,
        "per_request_out_tok_per_s_mean": sum(r["completion_tokens"] / r["latency_s"] for r in recs) / n,
        "latency_s_mean": sum(lat) / n,
        "latency_s_p50": lat[n // 2],
        "latency_s_max": lat[-1],
        "requests": recs,
    }


def gpu_snapshot() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {e}"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--levels", default="1,8,32,64", help="comma-separated concurrency levels")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    # Warm-up (not measured): first request pays one-off costs.
    await one(client, args.model, 0, 32)

    levels = []
    for c in [int(x) for x in args.levels.split(",")]:
        n = max(8, 2 * c)  # at least 2 full waves per level
        res = await run_level(client, args.model, c, n, args.max_tokens)
        res["gpu_after"] = gpu_snapshot()
        levels.append(res)
        print(f"concurrency={c:>3} n={n:>3} wall={res['wall_s']:.1f}s "
              f"throughput={res['throughput_out_tok_per_s']:.1f} out tok/s "
              f"per-request={res['per_request_out_tok_per_s_mean']:.1f} tok/s "
              f"lat_mean={res['latency_s_mean']:.2f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "base_url": args.base_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "levels": levels,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
