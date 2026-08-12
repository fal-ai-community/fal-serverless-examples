"""Fleet demo: register N synthetic users, drive hinted traffic, print the
routing story. This is the empirical validation for hint-steered cache
affinity at scale.

Usage:
    pip install fal-client==1.0.0
    python fleet_demo.py --app <username>/per-user-loras --users 1000 --requests 200
    python fleet_demo.py --app <username>/per-user-loras --users 1000 --requests 200 --no-hints

Requires FAL_KEY (or `fal auth login`). Scale the app to >=2 runners first to
see affinity: fal apps scale per-user-loras --min-concurrency 2
"""
import argparse
import collections
import concurrent.futures as cf
import random
import threading

import fal_client

PROMPTS = [
    "a portrait in this user's signature style",
    "a mountain village at golden hour",
    "a robot playing chess",
    "a bowl of ramen, studio lighting",
]


def user_ids(n: int) -> list[str]:
    return [f"load-u{i:05d}" for i in range(n)]


def register_users(app: str, users: list[str], workers: int) -> dict[str, str]:
    """Register the users; return each server-issued routing hint."""
    hints: dict[str, str] = {}
    lock = threading.Lock()

    def _reg(uid: str) -> None:
        reg = fal_client.run(app, {"user_id": uid}, path="/register")
        with lock:
            hints[uid] = reg["hint"]

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        done = 0
        for _ in pool.map(_reg, users):
            done += 1
            if done % 100 == 0:
                print(f"registered {done}/{len(users)}")
    return hints


def drive_traffic(app: str, users: list[str], hints: dict[str, str],
                  n_requests: int, use_hints: bool, workers: int, seed: int,
                  zipf: float, use_queue: bool = False) -> tuple[list[dict], dict[str, int]]:
    # Zipf-skewed popularity: a few hot users, long tail. Within a keep_alive
    # window real avatar traffic is concentrated (active users iterate in
    # edit-generate loops), so the default exponent leans hot; lower --zipf
    # toward 1.0 to model colder, flatter traffic.
    weights = [1.0 / (rank + 1) ** zipf for rank in range(len(users))]

    # Repeat-request worker stickiness: for a user's Nth request, did it land
    # on the same worker as their (N-1)th? This measures routing steering
    # directly, independent of adapter-cache (LRU) state. Random placement
    # gives ~1/num_workers; perfect affinity gives ~1.0.
    last_worker: dict[str, str] = {}
    stickiness = {"repeats": 0, "same_worker": 0}
    stats_lock = threading.Lock()

    def _one(i: int) -> dict:
        # Per-request RNG keyed on (seed, i): a shared generator would make
        # draw order depend on thread interleaving and break --seed replays.
        rng = random.Random(f"{seed}:{i}")
        uid = rng.choices(users, weights=weights, k=1)[0]
        with stats_lock:
            hint = hints.get(uid)
        # Clients never compute hints: send the server-issued one when we have
        # it (from /register, or captured off an earlier response below).
        kwargs = {"hint": hint} if use_hints and hint else {}
        arguments = {"user_id": uid, "prompt": rng.choice(PROMPTS),
                     "num_inference_steps": 1, "seed": i, "sync_mode": True}
        # subscribe() = queue path (production default); run() = sync path.
        call = fal_client.subscribe if use_queue else fal_client.run
        serving = call(app, arguments, **kwargs)["serving"]
        with stats_lock:
            hints[uid] = serving["hint"]
            prev = last_worker.get(uid)
            if prev is not None:
                stickiness["repeats"] += 1
                if prev == serving["worker_id"]:
                    stickiness["same_worker"] += 1
            last_worker[uid] = serving["worker_id"]
        return serving

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        servings = list(pool.map(_one, range(n_requests)))
    return servings, stickiness


def report(servings: list[dict], use_hints: bool, stickiness: dict | None = None) -> None:
    by_worker: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    total = collections.Counter()
    for s in servings:
        by_worker[s["worker_id"]][s["adapter_source"]] += 1
        total[s["adapter_source"]] += 1
    mode = "WITH hints" if use_hints else "WITHOUT hints (baseline)"
    print(f"\n=== {mode}: {len(servings)} requests, {len(by_worker)} workers ===")
    print(f"{'worker':<10} {'gpu-cache':>9} {'disk':>6} {'downloaded':>11}")
    for wid, c in sorted(by_worker.items()):
        print(f"{wid:<10} {c['gpu-cache']:>9} {c['disk']:>6} {c['downloaded']:>11}")
    n = len(servings)
    print(f"{'TOTAL':<10} {total['gpu-cache']:>9} {total['disk']:>6} {total['downloaded']:>11}")
    print(f"warm-adapter rate: {total['gpu-cache'] / n:.1%}")
    if stickiness and stickiness["repeats"]:
        rate = stickiness["same_worker"] / stickiness["repeats"]
        rand = 1.0 / max(len(by_worker), 1)
        print(f"repeat-request worker stickiness: {rate:.1%} "
              f"({stickiness['same_worker']}/{stickiness['repeats']} repeats; "
              f"random ≈ {rand:.1%}, perfect affinity ≈ 100%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app", required=True, help="<username>/per-user-loras")
    ap.add_argument("--users", type=int, default=1000)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zipf", type=float, default=1.3,
                    help="traffic skew exponent; 1.0 = flat-ish long tail")
    ap.add_argument("--no-hints", action="store_true")
    ap.add_argument("--queue", action="store_true",
                    help="drive via the queue API (subscribe) instead of sync run")
    ap.add_argument("--skip-register", action="store_true",
                    help="reuse users registered by a previous run")
    args = ap.parse_args()

    users = user_ids(args.users)
    hints: dict[str, str] = {}
    if not args.skip_register:
        hints = register_users(args.app, users, args.workers)
    servings, stickiness = drive_traffic(args.app, users, hints, args.requests,
                                         use_hints=not args.no_hints,
                                         workers=args.workers, seed=args.seed,
                                         zipf=args.zipf, use_queue=args.queue)
    report(servings, use_hints=not args.no_hints, stickiness=stickiness)


if __name__ == "__main__":
    main()
