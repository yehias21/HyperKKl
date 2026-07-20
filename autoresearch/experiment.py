"""FROZEN experiment harness for autoresearch on HyperKKL.

This file is the ground truth. The research agent MUST NOT modify it, and must not
modify src/systems.py or src/signals.py. Everything the agent optimizes lives in
src/models.py, src/training.py, src/dataset.py and configs/.

What it does:
  1. Trains phase 1 (autonomous T / T*) and the requested phase-2 method(s),
     by calling the (agent-editable) entry points in src.training.
  2. Scores the resulting observer under a FIXED evaluation protocol: fixed
     initial conditions, fixed input-signal parameters, fixed settle time.
  3. Prints one summary block with a single primary metric.

Primary metric: mean steady-state RMSE over {systems} x {input regimes} x {id, ood}.
Lower is better. RMSE (not SMAPE) is primary on purpose: SMAPE is unstable near the
zero crossings these oscillatory systems spend most of their time near.

Dev / test split: the overnight loop scores on --split dev. A disjoint set of ICs and
signal-parameter seeds is reserved for --split test, which is run only by the human
at the end, to check that overnight gains are real and not eval overfitting.

This study is HyperKKL_dyn ONLY: --method augmented (obs) is disabled by a scope guard.

Usage:
  python -m autoresearch.experiment --systems duffing --method lora
  python -m autoresearch.experiment --systems duffing vdp --method lora --split test
  python -m autoresearch.experiment --systems duffing --method lora --ekf
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")  # keep run.log readable for the agent

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smt.sampling_methods import LHS  # noqa: E402

from src.config import load_config  # noqa: E402
from src.systems import create_system  # noqa: E402
from src.signals import create_signal  # noqa: E402
from src.evaluation import simulate_observer, simulate_true_system  # noqa: E402
from src.training import train_phase1, train_augmented, train_dynamic  # noqa: E402
from src.dataset import generate_phase2_data  # noqa: E402

# --- integrity guard -------------------------------------------------------
# The agent may edit the METHOD (simulate_observer, models, training, data), but the
# ground truth must stay fixed. We hash the frozen surfaces and refuse to score if they
# changed, so a 100-experiment unattended run cannot drift into measuring itself.
# Regenerate deliberately with:  python -m autoresearch.experiment --bless

MANIFEST = Path(__file__).parent / "integrity.json"


def _frozen_digests() -> dict:
    import src.evaluation as ev
    import src.systems as sysmod
    import src.signals as sig

    def fh(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def sh(fn):
        return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()

    return {
        "src/systems.py": fh(ROOT / "src/systems.py"),
        "src/signals.py": fh(ROOT / "src/signals.py"),
        "evaluation.simulate_true_system": sh(ev.simulate_true_system),
        "evaluation.compute_rmse": sh(ev.compute_rmse),
        "systems.create_system": sh(sysmod.create_system),
        "signals.create_signal": sh(sig.create_signal),
    }


def check_integrity(bless: bool = False):
    cur = _frozen_digests()
    if bless or not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(cur, indent=2))
        print(f"[integrity] manifest written to {MANIFEST}")
        return
    ref = json.loads(MANIFEST.read_text())
    bad = [k for k in cur if ref.get(k) != cur[k]]
    if bad:
        raise SystemExit(
            "[integrity] FROZEN GROUND TRUTH WAS MODIFIED: " + ", ".join(bad) +
            "\nThese define the systems, the signals, the true trajectory and the metric. "
            "Results computed after changing them are meaningless. Revert with "
            "'git checkout -- <file>' and re-run. If the change was deliberate and "
            "correct, the human must re-bless with --bless and restart the study."
        )


# --- phase-1 cache ---------------------------------------------------------
# Phase 1 is retrained from scratch on every run, but it only depends on the phase-1 code
# and config -- not on anything a phase-2/hypernetwork experiment changes. We cache it,
# keyed on a hash of every source file that can affect it, so a phase-2 experiment reuses
# the identical base maps. Any edit to phase-1 code or config changes the key and forces a
# retrain, so the cache can never go stale. Bonus: reusing identical base maps removes
# phase-1 run-to-run noise from phase-2 comparisons, which makes small effects readable.

PHASE1_CACHE = Path(__file__).parent / "phase1_cache"
_PHASE1_INPUTS = ["src/models.py", "src/training.py", "src/dataset.py", "src/config.py",
                  "src/systems.py", "configs/default.yaml"]


def _phase1_key(sys_name: str, seed: int) -> str:
    h = hashlib.sha256()
    for rel in _PHASE1_INPUTS + [f"configs/systems/{sys_name}.yaml"]:
        fp = ROOT / rel
        if fp.exists():
            h.update(fp.read_bytes())
    h.update(f"{sys_name}|{seed}".encode())
    return h.hexdigest()[:16]


def get_phase1(system, sys_config, cfg, device, sys_name: str, seed: int):
    """Train phase 1, or reuse a cached copy when nothing affecting it has changed."""
    PHASE1_CACHE.mkdir(exist_ok=True)
    fp = PHASE1_CACHE / f"{sys_name}_{_phase1_key(sys_name, seed)}.pt"
    if fp.exists():
        try:
            blob = torch.load(fp, map_location=device, weights_only=False)
            print(f"[phase1] cache HIT {fp.name} (skipped retraining)")
            return blob["T"].to(device), blob["T_inv"].to(device)
        except Exception as e:
            print(f"[phase1] cache unreadable ({e}); retraining")
    t0 = time.time()
    T_net, T_inv_net, _ = train_phase1(system, sys_config, cfg, device, None)
    print(f"[phase1] cache MISS -> trained in {time.time() - t0:.0f}s, saving {fp.name}")
    try:
        torch.save({"T": T_net, "T_inv": T_inv_net}, fp)
    except Exception as e:
        print(f"[phase1] could not cache: {e}")
    return T_net, T_inv_net


# --- fixed evaluation protocol (do not change) -----------------------------
SPLIT_SEEDS = {"dev": 1234, "test": 9876}
N_TRIALS = 8
SETTLE_TIME = 5.0
REGIMES = ["zero", "constant", "sinusoid", "square"]
MODES = ["id", "ood"]


def _fixed_test_ics(sys_config, split: str, n: int) -> np.ndarray:
    """Deterministic ICs per (system, split). LHS is seeded through numpy global state."""
    np.random.seed(SPLIT_SEEDS[split] + abs(hash(sys_config.name)) % 1000)
    sampler = LHS(xlimits=sys_config.limits_np, random_state=SPLIT_SEEDS[split])
    return sampler(n)


def score_models(system, sys_config, models: dict, method_name: str,
                 device, window_size: int, split: str) -> dict:
    """Fixed-protocol scoring. Returns {mode: {regime: rmse_steady}}."""
    ics = _fixed_test_ics(sys_config, split, N_TRIALS)
    out = {}
    for mode in MODES:
        out[mode] = {}
        for regime in REGIMES:
            rng = np.random.RandomState(SPLIT_SEEDS[split])
            sig_gen = create_signal(regime, mode)
            vals = []
            for ic in ics:
                sig_gen.sample_params(rng)
                try:
                    x_true, x_hat, t = simulate_observer(
                        system, sys_config, ic, sig_gen, device,
                        window_size=window_size, **models)
                    err = np.linalg.norm(x_true - x_hat, axis=1)
                    s = np.searchsorted(t, SETTLE_TIME)
                    vals.append(float(np.sqrt(np.mean(err[s:] ** 2))))
                except Exception as e:  # a broken observer scores as failure, not a crash
                    print(f"    [score] {method_name}/{mode}/{regime} trial failed: {e}")
            out[mode][regime] = float(np.mean(vals)) if vals else float("nan")
    return out


# --- EKF reference ---------------------------------------------------------
# The model-based baseline CDC reviewers asked for (uses exactly the same f(x,u)
# knowledge the PDE loss requires). Reference line only - never optimized against
# directly, but it tells the agent how large the remaining gap is.

def ekf_score(system, sys_config, split: str, q: float = 1e-3, r: float = 1e-3) -> dict:
    ics = _fixed_test_ics(sys_config, split, N_TRIALS)
    n = sys_config.x_size
    dt = sys_config.dt
    out = {}
    for mode in MODES:
        out[mode] = {}
        for regime in REGIMES:
            rng = np.random.RandomState(SPLIT_SEEDS[split])
            sig_gen = create_signal(regime, mode)
            vals = []
            for ic in ics:
                sig_gen.sample_params(rng)
                try:
                    x_true, y, u_vals, t = simulate_true_system(system, sys_config, ic, sig_gen)
                    xh = np.zeros(n)
                    P = np.eye(n)
                    Q = q * np.eye(n)
                    R = r * np.eye(sys_config.y_size)
                    est = [xh.copy()]
                    for i in range(len(t) - 1):
                        xt = torch.tensor(xh, dtype=torch.float64).unsqueeze(0).requires_grad_(True)
                        ut = torch.tensor(np.atleast_1d(u_vals[i]), dtype=torch.float64)
                        fx = system.function(t[i], ut, xt)
                        F = torch.autograd.functional.jacobian(
                            lambda z: system.function(t[i], ut, z).squeeze(0),
                            xt.detach()).reshape(n, n).numpy()
                        xh = xh + dt * fx.detach().numpy().reshape(n)
                        Ad = np.eye(n) + dt * F
                        P = Ad @ P @ Ad.T + Q * dt
                        # measurement update
                        xt2 = torch.tensor(xh, dtype=torch.float64).unsqueeze(0)
                        H = torch.autograd.functional.jacobian(
                            lambda z: system.output(z).reshape(-1), xt2
                        ).reshape(sys_config.y_size, n).numpy()
                        yh = system.output(torch.tensor(xh, dtype=torch.float64).unsqueeze(0)).numpy().reshape(-1)
                        S = H @ P @ H.T + R
                        K = P @ H.T @ np.linalg.inv(S)
                        xh = xh + K @ (np.asarray(y[i + 1]).reshape(-1) - yh)
                        P = (np.eye(n) - K @ H) @ P
                        est.append(xh.copy())
                    est = np.array(est)
                    err = np.linalg.norm(x_true - est, axis=1)
                    s = np.searchsorted(t, SETTLE_TIME)
                    vals.append(float(np.sqrt(np.mean(err[s:] ** 2))))
                except Exception as e:
                    print(f"    [ekf] {mode}/{regime} failed: {e}")
            out[mode][regime] = float(np.mean(vals)) if vals else float("nan")
    return out


def _mean(score: dict) -> float:
    vals = [v for mode in score.values() for v in mode.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", default=["duffing", "vdp"])
    p.add_argument("--method", default="lora", choices=["lora", "full", "augmented"])
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--seed", type=int, default=42, help="TRAINING seed (eval seeds are fixed)")
    p.add_argument("--ekf", action="store_true", help="also compute the EKF reference")
    p.add_argument("--config_dir", default=None)
    p.add_argument("--epochs2", type=int, default=None,
                   help="override phase-2 epochs (screening runs; e.g. 10). Screening "
                        "numbers are comparable only to other screening numbers.")
    p.add_argument("--lane", default=None,
                   help="cosmetic lane tag so `ps` can tell concurrent lanes apart; "
                        "NEVER kill a process belonging to another lane")
    p.add_argument("--bless", action="store_true",
                   help="HUMAN ONLY: re-record the integrity manifest and exit")
    args = p.parse_args()

    if args.bless:
        check_integrity(bless=True)
        return
    check_integrity()

    # dyn-only: the human restricted this study to HyperKKL_dyn. Enforced here rather than
    # only in program.md so it binds even a session that read an earlier version of it.
    if args.method == "augmented":
        raise SystemExit(
            "[scope] This study is HyperKKL_dyn ONLY. '--method augmented' (HyperKKL_obs) "
            "is disabled. Use --method lora. The obs reference numbers are already recorded "
            "in program.md; do not spend a run regenerating them."
        )

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    per_system, per_system_auto, per_system_ekf = {}, {}, {}

    for sys_name in args.systems:
        cfg = load_config(sys_name, args.config_dir)
        if args.epochs2 is not None:
            cfg.phase2.epochs = args.epochs2
        sys_config = cfg.system
        system = create_system(sys_config)
        ws = cfg.phase2.window_size

        T_net, T_inv_net = get_phase1(system, sys_config, cfg, device, sys_name, args.seed)

        auto = score_models(system, sys_config,
                            {"method_type": "autonomous", "T_inv": T_inv_net},
                            "autonomous", device, ws, args.split)
        per_system_auto[sys_name] = auto

        train_data = generate_phase2_data(sys_config, sys_config.natural_inputs,
                                          cfg.phase2.n_train_traj, ws, args.seed)

        T_c, T_inv_c = copy.deepcopy(T_net), copy.deepcopy(T_inv_net)
        if args.method == "augmented":
            enc, phi, _ = train_augmented(T_c, T_inv_c, sys_config, train_data, cfg, device, None)
            models = {"method_type": "augmented", "T_inv": T_inv_c, "encoder": enc, "phi_net": phi}
        else:
            hypernet, T_mod, Tinv_mod, _ = train_dynamic(
                T_c, T_inv_c, sys_config, train_data, cfg, device, args.method, None)
            models = {"method_type": args.method, "hypernet": hypernet,
                      "T_base": T_mod, "T_inv_base": Tinv_mod,
                      "skip_bias": args.method == "lora"}

        per_system[sys_name] = score_models(system, sys_config, models, args.method,
                                            device, ws, args.split)
        if args.ekf:
            per_system_ekf[sys_name] = ekf_score(system, sys_config, args.split)

    primary = float(np.mean([_mean(s) for s in per_system.values()]))
    auto_primary = float(np.mean([_mean(s) for s in per_system_auto.values()]))

    print("\n---")
    print(f"val_rmse:         {primary:.6f}")
    print(f"autonomous_rmse:  {auto_primary:.6f}")
    if args.ekf and per_system_ekf:
        print(f"ekf_rmse:         {float(np.mean([_mean(s) for s in per_system_ekf.values()])):.6f}")
    for s, sc in per_system.items():
        print(f"  {s}: id={_mean({'id': sc['id']}):.4f} ood={_mean({'ood': sc['ood']}):.4f} "
              + " ".join(f"{r}={sc['ood'][r]:.3f}" for r in REGIMES))
    print(f"method:           {args.method}")
    print(f"systems:          {','.join(args.systems)}")
    print(f"split:            {args.split}")
    print(f"epochs2:          {args.epochs2 if args.epochs2 is not None else 'full'}")
    print(f"total_seconds:    {time.time() - t0:.1f}")

    # Archive every run's full breakdown. last_score.json is overwritten each run, so
    # without this the per-regime id/ood detail of every prior experiment is lost and
    # post-hoc analysis (e.g. "did this change help ood more than id?") is impossible.
    try:
        import subprocess
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=str(ROOT)).stdout.strip() or "nogit"
        arch = Path(__file__).parent / "scores_archive"
        arch.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tag = f"{stamp}_{sha}_{args.method}_{'-'.join(args.systems)}_{args.split}"
        if args.epochs2 is not None:
            tag += f"_e2-{args.epochs2}"
        (arch / f"{tag}.json").write_text(json.dumps(
            {"primary": primary, "autonomous": auto_primary, "per_system": per_system,
             "per_system_autonomous": per_system_auto,
             "ekf": per_system_ekf, "split": args.split, "method": args.method,
             "systems": args.systems, "epochs2": args.epochs2, "seed": args.seed,
             "commit": sha}, indent=2))
    except Exception as e:
        print(f"[archive] could not archive score: {e}")

    Path("last_score.json").write_text(json.dumps(
        {"primary": primary, "autonomous": auto_primary, "per_system": per_system,
         "ekf": per_system_ekf, "split": args.split, "method": args.method}, indent=2))


if __name__ == "__main__":
    main()
