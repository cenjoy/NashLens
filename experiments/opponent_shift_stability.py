#!/usr/bin/env python3
"""
Opponent-Shift Explanation-Stability Experiment (NashLens vs. post-hoc XAI)
===========================================================================

Reviewer request (优化建议 2): compare the *stability* of NashLens factor
attributions against post-hoc explainers (KernelSHAP, Integrated Gradients,
gradient saliency) when the *assumed opponent strategy* changes — a property
that generic XRL-Bench metrics (AIM/AUM/PGI/PGU/RIS) do not probe, because
those metrics fix a single environment/opponent.

Why this matters
----------------
In an imperfect-information game, a post-hoc explainer must answer the
counterfactual "what if this feature were absent?". To do so it needs a
*reference distribution* (KernelSHAP's background set; Integrated Gradients'
baseline). The only honest choice for that reference is the distribution of
states the *opponent* drives the agent into. Hence a post-hoc explanation is
implicitly conditioned on an assumed opponent. When the opponent's strategy
shifts (aggressive -> passive), the reference shifts, and the *same decision*
receives a *different* explanation — even though the agent's policy is fixed.

NashLens, by contrast, reads the policy's own additive logit decomposition
( z_reward + z_opponent + z_intrinsic ). This attribution is a function of the
policy parameters and the query state ALONE; it never consults a background or
baseline distribution, so it is invariant to the assumed opponent *by
construction*. Gradient saliency (input x gradient) shares this reference-free
property. The experiment below MEASURES the resulting drift.

What is fabricated: nothing. The additive policy is *distilled* (closed-form /
gradient descent in pure Python) to match the analytic near-Nash Kuhn target
policy — the same distillation objective as the paper's method, just torch-free
so it runs anywhere. Shapley values are computed EXACTLY by enumerating all
2^d feature coalitions (d = 4); Integrated Gradients uses a 128-step Riemann
sum. Stability is reported as cross-context cosine similarity, mean +/- std
over query states. NashLens / gradient stability is 1.0 because their
attribution literally does not depend on the opponent context; SHAP / IG drift
is whatever we measure.

Dependencies: Python 3 standard library only (math, itertools, json, random).
Deterministic: fixed seed; no randomness in the reported attributions
(backgrounds are enumerated grids, not Monte-Carlo samples).

Run:
    python3 opponent_shift_stability.py
Writes results to ../results/opponent_shift_stability.json
"""

import json
import math
import os
from itertools import combinations, product

# --------------------------------------------------------------------------
# 1. Feature encoding for a Kuhn-poker decision state.
#    x = [hand, facing_bet, pot, committed]
#      hand      in {0.0, 0.5, 1.0}  (J, Q, K)            -> reward factor
#      facing_bet in {0.0, 1.0}      (opponent has bet)   -> opponent factor
#      pot       in {0.0, 1.0}       (normalized pot size) -> opponent factor
#      committed in {0.0, 1.0}       (we already put chips in)-> intrinsic factor
#    The action explained is P(bet / call) = sigmoid(logit).
# --------------------------------------------------------------------------
FEATURES = ["hand", "facing_bet", "pot", "committed"]
D = len(FEATURES)

# Factor membership: which features feed which NashLens factor.
REWARD_IDX = [0]            # hand strength  -> z_reward
OPPONENT_IDX = [1, 2]       # facing_bet,pot -> z_opponent
INTRINSIC_IDX = [3]         # committed/bias -> z_intrinsic


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# --------------------------------------------------------------------------
# 2. Additive three-factor policy (the NashLens student form), realised as a
#    generalised additive model WITH INTERACTIONS over the four raw features.
#    Interactions are not cosmetic: in poker the value of calling depends on
#    hand strength and pot odds *jointly* (hand x facing_bet, hand x pot), so a
#    faithful policy must be nonlinear in the raw inputs. It is exactly this
#    nonlinearity that makes a post-hoc explainer's answer depend on the
#    reference distribution (and hence on the assumed opponent).
#
#    Basis  phi(x) = [hand, hand^2, facing_bet, pot, committed,
#                     hand*facing_bet, hand*pot]
#    logit  = theta . phi(x) + b
#    Factor assignment (NashLens decomposition):
#       z_reward    = theta_hand*hand + theta_hand2*hand^2
#       z_opponent  = theta_fb*fb + theta_pot*pot
#                     + theta_hxfb*hand*fb + theta_hxpot*hand*pot
#       z_intrinsic = theta_com*committed + b
#    We DISTILL theta to the analytic near-Nash Kuhn target (cross-entropy GD).
# --------------------------------------------------------------------------
NB = 7  # basis dimension


def basis(x):
    h, fb, pot, com = x
    return [h, h * h, fb, pot, com, h * fb, h * pot]


def kuhn_target_states():
    """Decision states with near-Nash P(bet/call) targets.

    Targets follow the analytic Kuhn equilibrium family (alpha in [0,1/3]); we
    take alpha = 1/3. Probabilities are exact rationals from the standard Kuhn
    solution (bet/bluff strong/weak hands; call in proportion to hand x pot
    odds), not hand-waved.
    """
    a = 1.0 / 3.0
    states = [
        # (hand, facing_bet, pot, committed) : target P(bet or call)
        ((0.0, 0.0, 0.0, 0.0), a),          # J, open: bluff-bet at rate alpha
        ((0.5, 0.0, 0.0, 0.0), 0.0),        # Q, open: check
        ((1.0, 0.0, 0.0, 0.0), 3.0 * a),    # K, open: bet (=1.0)
        ((0.0, 1.0, 1.0, 1.0), 0.0),        # J, facing bet: fold
        ((0.5, 1.0, 1.0, 1.0), a),          # Q, facing bet: call at rate alpha
        ((1.0, 1.0, 1.0, 1.0), 1.0),        # K, facing bet: call
    ]
    return states


def distill_additive_policy(epochs=20000, lr=0.5):
    """Fit logit = theta.phi(x) + b to the Kuhn targets via full-batch GD."""
    th = [0.0] * NB
    b = 0.0
    data = kuhn_target_states()
    n = len(data)
    for _ in range(epochs):
        gth = [0.0] * NB
        gb = 0.0
        for x, t in data:
            ph = basis(x)
            z = sum(th[j] * ph[j] for j in range(NB)) + b
            p = sigmoid(z)
            err = p - t
            for j in range(NB):
                gth[j] += err * ph[j]
            gb += err
        for j in range(NB):
            th[j] -= lr * gth[j] / n
        b -= lr * gb / n
    return th, b


def factor_logits(x, th, b):
    """NashLens additive decomposition: returns (z_reward, z_opp, z_intrinsic)."""
    h, fb, pot, com = x
    z_r = th[0] * h + th[1] * h * h
    z_o = th[2] * fb + th[3] * pot + th[5] * h * fb + th[6] * h * pot
    z_i = th[4] * com + b
    return z_r, z_o, z_i


def model_logit(x, th, b):
    ph = basis(x)
    return sum(th[j] * ph[j] for j in range(NB)) + b


def model_prob(x, th, b):
    return sigmoid(model_logit(x, th, b))


# --------------------------------------------------------------------------
# 3. Opponent contexts -> background state distributions.
#    Each context is a list of (state, weight) describing the distribution of
#    states the *opponent* drives the agent into. KernelSHAP marginalises
#    absent features over this background; IG uses its mean as the baseline.
# --------------------------------------------------------------------------
def opponent_contexts():
    hands = [0.0, 0.5, 1.0]
    # uniform: every (facing_bet,pot,committed) combination equally likely
    uniform = []
    for h, fb, pot, com in product(hands, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]):
        uniform.append(((h, fb, pot, com), 1.0))
    # aggressive opponent: bets often -> agent usually faces a bet, pot/committed high
    aggressive = []
    for h in hands:
        aggressive.append(((h, 1.0, 1.0, 1.0), 0.80))
        aggressive.append(((h, 0.0, 0.0, 0.0), 0.20))
    # passive opponent: rarely bets -> agent usually acts first, pot/committed low
    passive = []
    for h in hands:
        passive.append(((h, 0.0, 0.0, 0.0), 0.80))
        passive.append(((h, 1.0, 1.0, 1.0), 0.20))
    return {"uniform": uniform, "aggressive": aggressive, "passive": passive}


def normalize(bg):
    tot = sum(wt for _, wt in bg)
    return [(s, wt / tot) for s, wt in bg]


# --------------------------------------------------------------------------
# 4. EXACT Shapley values (KernelSHAP's estimand) for the query state x.
#    Value of a coalition S = E_background[ f(x_S , X_{not S}) ], i.e. present
#    features take their query values, absent features are integrated over the
#    background. With d=4 we enumerate all 2^d coalitions exactly.
# --------------------------------------------------------------------------
def coalition_value(present, x, w, b, bg):
    """E over background of f where features in `present` are fixed to x."""
    val = 0.0
    for s_bg, wt in bg:
        xx = [x[j] if j in present else s_bg[j] for j in range(D)]
        val += wt * model_prob(xx, w, b)
    return val


def exact_shapley(x, w, b, bg):
    phi = [0.0] * D
    others = list(range(D))
    for j in range(D):
        rest = [k for k in others if k != j]
        for r in range(len(rest) + 1):
            for S in combinations(rest, r):
                Sset = set(S)
                weight = (math.factorial(len(S)) *
                          math.factorial(D - len(S) - 1) / math.factorial(D))
                v_with = coalition_value(Sset | {j}, x, w, b, bg)
                v_without = coalition_value(Sset, x, w, b, bg)
                phi[j] += weight * (v_with - v_without)
    return phi


# --------------------------------------------------------------------------
# 5. Integrated Gradients w.r.t. the context baseline (background mean state).
# --------------------------------------------------------------------------
def background_mean(bg):
    m = [0.0] * D
    for s, wt in bg:
        for j in range(D):
            m[j] += wt * s[j]
    return m


def integrated_gradients(x, th, b, baseline, steps=128):
    phi = [0.0] * D
    for j in range(D):
        acc = 0.0
        for k in range(steps):
            t = (k + 0.5) / steps
            xt = [baseline[i] + t * (x[i] - baseline[i]) for i in range(D)]
            # d logit / d x_j  (raw feature j), through the basis with interactions
            h, fb, pot, com = xt
            if j == 0:    # hand: appears in hand, hand^2, hand*fb, hand*pot
                dlogit = th[0] + 2 * th[1] * h + th[5] * fb + th[6] * pot
            elif j == 1:  # facing_bet: fb, hand*fb
                dlogit = th[2] + th[5] * h
            elif j == 2:  # pot: pot, hand*pot
                dlogit = th[3] + th[6] * h
            else:         # committed
                dlogit = th[4]
            p = model_prob(xt, th, b)
            acc += p * (1.0 - p) * dlogit
        phi[j] = (x[j] - baseline[j]) * acc / steps
    return phi


# --------------------------------------------------------------------------
# 6. Reference-free attributions (invariant to opponent context by construction).
# --------------------------------------------------------------------------
def nashlens_attribution(x, th, b):
    """Per-feature contribution from the additive decomposition (no background).

    Interaction terms (hand*fb, hand*pot) belong to the opponent factor and are
    credited to the feature that carries opponent pressure (facing_bet, pot),
    matching how NashLens routes them through z_opponent."""
    h, fb, pot, com = x
    return [
        th[0] * h + th[1] * h * h,            # hand   -> reward factor
        th[2] * fb + th[5] * h * fb,          # facing_bet -> opponent factor
        th[3] * pot + th[6] * h * pot,        # pot    -> opponent factor
        th[4] * com,                          # committed -> intrinsic factor
    ]


def gradient_saliency(x, th, b):
    """input x gradient of the logit at x; reference-free."""
    h, fb, pot, com = x
    grad = [
        th[0] + 2 * th[1] * h + th[5] * fb + th[6] * pot,  # d/d hand
        th[2] + th[5] * h,                                 # d/d facing_bet
        th[3] + th[6] * h,                                 # d/d pot
        th[4],                                             # d/d committed
    ]
    return [grad[j] * x[j] for j in range(D)]


# --------------------------------------------------------------------------
# 7. Stability metric: cross-context cosine similarity of attribution vectors.
# --------------------------------------------------------------------------
def cosine(a, c):
    num = sum(ai * ci for ai, ci in zip(a, c))
    na = math.sqrt(sum(ai * ai for ai in a))
    nc = math.sqrt(sum(ci * ci for ci in c))
    if na == 0 or nc == 0:
        return 1.0 if na == nc else 0.0
    return num / (na * nc)


def l1_normalize(phi):
    s = sum(abs(v) for v in phi)
    if s == 0:
        return [0.0] * len(phi)
    return [v / s for v in phi]


def l2_drift(a, c):
    """L2 distance between L1-normalised attributions (captures redistribution
    of importance across features, which cosine misses when vectors only
    rescale). 0 = identical importance profile."""
    na, nc = l1_normalize(a), l1_normalize(c)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(na, nc)))


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / n
    return m, math.sqrt(v)


def main():
    th, b = distill_additive_policy()
    contexts = {k: normalize(v) for k, v in opponent_contexts().items()}
    cnames = list(contexts.keys())

    # Query states we ask each method to explain (representative decisions).
    queries = [s for s, _ in kuhn_target_states()]

    methods = {
        "NashLens (factor, reference-free)": "nashlens",
        "GradientSaliency (reference-free)": "gradient",
        "KernelSHAP (background-dependent)": "shap",
        "IntegratedGradients (baseline-dependent)": "ig",
    }

    results = {"policy_weights": {"theta": th, "b": b},
               "basis": ["hand", "hand^2", "facing_bet", "pot", "committed",
                         "hand*facing_bet", "hand*pot"],
               "distill_fit": [], "methods": {}}

    for x, t in kuhn_target_states():
        results["distill_fit"].append(
            {"state": x, "target": round(t, 4), "policy": round(model_prob(x, th, b), 4)})

    for label, key in methods.items():
        per_query_cos = []
        per_query_l2 = []
        per_query_attr = {}
        for x in queries:
            attrs = {}
            for c in cnames:
                bg = contexts[c]
                if key == "nashlens":
                    phi = nashlens_attribution(x, th, b)
                elif key == "gradient":
                    phi = gradient_saliency(x, th, b)
                elif key == "shap":
                    phi = exact_shapley(x, th, b, bg)
                elif key == "ig":
                    phi = integrated_gradients(x, th, b, background_mean(bg))
                attrs[c] = phi
            cos, l2 = [], []
            for i in range(len(cnames)):
                for j in range(i + 1, len(cnames)):
                    cos.append(cosine(attrs[cnames[i]], attrs[cnames[j]]))
                    l2.append(l2_drift(attrs[cnames[i]], attrs[cnames[j]]))
            per_query_cos.append(sum(cos) / len(cos))
            per_query_l2.append(sum(l2) / len(l2))
            per_query_attr[str(x)] = {c: [round(v, 4) for v in attrs[c]] for c in cnames}
        cm, csd = mean_std(per_query_cos)
        lm, lsd = mean_std(per_query_l2)
        results["methods"][label] = {
            "cosine_stability_mean": round(cm, 4),
            "cosine_stability_std": round(csd, 4),
            "cosine_stability_min": round(min(per_query_cos), 4),
            "l2_drift_mean": round(lm, 4),
            "l2_drift_max": round(max(per_query_l2), 4),
            "per_query_cosine": [round(s, 4) for s in per_query_cos],
            "per_query_l2": [round(s, 4) for s in per_query_l2],
            "attributions": per_query_attr,
        }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "opponent_shift_stability.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=== Opponent-Shift Explanation Stability (Kuhn) ===")
    print("Distillation fit (target -> policy):")
    for d in results["distill_fit"]:
        print(f"  {d['state']}  target={d['target']:.3f}  policy={d['policy']:.3f}")
    print("\nCross-context attribution stability over 3 opponent contexts")
    print("(cosine: 1.0 = invariant direction; L2 drift: 0 = invariant importance):")
    print(f"{'method':<42} {'cos.mean':>9} {'cos.min':>8} {'L2.mean':>8} {'L2.max':>8}")
    for label, r in results["methods"].items():
        print(f"{label:<42} {r['cosine_stability_mean']:>9.3f} "
              f"{r['cosine_stability_min']:>8.3f} {r['l2_drift_mean']:>8.3f} "
              f"{r['l2_drift_max']:>8.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
