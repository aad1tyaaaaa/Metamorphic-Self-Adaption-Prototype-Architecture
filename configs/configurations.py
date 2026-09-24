"""Single source of truth for MSA configurations, features and compute estimates."""

TOTAL_LAYERS = 12
TOTAL_HEADS = 12
HEAD_DIM = 64
HIDDEN_SIZE = 768
FFN_INNER = 3072
FFN_CHUNKS = 4

DEPTH_MAP = {"shallow": 4, "medium": 8, "deep": 12}
CONFIG_NAMES = ["shallow", "medium", "deep"]

ATTENTION_HEAD_FRACTION = {"full": 1.0, "reduced": 0.5}
FFN_CHUNK_FRACTION = {"full": 1.0, "partial": 0.5}

# Phase 9 configuration library. Kept to three entries so every configuration
# can be calibrated empirically rather than assumed.
CONFIGURATIONS = {
    "shallow": {"depth": 4, "attention_mode": "reduced", "ffn_mode": "partial"},
    "medium": {"depth": 8, "attention_mode": "reduced", "ffn_mode": "partial"},
    "deep": {"depth": 12, "attention_mode": "full", "ffn_mode": "full"},
}

# Phase 6 baseline: identical depths, no attention/FFN adaptation.
DEPTH_ONLY_CONFIGURATIONS = {
    name: {"depth": depth, "attention_mode": "full", "ffn_mode": "full"}
    for name, depth in DEPTH_MAP.items()
}

MECHANISMS = {"depth": DEPTH_ONLY_CONFIGURATIONS, "full": CONFIGURATIONS}

# Phase 3 operating point, selected by controller/threshold_sweep.py
# (results/policy_threshold_sweep.csv): highest agreement with c*(x) on held-out
# prompts, ties broken toward lower FLOPs.
CONFIDENCE_THRESHOLD = 0.40
FALLBACK_CONFIGURATION = "deep"

FEATURES_V1 = ["input_length", "reasoning", "domain", "structure"]
FEATURES_V2 = FEATURES_V1 + [
    "numeric_density",
    "math_expression",
    "question_type",
    "reasoning_steps",
]
FEATURE_SETS = {"v1": FEATURES_V1, "v2": FEATURES_V2}


def resolve_config(name, attention=False, ffn=False):
    """Knobs for a configuration with each adaptation mechanism toggled.

    Depth always applies; attention and FFN adaptation are opt-in so the Phase 15
    ablations (depth only, depth+attention, depth+FFN, all) are expressible.
    """
    if name not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration {name!r}; expected one of {CONFIG_NAMES}")

    adapted = CONFIGURATIONS[name]

    return {
        "depth": DEPTH_MAP[name],
        "attention_mode": adapted["attention_mode"] if attention else "full",
        "ffn_mode": adapted["ffn_mode"] if ffn else "full",
    }


def config_for(name, mechanism="full"):
    """Return the knob dict for a named configuration under a given mechanism."""
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}; expected {list(MECHANISMS)}")

    full = mechanism == "full"
    return resolve_config(name, attention=full, ffn=full)


def estimate_flops(depth, attention_mode="full", ffn_mode="full", seq_len=32,
                   ffn_fraction=None):
    """Analytical forward FLOPs for the transformer stack only.

    Excludes embeddings and the lm_head, which are identical across every
    configuration and would only dilute the reported reduction. Counts a
    multiply-accumulate as 2 FLOPs.
    """
    head_fraction = ATTENTION_HEAD_FRACTION[attention_mode]
    if ffn_fraction is None:
        ffn_fraction = FFN_CHUNK_FRACTION[ffn_mode]

    heads = max(1, round(TOTAL_HEADS * head_fraction))
    active_dim = heads * HEAD_DIM
    inner = max(1, round(FFN_INNER * ffn_fraction))

    qkv = 2 * seq_len * HIDDEN_SIZE * (3 * active_dim)
    scores = 2 * 2 * heads * seq_len * seq_len * HEAD_DIM
    out_proj = 2 * seq_len * active_dim * HIDDEN_SIZE
    mlp = 2 * 2 * seq_len * HIDDEN_SIZE * inner

    return depth * (qkv + scores + out_proj + mlp)


def relative_flops(depth, attention_mode="full", ffn_mode="full", seq_len=32,
                   ffn_fraction=None):
    """Transformer FLOPs as a fraction of the static 12-layer full model."""
    full = estimate_flops(TOTAL_LAYERS, "full", "full", seq_len)
    return estimate_flops(depth, attention_mode, ffn_mode, seq_len, ffn_fraction) / full


def active_parameters(depth, attention_mode="full", ffn_mode="full", ffn_fraction=None):
    """Transformer-block weights actually read by one forward pass.

    A deterministic proxy for the weight-memory footprint of a configuration,
    since CPU peak-memory measurement is not reliable here. Excludes embeddings
    and the lm_head, which every configuration reads.
    """
    if ffn_fraction is None:
        ffn_fraction = FFN_CHUNK_FRACTION[ffn_mode]

    heads = max(1, round(TOTAL_HEADS * ATTENTION_HEAD_FRACTION[attention_mode]))
    active_dim = heads * HEAD_DIM
    inner = max(1, round(FFN_INNER * ffn_fraction))

    layer_norms = 2 * 2 * HIDDEN_SIZE
    attention = HIDDEN_SIZE * 3 * active_dim + 3 * active_dim + active_dim * HIDDEN_SIZE + HIDDEN_SIZE
    mlp = HIDDEN_SIZE * inner + inner + inner * HIDDEN_SIZE + HIDDEN_SIZE

    return depth * (layer_norms + attention + mlp)
