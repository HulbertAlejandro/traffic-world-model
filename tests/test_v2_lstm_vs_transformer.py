"""Statistics of the pre-registered LSTM vs Transformer comparison (docs/v2/ADDENDUM_LSTM_VS_TRANSFORMER.md).

scipy is not in the venv, so the Student t distribution is computed in the analysis script itself;
these tests pin it to table values and pin the pre-registered decision rule.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts" / "v2"))

import analyze_lstm_vs_transformer as an  # noqa: E402


@pytest.mark.parametrize("t, df, p", [(2.228139, 10, 0.05), (12.706205, 1, 0.05), (2.093024, 19, 0.05),
                                      (3.249836, 9, 0.01), (1.959964, 1e6, 0.05), (0.0, 5, 1.0)])
def test_two_sided_p_matches_t_tables(t, df, p):
    assert an.t_two_sided_p(t, df) == pytest.approx(p, abs=1e-5)


@pytest.mark.parametrize("df, q", [(10, 2.228139), (2, 4.302653), (17.3, 2.106)])
def test_t_quantile_matches_t_tables(df, q):
    assert an.t_quantile(0.975, df) == pytest.approx(q, abs=2e-3)


def test_welch_matches_a_hand_computation():
    x, y = np.array([7.13, 7.40, 7.22, 7.31]), np.array([7.40, 7.45, 7.42])
    w = an.welch(x, y)
    vx, vy = x.var(ddof=1) / 4, y.var(ddof=1) / 3
    assert w["d"] == pytest.approx(x.mean() - y.mean())
    assert w["t"] == pytest.approx(w["d"] / np.sqrt(vx + vy))
    assert w["df"] == pytest.approx((vx + vy) ** 2 / (vx ** 2 / 3 + vy ** 2 / 2))
    lo, hi = w["ci95"]
    assert (lo + hi) / 2 == pytest.approx(w["d"])
    assert (w["p"] < 0.05) == (lo > 0 or hi < 0)  # the CI and the test agree
    shifted = an.welch(x - 1.0, y)                  # far apart: significant, CI below 0
    assert shifted["p"] < 0.05 and shifted["ci95"][1] < 0


def test_decision_needs_both_intervals_on_the_same_side():
    assert an.decide({"ci95": [-0.3, -0.1], "bootstrap_ci95": [-0.25, -0.05]}) == "transformer"
    assert an.decide({"ci95": [0.1, 0.3], "bootstrap_ci95": [0.05, 0.2]}) == "lstm"
    assert an.decide({"ci95": [-0.3, 0.02], "bootstrap_ci95": [-0.25, -0.05]}) == "sin evidencia suficiente"
    assert an.decide({"ci95": [-0.3, -0.1], "bootstrap_ci95": [-0.25, 0.01]}) == "sin evidencia suficiente"


def test_all_runs_are_new_and_none_is_a_phase_2_run():
    for arch in ("lstm", "transformer"):
        for seed in an.SEEDS:
            d = an.run_dir(arch, seed)
            assert d.parent.name == "arch_comparison" and d.name == f"{arch}_raw_s{seed}"
    assert list(an.SEEDS) == list(range(10))


def test_pre_registered_sizes_have_the_pre_registered_parameter_counts():
    import training.v2_compression_experiment as exp
    from run_lstm_vs_transformer import ARCH_HPARAMS, EXPECTED_PARAMS
    assert ARCH_HPARAMS["lstm"] == {"hidden_dim": 137} and ARCH_HPARAMS["transformer"]["d_model"] == 80
    assert ARCH_HPARAMS["transformer"]["d_model"] % ARCH_HPARAMS["transformer"]["nhead"] == 0
    for arch, hp in ARCH_HPARAMS.items():
        model = exp.build_temporal_model({"architecture": arch, "latent_dim": exp.STATE_DIM,
                                          "action_dim": exp.ACTION_DIM, "sequence_length": 16, **hp})
        assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMS[arch]
    # Closed forms of the addendum (section 2), input 104 + 8 actions.
    h, d = 137, 80
    assert EXPECTED_PARAMS["lstm"] == 4 * h * h + 561 * h + 105
    assert EXPECTED_PARAMS["transformer"] == 8 * d * d + 1260 * d + 617
    # The training protocol is still Phase 2's, unchanged.
    assert exp.LSTM_PROTOCOL.hidden_dim == 128 and exp.TRANSFORMER_HPARAMS.d_model == 128
