import copy
import math

import pytest
import torch

import model as M


def cfg_for(base_cfg, pos):
    m = copy.deepcopy(base_cfg["model"]); m["pos_encoding"] = pos
    return m


def build(base_cfg, pos, seed=0):
    return M.build_model(cfg_for(base_cfg, pos), 168, 24, seed)


def shared(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith("pos_emb")}


def test_shared_backbone_is_bitwise_identical_across_arms(base_cfg):
    """Arm-specific parameter creation must not shift the shared weights."""
    a, b, c = (build(base_cfg, p, seed=3) for p in M.POS_TYPES)
    sa, sb, sc = shared(a), shared(b), shared(c)
    assert sa.keys() == sb.keys() == sc.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]) and torch.equal(sa[k], sc[k]), k


def test_shared_backbone_differs_across_seeds(base_cfg):
    a, b = build(base_cfg, "learned", 0), build(base_cfg, "learned", 1)
    assert not torch.equal(a.in_proj.weight, b.in_proj.weight)


def test_same_seed_gives_identical_model(base_cfg):
    assert M.weights_sha256(build(base_cfg, "rope", 5)) == M.weights_sha256(build(base_cfg, "rope", 5))


def test_learned_embedding_does_not_depend_on_backbone_rng(base_cfg):
    """The positional module has its own seed, so it is the same whatever the backbone consumed."""
    a = build(base_cfg, "learned", 2)
    b = build(base_cfg, "learned", 2)
    assert torch.equal(a.pos_emb.weight, b.pos_emb.weight)
    c = build(base_cfg, "learned", 3)
    assert not torch.equal(a.pos_emb.weight, c.pos_emb.weight)


def test_parameter_counts(base_cfg):
    a, b, c = (M.param_counts(build(base_cfg, p)) for p in M.POS_TYPES)
    assert a["positional"] == 168 * 64 == 10752
    assert b["positional"] == c["positional"] == 0
    assert a["total"] - b["total"] == 10752 and b["total"] == c["total"]


def test_forward_shape_and_finite(base_cfg):
    for p in M.POS_TYPES:
        out = build(base_cfg, p)(torch.randn(5, 168))
        assert out.shape == (5, 24) and torch.isfinite(out).all()


def test_sinusoidal_table_known_values():
    pe = M.sinusoidal_table(168, 64)
    assert torch.allclose(pe[0, 0::2], torch.zeros(32)) and torch.allclose(pe[0, 1::2], torch.ones(32))
    assert math.isclose(float(pe[1, 0]), math.sin(1.0), rel_tol=1e-6)
    assert math.isclose(float(pe[1, 1]), math.cos(1.0), rel_tol=1e-6)


def test_rope_preserves_norm_and_position_zero_is_identity():
    r = M.Rotary(16, 168, 10000)
    x = torch.randn(2, 4, 168, 16)
    y = r(x)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)
    assert torch.allclose(y[:, :, 0], x[:, :, 0], atol=1e-6)


def test_rope_scores_depend_only_on_relative_offset():
    r = M.Rotary(16, 168, 10000)
    q = torch.randn(16).expand(1, 1, 168, 16).contiguous()
    k = torch.randn(16).expand(1, 1, 168, 16).contiguous()
    s = (r(q)[0, 0] @ r(k)[0, 0].T)                      # s[m, n] = q_m . k_n after rotation
    for off in (-30, 0, 7, 24):
        diag = torch.diagonal(s, offset=off)
        assert torch.allclose(diag, diag[0].expand_as(diag), atol=1e-4), off


def test_rope_is_applied_to_q_and_k_only_in_every_layer(base_cfg):
    model = build(base_cfg, "rope")
    calls = []
    for blk in model.blocks:
        blk.attn.rotary.register_forward_hook(lambda mod, inp, out: calls.append(tuple(inp[0].shape)))
    model(torch.randn(2, 168))
    n_layers = base_cfg["model"]["n_layers"]
    assert len(calls) == 2 * n_layers                    # one call for q and one for k, per layer; v is never rotated
    assert not hasattr(model, "pos_emb") and not hasattr(model, "pe")   # nothing added to the input embedding


def test_additive_arms_have_no_rotary(base_cfg):
    for p in ("learned", "sinusoidal"):
        assert all(b.attn.rotary is None for b in build(base_cfg, p).blocks)


def test_positional_encoding_makes_the_model_order_sensitive(base_cfg):
    x = torch.randn(4, 168)
    xp = x.flip(1)
    for p in M.POS_TYPES:
        model = build(base_cfg, p).eval()
        assert not torch.allclose(model(x), model(xp), atol=1e-6), p


def test_shared_init_hash_matches_across_arms(base_cfg):
    hashes = {p: M.weights_sha256(build(base_cfg, p, seed=4), shared_only=True) for p in M.POS_TYPES}
    assert len(set(hashes.values())) == 1
