"""Fast structural checks for the H3 coarse-to-fine latent extension."""
from __future__ import annotations

import torch

from methods.senseiver.network import Senseiver


def build(mode, grid=(8, 4)):
    return Senseiver(
        im_ch=4, grid=grid, space_bands=16, enc_preproc_ch=32, num_latents=64,
        enc_num_latent_channels=32, num_layers=3, num_cross_attention_heads=2,
        enc_num_self_attention_heads=2, num_self_attention_layers_per_block=3,
        dec_preproc_ch=32, dec_num_latent_channels=32,
        dec_num_cross_attention_heads=1, latent_mode=mode, readout="direct",
        time_window=16, time_dim=8, time_scalar=True,
        share_encoder_blocks=False)


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    flat, hier = build("grid"), build("hierarchical")
    assert flat.num_params == hier.num_params == 82_596
    assert hier.encoder.scale_shapes == ((2, 1), (4, 2), (8, 4))

    b, ns, p = 2, 12, hier.pos_enc.shape[1]
    tok = torch.randn(b, ns, 4 + p)
    pad = torch.zeros(b, ns, dtype=torch.bool)
    dt = torch.randint(0, 16, (b, ns))
    out = hier.reconstruct(tok, pad, dt)
    assert out.shape == (b, 4, 8, 4)

    perm = torch.randperm(ns)
    with torch.no_grad():
        delta = float((out - hier.reconstruct(tok[:, perm], pad[:, perm],
                                              dt[:, perm])).abs().max())
    assert delta < 1e-4, delta

    out.square().mean().backward()
    missing = [n for n, p in hier.named_parameters() if p.grad is None]
    assert not missing, missing

    try:
        build("hierarchical", grid=(7, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("non-divisible hierarchical grid was accepted")

    print(f"ALL PASS  params={hier.num_params:,}  shape={tuple(out.shape)}  "
          f"sensor-permutation max|delta|={delta:.2e}")


if __name__ == "__main__":
    main()
