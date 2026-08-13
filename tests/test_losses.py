from __future__ import annotations

import torch

from cytobridge.losses import (
    CytoBridgeLoss,
    CytoBridgeLossConfig,
    InfoNCEConfig,
    info_nce_with_hard_negatives,
    zinb_nll,
    logfc_mse,
    normalized_logfc_mse,
)


def test_zinb_nll_is_finite_for_extreme_inputs():
    x = torch.randint(0, 100, (8, 32)).float()
    mu = torch.exp(torch.randn(8, 32).clamp(-4, 8))
    theta = torch.nn.functional.softplus(torch.randn(8, 32)) + 1e-4
    pi = torch.randn(8, 32) * 6

    loss = zinb_nll(x, mu, theta, pi)

    assert loss.shape == (8,)
    assert torch.isfinite(loss).all()


def test_zinb_nonzero_case_matches_negative_binomial_component():
    """Regression test: nonzero ZINB term must use log(1-sigmoid(pi)) = -softplus(+pi).
    A previous bug used softplus(-pi) and silently passed when pi=0 because
    softplus is symmetric there. Use non-zero pi here so the sign matters.
    """
    x = torch.tensor([[3.0, 5.0]])
    mu = torch.tensor([[2.0, 4.0]])
    theta = torch.tensor([[1.5, 2.5]])
    pi = torch.tensor([[1.2, -0.7]])
    nb_log_prob = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + x * (torch.log(mu) - torch.log(theta + mu))
    )
    # log(1 - sigmoid(pi)) = -softplus(+pi)
    expected = -(nb_log_prob - torch.nn.functional.softplus(pi)).sum(dim=-1)

    observed = zinb_nll(x, mu, theta, pi)

    assert torch.allclose(observed, expected, atol=1e-5)


def test_info_nce_accepts_hard_negatives():
    z = torch.randn(4, 16)
    z_pos = z + 0.01 * torch.randn_like(z)
    z_hard = torch.randn(4, 3, 16)

    loss = info_nce_with_hard_negatives(z, z_pos, z_hard)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_hard_negative_weight_increases_loss_when_negative_is_close():
    z = torch.eye(3)
    z_pos = z.clone()
    z_hard = z[:, None, :].clone()

    base = info_nce_with_hard_negatives(z, z_pos, z_hard)
    heavier = info_nce_with_hard_negatives(
        z, z_pos, z_hard, cfg=InfoNCEConfig(weight_hard=4.0)
    )

    assert heavier > base


def test_combined_loss_backpropagates():
    outputs = {
        "mu": torch.exp(torch.randn(4, 20, requires_grad=True).clamp(-3, 3)),
        "theta": torch.nn.functional.softplus(torch.randn(4, 20, requires_grad=True))
        + 1e-4,
        "pi": torch.randn(4, 20, requires_grad=True),
        "z": torch.randn(4, 8, requires_grad=True),
        "z_pos": torch.randn(4, 8, requires_grad=True),
        "z_hard_neg": torch.randn(4, 2, 8, requires_grad=True),
        "pathway_pred": torch.randn(4, 5, requires_grad=True),
        "pathway_attn": torch.softmax(torch.randn(4, 5, requires_grad=True), dim=-1),
    }
    batch = {
        "treated_counts": torch.randint(0, 20, (4, 20)).float(),
        "pathway_gsea": torch.rand(4, 5),
    }
    loss_dict = CytoBridgeLoss(CytoBridgeLossConfig()).forward(outputs, batch)

    loss_dict["loss"].backward()

    assert torch.isfinite(loss_dict["loss"])
    assert set(loss_dict) == {
        "loss",
        "L_recon",
        "L_contrast",
        "L_pathway",
        "L_kl",
        "L_delta",
        "L_direction",
        "L_drugspec",
        "L_logfc",
        "L_norm_recon",
    }


def test_logfc_mse_zero_when_prediction_matches_treated():
    control = torch.rand(3, 12) * 5.0
    treated = torch.rand(3, 12) * 5.0

    assert torch.allclose(
        logfc_mse(treated, treated, control), torch.zeros(()), atol=1e-6
    )


def test_zero_reconstruction_weight_skips_zinb_and_uses_truth_control():
    batch_size, n_genes, latent, pathways = 3, 20, 8, 5
    outputs = {
        "mu": torch.full((batch_size, n_genes), 1.0e30),
        "theta": torch.ones(batch_size, n_genes),
        "pi": torch.zeros(batch_size, n_genes),
        "z": torch.randn(batch_size, latent),
        "z_pos": torch.randn(batch_size, latent),
        "z_hard_neg": None,
        "pathway_pred": torch.randn(batch_size, pathways),
        "pathway_attn": torch.softmax(torch.randn(batch_size, pathways), dim=-1),
    }
    batch = {
        "treated_counts": torch.rand(batch_size, n_genes) * 5.0,
        "pathway_gsea": torch.rand(batch_size, pathways),
        "input_control_counts": torch.zeros(batch_size, n_genes),
        "truth_control_counts": torch.rand(batch_size, n_genes) * 5.0,
        "cell_lines": ["A", "A", "B"],
    }
    loss_fn = CytoBridgeLoss(
        CytoBridgeLossConfig(
            lam_recon=0.0,
            lam_logfc=0.5,
            lam_contrast=0.0,
            lam_pathway=0.0,
            lam_kl=0.0,
        )
    )

    result = loss_fn(outputs, batch)

    assert result["L_recon"].item() == 0.0
    assert result["L_logfc"].item() > 0.0
    assert torch.isfinite(result["loss"])


def test_normalized_reconstruction_uses_frozen_gene_scale():
    control = torch.ones(2, 3)
    treated = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    prediction = treated.clone()
    scale = torch.tensor([0.5, 1.0, 2.0])

    matched = normalized_logfc_mse(prediction, treated, control, scale)
    shifted = normalized_logfc_mse(prediction + 1.0, treated, control, scale)

    assert matched.item() == 0.0
    assert shifted > matched
