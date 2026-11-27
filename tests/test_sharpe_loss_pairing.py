"""Demonstrate the Sharpe-loss pairing defect in Chapter5/03_deep_momentum_strategy.ipynb (K-2703).

The notebook's MLP ends in ``nn.Linear(32, 1)``, so positions have shape (B, 1), while the target
returns have shape (B,). ``Utilis.loss.SharpeLoss`` multiplies the two, which broadcasts to (B, B):
every position is paired with every return instead of with its own.

These tests pin down the behaviour of ``SharpeLoss`` as it stands. They pass against the defective
call, on purpose: they document it. The shape-checked replacement is tracked in K-2709.
"""

import pytest
import torch

from Utilis.loss import SharpeLoss

POSITIONS = [1.0, -1.0, 0.5]
RETURNS = [0.02, -0.01, 0.03]


@pytest.fixture
def positions() -> torch.Tensor:
    return torch.tensor(POSITIONS)


@pytest.fixture
def returns() -> torch.Tensor:
    return torch.tensor(RETURNS)


def notebook_loss(positions: torch.Tensor, returns: torch.Tensor) -> float:
    """Call the loss with the shapes the notebook uses: (B, 1) positions, (B,) returns."""
    return float(SharpeLoss()(positions.unsqueeze(1), returns))


def test_paired_loss_is_minus_three(positions: torch.Tensor, returns: torch.Tensor) -> None:
    # Strategy returns [2%, 1%, 1.5%]: mean 1.5%, sample std 0.5%, so the loss is -3.
    strategy_returns = positions * returns
    assert torch.allclose(strategy_returns, torch.tensor([0.02, 0.01, 0.015]))
    assert float(SharpeLoss()(positions, returns)) == pytest.approx(-3.000, abs=5e-4)


def test_notebook_shapes_broadcast_to_all_pairs(
    positions: torch.Tensor, returns: torch.Tensor
) -> None:
    all_pairs = positions.unsqueeze(1) * returns
    assert all_pairs.shape == (3, 3)
    assert notebook_loss(positions, returns) == pytest.approx(-0.113, abs=5e-4)


def test_broadcast_loss_ignores_which_return_goes_with_which_position(
    positions: torch.Tensor, returns: torch.Tensor
) -> None:
    shuffled = returns[torch.tensor([2, 0, 1])]

    # Correctly paired, a shuffle changes the strategy and so the loss.
    paired = float(SharpeLoss()(positions, returns))
    assert float(SharpeLoss()(positions, shuffled)) != pytest.approx(paired, abs=1e-2)

    # Under the broadcast it changes nothing: the loss sees no timing information.
    assert notebook_loss(positions, shuffled) == pytest.approx(
        notebook_loss(positions, returns), abs=1e-6
    )


def test_broadcast_loss_is_minimised_by_a_constant_position() -> None:
    generator = torch.Generator().manual_seed(0)
    batch = 128
    returns = 0.001 + 0.02 * torch.randn(batch, generator=generator)
    assert float(returns.mean()) > 0

    constant = notebook_loss(torch.full((batch,), 0.3), returns)

    # The size of the constant is irrelevant.
    assert notebook_loss(torch.full((batch,), 0.9), returns) == pytest.approx(constant, abs=1e-5)

    # No position vector inside Softsign's (-1, 1) range does better, including one with
    # perfect foresight of the sign of each return.
    candidates = [torch.sign(returns)]
    candidates += [2 * torch.rand(batch, generator=generator) - 1 for _ in range(200)]
    for candidate in candidates:
        assert notebook_loss(candidate, returns) >= constant - 1e-6
