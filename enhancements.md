# Enhancements and corrections

Corrections to the book's companion code and to earlier notes about it, plus enhancements that
follow from them. Newest first.

## Correction: the Chapter 5 deep momentum notebook trains on a mispaired Sharpe loss (K-2703)

**Status:** defect demonstrated in `tests/test_sharpe_loss_pairing.py`. Not yet fixed; the
shape-checked replacement loss is K-2709.

**What this corrects.** The removed `status-20260416.md` called `SharpeLoss` in `Utilis/loss.py`
"the single most portable idea in the book" and recommended taking "the exact MLP + SharpeLoss" from
`Chapter5/03_deep_momentum_strategy.ipynb` as a near-deployable starting point. That recommendation
is withdrawn. Do not reuse `SharpeLoss` as written, and do not treat the notebook's result as
evidence that the approach works.

**The defect.** The notebook's MLP ends in `nn.Linear(32, 1)`, so its positions have shape (B, 1).
The target returns have shape (B,). `SharpeLoss.forward` computes `outputs_prev * future_rets`,
which broadcasts to (B, B): every position is multiplied by every return in the batch, not by its
own. Nothing raises, and the loss still falls during training.

**Hand example.** Positions [1, −1, 0.5] and returns [2%, −1%, 3%]:

| Call | Product | Loss |
|---|---|---|
| Paired, both (B,) | [2%, 1%, 1.5%] | −3.000 |
| As in the notebook, (B, 1) × (B,) | all 9 cross products | −0.113 |

**Consequences.** Under the broadcast the loss depends only on the separate distributions of
positions and returns:

- It is unchanged when the returns are shuffled, so it carries no information about which position
  went with which return.
- It is minimised by a constant position. The size of the constant does not matter, and a position
  with perfect foresight of each return's sign scores worse.

The notebook's reported test result (annualised Sharpe 0.76 at 3.7% volatility on AAPL) therefore
carries no timing information. Retraining the notebook as written reproduces the 0.76 exactly, and
the trained model's test-period position is close to a constant: mean 0.118, range 0.089 to 0.140,
with a correlation of −0.03 to the next-day return. The result is a scaled-down buy-and-hold of
AAPL, whose own Sharpe over the same period is 0.79 at 31.7% volatility.

**Scope.** Only `Chapter5/03_deep_momentum_strategy.ipynb` imports `SharpeLoss`.
`Chapter6/02_deep_portfolio_optimization.ipynb` defines its own loss and multiplies (B, 20) weights
by (B, 20) returns, so it is not affected.

**Reproduce.**

```sh
uv run --locked pytest tests/test_sharpe_loss_pairing.py
```

The tests pass against the current code on purpose: they pin down the defective behaviour. When
K-2709 lands, the (B, 1) × (B,) call should raise instead, and these tests should be updated to
assert that.
