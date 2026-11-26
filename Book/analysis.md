# Analysis: Deep Learning in Quantitative Trading

**Authors**: Zihao Zhang (University of Oxford) and Stefan Zohren (University of Oxford, Oxford-Man Institute)
**Publication**: Cambridge Elements in Quantitative Finance (series editor Riccardo Rebonato), Cambridge University Press, first published online September 2025; DOI 10.1017/9781009707091; ISBN 978-1-009-70712-1 (hardback), 978-1-009-70711-4 (paperback); 184 pages

---

## 1. Summary

The Element sets out to be a self-contained bridge between classical quantitative trading and modern deep learning. Its stated motivation is that linear time-series models (ARIMA, GARCH) and hand-crafted features impose assumptions (linearity, stationarity, normality) that financial data violate, whereas multi-layer networks can learn hierarchical, non-linear representations directly from raw market inputs, handle unstructured data and adapt across regimes. The authors are careful, however, to frame deep learning as a complement to, not a replacement for, established strategy design: every application in Part II starts from a classical strategy (volatility targeting, time-series momentum, MACD trend signals, cross-sectional momentum, mean-variance optimisation, limit-order-book imbalance) and asks how a network can improve one component of it.

Part I covers the statistics of financial returns (heavy tails, volatility clustering, serial correlation, stationarity testing, AR/ARMA/GARCH), supervised learning (regression and classification losses, evaluation metrics), and a tour of architectures from fully connected and convolutional networks through WaveNet, LSTMs, attention and transformers to graph neural networks, large language models, state-space models and xLSTM. Section 4 presents a six-stage training workflow (problem set-up, data collection and cleaning, feature extraction, model construction, cross-validation and hyper-parameter tuning, pipeline deployment) with a strong emphasis on time-series-aware validation: samples must never be shuffled across time because overlapping targets (for example a 300-second forward return sampled every second) leak information between folds, so a rolling, chronological hold-out scheme is prescribed. Hyper-parameter search is discussed in terms of grid, random and Bayesian (Gaussian-process, Optuna) methods, and Ray, Dask and Spark are compared as pipeline back-ends.

Part II contains the substantive contributions. Section 5 shows how a network can replace the two-step "predict, then size" procedure of a CTA-style strategy by directly outputting a position and being trained on a differentiable Sharpe-ratio loss (Deep Momentum Networks, Lim, Zohren and Roberts 2019), how attention-based architectures extend this (the Momentum Transformer, Wood et al. 2021–2023), and how learning-to-rank algorithms borrowed from information retrieval improve the ranking step in cross-sectional momentum (Poh et al. 2021–2022). Section 6 covers volatility forecasting (HAR, HEAVY, HARNet, DeepVol) and an end-to-end portfolio-optimisation framework in which a "score block" network feeds a differentiable "portfolio block" that enforces long-only, maximum-position, cardinality or leverage constraints, trained directly on Sharpe, minimum-variance or mean-variance objectives (Zhang, Zohren and Roberts 2020; Zhang et al. 2021). Section 7 introduces exchange mechanics and Level 2/Level 3 data, then reviews DeepLOB (CNN plus LSTM on order-book snapshots), multi-horizon attention extensions, market-by-order models, deep reinforcement learning for execution (Nagy, Calliess and Zohren 2023; Schnaubelt 2022) and generative models for order flow. Section 8 concludes with open problems: alternative data, explainability (SHAP, Integrated Gradients, LIME) and the systemic-risk implications of algorithmic trading.

For a systematic firm the Element's value is twofold. It is a compact, authoritative statement of the Oxford-Man research programme with all the loss functions and constraint parameterisations written out, and it provides a code base that makes the headline results reproducible in principle. Its weakness, discussed in Section 4 below, is that the empirical evidence is reproduced from the source papers without new out-of-sample extension, and transaction costs are largely set aside.

## 2. Trading Strategies

The Element presents several explicit, testable strategies. The parameterisations below are taken from the text.

### Strategy A: Deep Momentum Network (DMN) – end-to-end time-series momentum on futures

- **Signal generation**: A network f(u; θ) maps a feature vector u (volatility-normalised returns over 1-day, 1-month, 3-month, 6-month and 12-month windows and three MACD indicators M(8,24), M(16,48), M(32,96)) directly to a position w ∈ [−1, 1] per instrument per day (tanh output). No separate forecast step.
- **Training objective**: negative annualised Sharpe ratio of the volatility-scaled strategy return R(i,t) = w_t · (σ_tgt / σ_t) · r_{t,t+1}, computed across all instruments and dates in the training window (Equation 81); an average-return loss is also tested. The loss is differentiable, so standard gradient descent applies. The PyTorch loss is five lines (negative mean divided by standard deviation of the position-times-return series).
- **Entry/exit rules**: positions are refreshed daily; there are no discrete entries or exits.
- **Position sizing**: the network output is the sizing. Each instrument's position is scaled by σ_tgt/σ_t inside the loss (Equation 81); Table 5 additionally rescales the overall strategy return to the 15% annual target.
- **Universe**: 88 ratio-adjusted continuous futures from the Pinnacle Data Corp CLC database, 1990–2015. The Element lists commodities, fixed income and currencies; the source paper's appendix also includes equity-index futures (roughly 50 commodity, 13 equity, 12 fixed-income and 13 FX contracts).
- **Holding period**: effectively one day, though the network learns persistence.
- **Expected characteristics** (Table 5, volatility-rescaled, 1995–2015, gross of costs): Sharpe-optimised LSTM 2.907 Sharpe, 0.209 maximum drawdown, 59.6% positive days, versus 1.392 for sign-of-12-month-return TSMOM, 1.111 for MACD and 0.759 long-only. The Sharpe-optimised MLP achieves 2.017 and the linear model 1.496. Average-return-optimised variants are materially worse (LSTM 1.349).

### Strategy B: Momentum Transformer – attention-based DMN

- **Signal generation**: same inputs and loss as Strategy A, but the network is a Temporal Fusion Transformer variant (LSTM encoder for local pattern summarisation feeding multi-head attention), compared against plain, decoder-only and convolutional transformers and Informer.
- **Universe and period**: "the portfolio of 88 futures" according to the Element (the source paper, Wood et al., uses 50 contracts; see the discrepancies in Section 4), tested 1995–2020, with sub-periods 2015–2020 (when classical and LSTM strategies underperformed) and the COVID-19 crash and rebound.
- **Expected characteristics** (Table 6, raw outputs, gross): decoder-only TFT Sharpe 2.54 over 1995–2020 versus LSTM 1.70 and TSMOM 1.03; 1.71 versus 0.82 and 0.24 over 2015–2020; during COVID-19 the LSTM lost money (Sharpe −1.50) while the vanilla transformer achieved 3.38 and the decoder-only transformer 3.01, with the TFT lagging at 1.22. Variable-importance tables show the one-day return receiving the largest attention weight (30.8% in 2015–2020), with quarterly returns gaining weight during COVID-19.

### Strategy C: Learning-to-Rank cross-sectional momentum (equities)

- **Signal generation**: each monthly rebalance is treated as an information-retrieval "query" and each stock as a "document"; the label is the stock's realised next-month return decile. Pairwise and listwise LTR algorithms (RankNet, LambdaMART, ListNet, ListMLE) are trained on the same momentum feature set to produce a score; stocks are ranked, the top and bottom deciles selected.
- **Entry/exit rules**: rebalance on the last trading day of each month; a fixed 100 stocks long and 100 short at all times (about 10% of tradeable names; the decile language is an approximation).
- **Position sizing**: each selected stock's signed position is scaled by its own ex-ante volatility, σ_tgt/σ_i with a 15% target (Equation 82; 63-day exponentially weighted span in the source paper), then averaged across names. Table 8 applies a second, portfolio-level rescaling to 15%. This is not equal weighting within the sleeve.
- **Universe**: NYSE-listed stocks, 1980–2019.
- **Where the edge sits (source paper, Exhibits 3–4)**: in long-only decile portfolios rescaled to target volatility, LambdaMART's bottom decile has a Sharpe of 0.075 against 0.582 for the MACD rule and 0.675 for random selection; its top decile has 1.232 against 1.130 for MACD. Most of the ranker's advantage therefore comes from identifying losers, i.e. from the short book, which is where borrow cost and availability bite. Ranking quality differs far less than Sharpe: Kendall's tau is 0.032 for LambdaMART against 0.013 for MACD.
- **Source-paper detail (Poh et al. 2021, not in the Element)**: CRSP share codes 10 and 11, price above $1 at rebalance and actively traded over the prior year. Features are raw and volatility-normalised returns over 3, 6 and 12 months plus MACD-based indicators (the three MACD signals at the rebalance date and at 1, 3, 6 and 12 months earlier), not the DMN feature set. The paper states 16 MACD features but enumerates 15 and gives no overall total; 22 predictors (with the composite MACD signal as the sixteenth) is a plausible reconstruction, not a documented count. Decile labels are formed from returns 21 days ahead; volatility uses a 63-day span; models are re-tuned every 5 years. LambdaMART is implemented in XGBoost (`rank:pairwise`).
- **Expected characteristics** (Table 8, gross of costs): LambdaMART Sharpe 2.156, Calmar 1.555, 76.2% positive months; ListNet 1.970; RankNet 1.502; versus MACD-scored benchmark 0.696, 12-month-return benchmark 0.551, regress-then-rank MLP 0.265 and random selection 0.155.

### Strategy D: End-to-end deep portfolio optimisation (equities)

- **Signal generation**: a score network (linear, MLP, LSTM or CNN) maps past returns and momentum features of all assets to fitness scores; a differentiable portfolio block maps scores to weights satisfying a chosen constraint set: signed softmax for long–short with unit gross exposure (Equation 104), softmax for long-only, generalised sigmoid for a maximum position u (Equation 108), NeuralSort relaxation for cardinality (Equations 109–112), scaling for leverage L.
- **Training objective**: directly the Sharpe ratio, global minimum variance or mean–variance with risk aversion λ of the realised portfolio return.
- **Universe and period**: 735 Russell 3000 stocks with data from 3 January 1984 to 6 July 2021; test period 2001–2021; daily.
- **Expected characteristics** (Table 9, gross): E2E-LSTM maximum-Sharpe portfolio 2.604 Sharpe with 0.017 maximum drawdown and beta 0.017, versus two-step MPT-LSTM 0.858, S&P 500 0.402, equal weight 0.682, maximum diversification 1.641 and GMVP 0.992. Under constraints: 5% maximum-position variant 2.683; leverage-5 variant 2.540; long-only 1.691 (beta 0.767); cardinality (top/bottom decile) only 0.588.
- **Other objectives** (Table 9): the result is strongly objective-dependent. The minimum-variance objective gives a Sharpe of 0.047, and mean–variance gives 0.349, 1.055 and 1.394 for λ = 10, 20 and 30.
- **Return profile and source-paper detail**: the E2E-LSTM earns 6.0% a year at 2.3% volatility, so the 2.6 Sharpe is an unlevered, low-volatility result. The equity experiment comes from Chao Zhang, Zihao Zhang, Cucuringu and Zohren (2021), which uses WRDS data, trains on 1984–1999 with 2000 as validation and rolls forward one year at a time, and reports the E2E-LSTM Sharpe falling to 1.693 at 2 basis points of cost.

### Strategy E: Deep learning on limit-order-book data (DeepLOB family) and RL execution

- **Signal generation**: CNN filters over a window of Level 2 snapshots (prices and volumes at several depths) followed by LSTM units predict the direction of the mid-price over a horizon k; multi-horizon Seq2Seq/attention extensions produce a forecast path; market-by-order (Level 3) models produce comparably accurate but less correlated signals.
- **Execution**: a deep double Q-network agent trained with experience replay in a full order-book simulator learns passive order placement and inventory management, outperforming a heuristic that trades the same signal aggressively (Nagy, Calliess and Zohren 2023, AAPL example). Schnaubelt (2022) compares submit-and-leave, backward-induction Q-learning, DDQN and PPO for cryptocurrency execution, with the DRL agents front-loading execution.
- **Expected characteristics**: the Element reports no Sharpe ratios for these; results are classification accuracy and qualitative execution comparisons, so they are research directions rather than specified strategies.

Additionally, the generative order-flow model of Nagy, Frey, Sapora et al. (2023) is shown to produce returns whose correlation with realised returns over the next 100 messages is around 0.1 (Google) and 0.2 (Intel). Intel's correlation is significant at the 5% level out to 100 messages; Google's p-values sit at or near the threshold for about 80 messages. The authors describe this as a potential source of alpha.

## 3. Data Sources

| Dataset | Description | Source | Time Period | Frequency |
|---------|-------------|--------|-------------|-----------|
| CLC continuous futures | 88 ratio-adjusted continuous futures across commodities, fixed income, FX | Pinnacle Data Corp | 1990–2015 (DMN); to 2020 (Momentum Transformer) | Daily |
| NYSE equities | All tradable NYSE stocks; top/bottom 10% (100 each side) traded | Not specified in the Element (CRSP in the source paper) | 1980–2019 | Daily prices, monthly rebalance |
| Russell 3000 subset | 735 stocks with continuous history | Not specified in the Element (WRDS in the source paper) | 1984-01-03 to 2021-07-06 (test 2001–2021) | Daily |
| LOB data | Level 2 snapshots and Level 3 market-by-order messages for large-cap US and LSE stocks | Exchange feeds (LOBSTER / LSE in the source papers) | Varies by paper | Event / sub-second |
| Crypto LOB | Order books for execution experiments | Exchange APIs (Schnaubelt 2022) | 2019–2021 in source | Event |

**Data preprocessing notes**:
- Futures are stitched into continuous series by ratio adjustment at roll dates; the Element explains why "nearest futures" concatenation produces spurious jumps and must not be used for backtesting.
- All strategy returns are volatility-normalised using an exponentially weighted standard deviation and scaled to a 15% annual target; the authors note that in practice one targets dollar volatility.
- Features are normalised by trailing measures (for example log(V_{t+1}/median_20(V))) to remove non-stationarity; missing-value imputation must be strictly backward-looking to avoid look-ahead bias.
- Cross-validation is chronological and rolling; no shuffling. The authors explicitly warn about leakage from overlapping forward returns.
- Survivorship bias is not discussed for the equity universes; the Russell 3000 experiment uses 735 stocks with continuous histories, which is a survivorship-prone selection.

## 4. Critical Evaluation

### Methodological Strengths
- Every strategy is specified down to the loss function and the weight-mapping transformation, which is unusual for a monograph and makes faithful replication possible.
- The end-to-end framing (optimising the trading objective rather than a forecasting proxy) is theoretically well motivated: forecast error is not monotonically related to strategy utility, and the signed-softmax and NeuralSort constructions are elegant ways of keeping constraints inside the differentiable graph.
- Time-series-aware validation, rolling retraining and the treatment of futures rolls are handled correctly and explained clearly.
- The Momentum Transformer analysis reports sub-period performance including a known bad period for trend following (2015–2020) and a regime shock (COVID-19), rather than only full-sample averages.
- Interpretability is taken seriously: attention weights and variable-importance tables are reported.

### Potential Concerns

#### Data Snooping / Overfitting
- The reported Sharpe ratios (2.9 for the LSTM DMN, 2.6 for the E2E-LSTM, 2.2 for LambdaMART) are very high for daily-rebalanced, gross-of-cost strategies and should be treated as upper bounds. The numbers come from the original papers; the Element does not extend any of them with a fresh out-of-sample period, so there is no independent confirmation that performance persisted after publication (the DMN paper appeared in 2019, the LTR papers in 2021).
- Each comparison involves a family of architectures and loss functions (linear, MLP, WaveNet, LSTM, several transformers, four LTR algorithms, four constraint sets); only the best variant is emphasised. No deflated Sharpe ratio or other multiple-testing correction is reported.
- Rolling retraining with hyper-parameter search every few years multiplies the effective number of trials further.

#### Robustness
- The futures results depend on a single data vendor's continuous contracts; no robustness to roll methodology, universe composition or alternative volatility estimators is shown.
- The equity E2E results use 735 Russell 3000 stocks over 1984–2021. Neither the Element nor the source paper documents how they were selected. A fixed cross-section over 37 years makes a complete-history filter, and hence survivorship bias, very likely, but it is an inference; record it as a serious unresolved risk.
- Sensitivity to the 15% volatility target, the look-back lengths, the number of selected stocks (100) and the retraining frequency is not reported.
- The COVID-19 sub-period in Table 6 is short (1 January to 15 October 2020 in the source paper), so Sharpe ratios above 3 and Calmar ratios of 7–9 over that window carry very wide confidence intervals.

#### Practical Implementation
- Transaction costs are explicitly excluded in the LTR equity results and are not modelled in the DMN or E2E tables. Daily-rebalanced network positions can turn over far more than a monthly TSMOM rule; the DMN source paper found that LSTM performance degraded sharply once costs above a few basis points were charged, a finding not reproduced in the Element. This is the single most important caveat for a practitioner.
- Capacity: the LTR strategy trades 100 NYSE names each side monthly, but its results are gross and the paper reports no turnover, so monthly rebalancing alone establishes neither low cost nor capacity; the short book in stocks priced just above $1 is the main constraint; the DeepLOB and RL execution results rely on queue position and sub-second reaction and are capacity-constrained and latency-sensitive.
- The E2E portfolio with beta near zero and 1.7% maximum drawdown over twenty years is implausible net of costs and borrowing constraints; shorting costs and availability are ignored. The source paper's own 2 basis point test cuts its Sharpe from 2.604 to 1.693.
- Data availability: Pinnacle CLC and full-depth LOB data are commercial; the code repository provides sample data only.

#### Net-of-cost evidence in the source papers (not reproduced in the Element)

| Source | Result | Zero cost | With costs |
|---|---|---|---|
| Wood et al. (2021) | LSTM, 2015–2020 | 0.82 | 0.20 at 1 bp; −0.43 at 2 bp; −1.05 at 3 bp |
| Wood et al. (2021) | Decoder-only TFT, 2015–2020 | 1.71 | 1.01 at 1 bp; 0.32 at 2 bp; −0.37 at 3 bp |
| Wood et al. (2021) | Decoder-only TFT, 25 commodity futures only, 2015–2020 | n/a | 1.23 at 3 bp |
| Zhang et al. (2021) | E2E-LSTM, 2001–2021 | 2.604 (6.0% return) | 1.693 at 2 bp (3.9% return, 2.3% volatility) |
| Zhang et al. (2021) | Other E2E variants at 2 bp | 0.931 to 1.688 | linear −0.090; MLP −0.295; CNN −0.217; cardinality-constrained LSTM −0.529 |
| Lim et al. (2019) | Sharpe-optimised LSTM, 1995–2015 | 2.907 (rescaled) | beats benchmarks only up to 2–3 bp; −5.313 at 10 bp; 0.912 at 10 bp with the turnover penalty |

Implications. The unregularised daily DMN loses its edge within 1–3 basis points. The E2E equity result is fragile and architecture-dependent, not eliminated: the LSTM keeps a 1.693 Sharpe at 2 basis points while every other end-to-end architecture turns negative, and a linear extrapolation of the LSTM's return loss puts its break-even near 5–6 basis points (own estimate). The stronger reasons to defer E2E are the 735-stock daily universe, the untreated borrow costs and the unresolved survivorship question. The turnover-regularised DMN at 10 basis points beats simple trend by only 0.051 Sharpe (0.912 against 0.861), and no paper reports a regulariser tuned to 1–3 basis points, so its incremental value at realistic costs is untested. The LTR results are gross throughout.

#### Discrepancies between the Element and its source papers
- **Momentum Transformer universe**: the Element says Table 6 covers "the portfolio of 88 futures"; Wood et al. use 50 liquid Pinnacle CLC contracts (25 commodities, 11 equities, 5 fixed income, 9 FX). Replicate with the paper's 50.
- **Table 6, Transformer row, 1995–2020**: the Element prints a Sharpe of 1.14; the paper gives 1.41 (Sortino 2.13 and Calmar 1.53 agree). The Element's figure is a transposition.
- **Changepoint-detection variant**: the paper's best model is the Decoder-Only TFT with CPD features (Sharpe 2.62 for 1995–2020, 2.00 for 2015–2020, 2.47 for COVID-19). The Element omits this row. Each experiment in the paper is averaged over five runs.
- **Table 7**: the label M(32, 98) is carried over from the paper's exhibit; the feature is MACD(32, 96).

#### Other Limitations
- Volatility forecasting (Section 6.3) is reviewed qualitatively; no forecast-loss tables are reproduced, so the claimed gains of HARNet and DeepVol cannot be assessed from the Element alone.
- The generative-LOB result (correlation of 0.1–0.2 between generated and realised returns) is interesting but is a correlation over 100 future messages, not a trading result.
- As a monograph that reproduces the group's own papers, it does not engage with independent replications or critiques of DMNs and LTR momentum in the literature. The two main LOB benchmark studies (Briola, Bartolucci and Aste 2024; Prata et al. 2024) are cited in one sentence each, without discussion of their findings.

## 5. Suggested Improvements

### Strategy Enhancements
- Add an explicit turnover penalty to the Sharpe loss (cost-aware training, as in the original DMN paper's extension) so that the network learns to trade less; compare Sharpe net of 1, 3 and 5 basis points per unit turnover.
- Ensemble several DMN seeds and architectures (LSTM, decoder-only TFT, Informer) with equal or inverse-variance weights; the COVID-19 results show that different architectures fail in different regimes, so a committee should be more stable than any single model.
- For the LTR strategy, add fundamental and quality characteristics to the momentum-only feature set and test listwise objectives that optimise the top-k return directly.
- For the E2E portfolio, add a transaction-cost term to the objective and constrain daily weight changes; test the framework on futures (where the near-diagonal covariance makes the problem easier) as well as equities.

### Additional Robustness Checks
- Re-run the DMN and Momentum Transformer on later data, labelling the periods separately: post-sample (after the authors' data end) and post-preprint (after publication). For the DMN these are 2016 onwards and May 2019 onwards; Wood et al. already report the LSTM over 2015–2020, so only 2021 onwards is new evidence. For the Momentum Transformer, post-sample starts in 2021 and the revised specification dates from November 2022. Include the 2022 trend-following year and the 2025 tariff-shock episode.
- Add a volatility-scaled long-only benchmark matched for risk (Kim, Tse and Wald 2016), since part of the apparent TSMOM and DMN advantage comes from volatility scaling itself.
- Compute probabilistic and deflated Sharpe ratios across the full set of architectures tried; report the distribution of performance across random seeds.
- Replace the survivorship-biased Russell 3000 subset with a point-in-time universe including delistings.
- Vary the volatility target, look-back windows and retraining frequency; test the strategies with alternative roll conventions and with a second futures data source.
- Block-bootstrap the return series to produce confidence intervals on Sharpe and drawdown statistics, especially for the short COVID-19 window.

### Implementation Considerations
- Daily network positions should be passed through a size-discretisation or no-trade band to suppress jitter before execution.
- Model retraining and hyper-parameter search should be scheduled off-line with a strictly expanding information set; a champion/challenger process is advisable before any model replaces a live one.
- For any LOB-based model, replace a blanket co-location assumption with a measured test: compare the signal's decay against feed-to-order latency and queue-adjusted fills on the intended venue. The Element's RL results should not be assumed to transfer to a slower set-up until that test is passed. The same criterion applies to third-party LOB signals.
- Volatility forecasting (Section 6.3) is the part of the Element closest to the VRP book and belongs in that workstream, not this plan. Start with HAR-RV and EWMA baselines scored by QLIKE; DeepVol targets day-ahead volatility from intraday data, so state how that horizon relates to the option holding period before testing any sizing or entry rule. The Element itself reports little gain from deep models over HAR on daily inputs.

## 6. Replication Research Plan

**Priority order (revised 2026-11-26).**
1. **Strategy C (LTR) first, as the basis for a new cross-sectional equity sleeve.** Korkea has no such sleeve today, so the question is absolute as well as relative: does a learned ranker produce a tradable net return, and does it add anything over the simple published rules? The experiment is specified in Section 6.8. The 1980–2019 sample cannot be reproduced on current data (see 6.1), so this is a contemporary test, not a fidelity replication.
2. **Strategy A (DMN) second, with the turnover penalty in the loss from the start.** Gate: net Sharpe above volatility-scaled TSMOM at 2 basis points, with contract-specific commissions, spreads, rolls and integer sizing, reported by asset class and judged on 2021 onwards; stop if it fails.
3. **Unregularised DMN, Momentum Transformer and E2E equity portfolio: reference only.** The source papers' own cost tests (Section 4) make them unattractive at realistic execution costs.

Strategies A and C both map naturally onto QuantRocket (Moonshot for the backtest, with a PyTorch or LightGBM model trained outside and loaded as a prediction file) or QuantConnect LEAN (Python algorithm with a pre-trained model in the object store).

### 6.1 Data Requirements

| Variable | Description | Suggested Source | Format |
|----------|-------------|------------------|--------|
| Continuous futures prices | 50–90 liquid futures, ratio-adjusted at roll | Pinnacle CLC (purchase) for 1990 onwards; QuantConnect continuous futures (`BackwardsRatio`) for May 2009 onwards, US contracts only; QuantRocket only with imported third-party history | Daily OHLCV |
| Equity prices and returns | Point-in-time US equity universe with delistings | QuantRocket Sharadar (1998 onwards) or usstock bundle (2007 onwards); QuantConnect US Equities. The paper's CRSP sample starts in 1980 and cannot be reproduced | Daily adjusted close, plus unadjusted close for the price screen |
| Short availability and borrow fees | Timestamped shortable shares and borrow fees | QuantRocket IBKR archive (April 2018 onwards) | Daily |
| Risk-free rate | For excess returns in Sharpe calculations | FRED (3-month T-bill) | Daily |
| Volatility estimates | EWMA standard deviation, 60-day span | Derived | Daily |

**Data acquisition steps**:
1. Futures history is the binding constraint. Interactive Brokers supplies history only for futures that expired within the last two years, so QuantRocket cannot source 1990–2015 natively, and Moonshot's built-in stitching (`CONT_FUT = "concat"`) is the unadjusted concatenation the Element warns against. QuantConnect's US futures data start in May 2009 and support backwards-ratio adjustment. Therefore split the futures work: (a) a fidelity replication of 1995–2015, possible only with purchased Pinnacle CLC (or equivalent) data loaded as a custom database with ratio adjustment built by hand; (b) a post-publication test in LEAN, training on 2009–2015 and testing on 2016–2026, on a smaller US-only universe. Verify roll dates against open-interest switching in either case.
2. Collect the point-in-time US equity universe including delisted securities; restrict to NYSE common stocks (CRSP share codes 10 and 11 equivalent) priced above $1 at each rebalance, as in the source paper, applying the screen to the contemporaneous unadjusted price. Verify exchange and share-type eligibility as of each date, and include dividends and delisting returns in realised returns. Run a second variant with a liquidity and borrow-availability filter, because the $1 floor admits hard-to-borrow names.
3. Store feature matrices in Parquet with explicit timestamps; confirm that every feature at date t uses only data through t.

### 6.2 Data Preprocessing Pipeline

1. **Load and validate**
   - Check continuity of futures series across rolls; flag gaps and holidays; confirm equity delisting returns are included.
2. **Clean**
   - For the DMN, follow Lim et al.: cap and floor the data at five exponentially weighted standard deviations around an exponentially weighted mean, both with a 252-day half-life. Percentile winsorisation would be a different experiment.
   - Forward-fill prices across exchange holidays only; never impute with future information.
3. **Transform**
   - Compute log returns; EWMA volatility with a 60-day span; normalised returns r/σ over 1, 21, 63, 126 and 252 trading days; MACD(8,24), (16,48), (32,96) normalised by 63-day price standard deviation and 252-day MACD standard deviation as in Baz et al. (2015).
   - For LTR, do not reuse the DMN features. Compute the equity paper's set at each month-end: raw and volatility-normalised returns over 3, 6 and 12 months and the current and lagged MACD indicators, with volatility from a 63-day exponentially weighted span. Labels are deciles of the return over the next 21 trading days. The 21-day label clock and the month-end rebalance clock are distinct; keep both explicit.
4. **Align**
   - Align all instruments on a common trading calendar; lag features by one day relative to the return they are paid on.

### 6.3 Analysis Steps

#### Step 1: Descriptive statistics
- Summary statistics, autocorrelation at lags 1–252 and Ljung–Box tests on normalised returns; verify volatility clustering; compute the futures correlation matrix to confirm block-diagonal structure.

#### Step 2: Classical benchmarks
- TSMOM: sign of 252-day return, volatility-scaled to 15%; MACD composite; long-only. Reproduce Table 5 benchmark Sharpe ratios of roughly 1.4, 1.1 and 0.76 for 1995–2015.
- Volatility-scaled long-only at matched risk, to separate the trend signal from the volatility-scaling effect.
- Cross-sectional: 12-month return and MACD composite scores, fixed 100 long and 100 short, monthly, inverse-volatility sizing identical to the ranker's (see 6.8).

#### Step 3: Main models
- **DMN**: single-layer LSTM (hidden size tuned in {5, 10, 20, 40, 80}), dropout {0.1–0.5}, tanh output; loss = negative Sharpe over the training set (Equation 81); Adam, learning rate in {1e-4, 1e-3, 1e-2}; sequence length 63 days; train on expanding windows, retrain every 5 years with 10% of the most recent training data as validation (as in Lim et al. 2019). Hyper-parameter search with 50 random draws or Optuna.
- **Momentum Transformer**: decoder-only TFT with LSTM encoder, 4 attention heads, hidden size in {5, 10, 20, 40, 80, 160}, sequence length 252 days (not 63), as in Wood et al.; same loss and retraining schedule; average over five seeds. Reference only (see priority order).
- **LTR**: LambdaMART via LightGBM (lambdarank objective, 10 relevance levels = deciles of 21-day-ahead returns), ListNet and RankNet in PyTorch; the paper's feature set (raw and normalised 3-, 6- and 12-month returns plus current and lagged MACD indicators), not the DMN set; re-tune every 5 years, rebalance monthly, select a fixed top and bottom 100, size each name by inverse volatility as in Equation 82, and group observations by decision date for the ranking objective. Add a boosted-tree regressor on the same features and dates as a control, to separate the value of ranking from the value of a non-linear tree model. The paper uses XGBoost with `rank:pairwise`; run that as the fidelity variant and record LightGBM lambdarank as a deliberate departure.
- **E2E portfolio**: score block = LSTM over 50-day windows of all assets' returns (the 50-day lag is an assumption; neither the Element nor the source paper states it); portfolio block = signed softmax; loss = negative Sharpe; weight-change penalty as an extension.
- **Standard errors**: Newey–West (lag 21 for daily, 6 for monthly) on mean returns; block bootstrap (block length 63 days) for Sharpe confidence intervals.

#### Step 4: Robustness checks
- Vary volatility target (10%, 15%, 20%); retraining frequency (1, 3, 5 years); random seeds (10 per configuration); exclude the three best-performing contracts; sub-period splits 1995–2007, 2008–2015, 2016–2026.

#### Step 5: Strategy backtesting
- **Framework**: Moonshot daily rebalancing with the model's positions as target weights; or LEAN with scheduled daily rebalance.
- **Transaction costs**: 1, 3 and 5 basis points of notional per unit turnover for futures; 10 basis points for equities, with timestamped IBKR borrow fees and short availability from April 2018 in place of a flat borrow assumption, and unavailable shorts removed under a rule fixed in advance; slippage half a tick. For equities, signal on the month-end close and execute at the next session's open; keep a paper-style close-to-close result alongside to measure the timing cost.
- **Metrics**: annualised return, volatility, Sharpe, Sortino, Calmar, maximum drawdown, hit rate, average profit over average loss, annualised turnover, probabilistic and deflated Sharpe ratio. For LTR also report long and short contributions separately, gross exposure, monthly Kendall's tau and NDCG@100 for each tail.

### 6.4 Statistical Tests

| Test | Purpose | Implementation Notes |
|------|---------|---------------------|
| Newey–West t-test | Significance of mean strategy return | statsmodels OLS with HAC covariance, maxlags 21 |
| Block bootstrap of Sharpe | Confidence interval accounting for autocorrelation | arch.bootstrap.StationaryBootstrap, 10,000 draws |
| Probabilistic / deflated Sharpe ratio | Adjust for non-normality and number of trials | Implement Bailey and López de Prado (2012, 2014) formulae; count all architectures and seeds as trials |
| Ledoit–Wolf (2008) Sharpe-difference test | Compare DMN versus TSMOM Sharpe ratios | HAC or studentised circular block bootstrap version; Diebold–Mariano is for forecast losses, not P&L, and is not appropriate here |
| Paired sub-period comparison | Stability across regimes | Report Sharpe per sub-period with bootstrap intervals |

### 6.5 Validation Approach

- **In-sample period**: 1990–1994 initial training window, expanding thereafter (requires Pinnacle CLC or equivalent); 2009–2015 for the LEAN-only variant.
- **Out-of-sample period**: 1995–2015 to reproduce the Element (purchased data only); then later periods labelled as post-sample and post-preprint, not pooled. For LTR: in-sample to 2019, post-sample 2020, post-preprint January 2021 onwards (about 70 months, so low power on its own). Any period that has already informed a design choice is retrospective; reserve subsequently arriving data for forward evaluation.
- **Cross-validation**: rolling chronological folds; validation set is the last 10% of each training window; no shuffling. Purge by label horizon at every training/validation and validation/test boundary: drop observations whose forward return was not yet known at the next decision time (one day for the DMN, 21 trading days for LTR).
- **Benchmark comparison**: TSMOM (Moskowitz, Ooi and Pedersen 2012), MACD trend composite, long-only; for equities, 12-month momentum decile portfolio and Fama–French UMD.

### 6.6 Python Libraries

| Library | Purpose |
|---------|---------|
| pandas, numpy | Data handling and feature construction |
| torch | DMN, Momentum Transformer, E2E portfolio networks |
| lightgbm, xgboost | LambdaMART (lambdarank objective; XGBoost `rank:pairwise` for fidelity to the paper) |
| optuna | Hyper-parameter search |
| statsmodels, arch | HAC standard errors, bootstrap |
| quantrocket-moonshot or LEAN | Backtest execution and cost modelling |
| shap | Feature attribution for the trained models |

### 6.7 Expected Outputs

1. A table reproducing Table 5 of the Element (DMN variants versus classical benchmarks, 1995–2015, volatility-rescaled), with added columns for turnover and net-of-cost Sharpe.
2. A table reproducing Table 6 sub-period results for the Momentum Transformer, extended to 2016–2026.
3. For LTR, one monthly attribution table: paper-style portfolio, executable portfolio, long contribution, short contribution, borrow drag, and net return difference against the MACD rule. A like-for-like copy of Table 8 is not attainable on post-1998 data.
4. Bootstrap confidence intervals and deflated Sharpe ratios for every model, and a seed-dispersion chart showing how much of the headline Sharpe survives averaging across random initialisations.

### 6.8 First experiment: LTR for a new cross-sectional equity sleeve

**Question.** Does a learned ranker earn a tradable net return on US equities, and does it beat the simple published momentum rules when all trade the same stocks with the same sizing and execution?

1. **Freeze four scoring rules before running anything**: trailing 12-month return; the MACD composite (the stronger published benchmark, gross Sharpe 0.696); XGBoost `rank:pairwise` on the paper's features; a boosted-tree regressor on the same features. All four share eligibility rules, 100 positions per side, inverse-volatility sizing, the portfolio risk overlay and the execution timestamp.
2. **Build the panel in QuantRocket with auditable timing**: features use only information available at the decision time; the $1 screen uses the unadjusted price; realised returns include dividends and delistings; execution at the next open, with the close-to-close variant kept for comparison.
3. **Make the short book the first diagnostic**: report long and short contributions separately, then rerun from April 2018 with actual borrow fees and with unavailable shorts removed. If the edge sits in names that cannot be borrowed, the long–short form fails; the long-only rankings remain a separate hypothesis from the same run.
4. **Evaluate ranking quality alongside returns**: Kendall's tau, NDCG@100 per tail, paired monthly net return differences against the MACD rule with a block bootstrap that resamples the same months for all strategies.

**Decision rule (fix the numbers before the run).** Advance to paper trading only if (a) the ranker's net Sharpe after executable timing, borrow and trading costs clears a pre-declared absolute hurdle, (b) its paired net return difference against the MACD rule is positive with a bootstrap interval excluding zero, and (c) both survive a pre-declared liquidity screen and a cost stress. If (a) holds but (b) fails, build the sleeve on the simple rule instead. If the advantage disappears when unavailable shorts are removed, stop the long–short implementation. Replay the frozen monthly targets in LEAN only after the gate is passed.
