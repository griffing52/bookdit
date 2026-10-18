# Evaluation protocol

What each reported aggregate means, and what has to match before two of them can be compared. A
LOB-Bench aggregate depends on the model, the window set, the decoder configuration and the
aggregate view, and all of those are pinned below. Paper references use the published paper's numbering
(Table 1 = distributional fidelity, Figure 2 = overview and long-horizon drift, Figure 3 =
divergence growth).

Contents: [1 the official lane](#1-the-official-lane) ·
[2 the three aggregate views](#2-the-three-aggregate-views) ·
[3 the conditioning arithmetic](#3-the-conditioning-arithmetic) ·
[4 the IDM decoder](#4-the-idm-decoder) · [5 the level-ten filter](#5-the-level-ten-filter) ·
[6 the LOBSTER sentinel](#6-the-lobster-empty-level-sentinel) ·
[7 window counts](#7-window-counts) ·
[8 the long-horizon and divergence lanes](#8-the-long-horizon-and-divergence-lanes)

## 1. The official lane

**Windows.** LOB-Bench's own released GOOG January-2023 evaluation windows, from the HuggingFace
release `peernagy/lob_bench`, over nine trading days:

```
2023-01-03  2023-01-04  2023-01-05  2023-01-06  2023-01-09
2023-01-10  2023-01-11  2023-01-12  2023-01-13
```

**Reference.** Every arm is scored against the real LOBSTER message stream for the same windows,
filtered to visible-book event types `(1, 2, 3, 4)` inside market hours `34200 ≤ t ≤ 57600` seconds
after midnight, the stream LOB-Bench's own real files are built from
(`scripts/baselines/build_lobbench_hf_real.py`, `KEEP_TYPES` and the `MKT_OPEN`/`MKT_CLOSE` bounds).

**Scorer.** LOB-Bench's own `scoring.run_benchmark`, with the repository's `DEFAULT_SCORING_CONFIG`
and `DEFAULT_METRICS`, from the pinned clone (`scripts/data/setup_libs.sh`, SHA
`276baf3039bf7630b555891b9d04ad49a92b3356`). The one replay of it, `scripts/analysis/_l1replay.py`,
which makes the window bootstrap affordable, is checked against LOB-Bench's own profiles by
`horizon_matched_bands.py --validate` over the three long-horizon arms (`ARMS` there).

**Direction of comparison.** Each statistic is LOB-Bench's half-total-variation L1 between the
generated and the real empirical distribution, pooled across the window set; the aggregate is the
unweighted mean over the statistics in the view (§2).

The HuggingFace release ships only `data_gen`; `data_real` and `data_cond` are rebuilt locally
from your own LOBSTER data (`docs/REPRODUCE.md` §3.2).

## 2. The three aggregate views

LOB-Bench's default configuration produces 21 statistics. Three aggregate views over them are
reported, always labelled:

| View | Definition |
|---|---|
| **L1²¹** | unweighted mean over all 21 statistics |
| **L1¹⁸** | L1²¹ minus `log_inter_arrival_time`, `log_time_to_cancel`, `vol_per_min` |
| **book-10** | mean over the ten book-geometry statistics listed below |

The 21, in the order the scorer's `per_stat_l1.csv` uses:

```
spread                    orderbook_imbalance       log_inter_arrival_time
log_time_to_cancel        ask_volume_touch          bid_volume_touch
ask_volume                bid_volume                limit_ask_order_depth
limit_bid_order_depth     ask_cancellation_depth    bid_cancellation_depth
limit_ask_order_levels    limit_bid_order_levels    ask_cancellation_levels
bid_cancellation_levels   vol_per_min               ofi
ofi_up                    ofi_stay                  ofi_down
```

book-10 is exactly `spread`, `orderbook_imbalance`, `ask_volume_touch`, `bid_volume_touch`,
`ask_volume`, `bid_volume`, `ofi`, `ofi_up`, `ofi_stay`, `ofi_down`: the `BOOK10_STATS` constant,
defined identically in `scripts/baselines/score_official.py`,
`scripts/baselines/horizon_official_protocol.py` and `scripts/baselines/rescore_horizon_fixedset.py`.
The statistics L1¹⁸ drops are `CLOCK_STATS` in the first two.

**Why L1¹⁸ exists.** The dropped statistics are message-timing quantities. A message model emits
them directly; on the book lane they exist only once the inverse-dynamics decoder has worked out
how many messages produced each frame transition and when inside the frame each landed (§4), so
there they measure the decoder at least as much as the model. L1¹⁸ is therefore the
*clock-excluded* view. It is not fully representation-matched: some of its statistics still depend
on event information a top-ten state does not carry. book-10 is the narrowest view of visible-book fidelity. `vol_per_min` also breaks down in
the horizon-resolved lanes (the `DEGENERATE` set in `scripts/baselines/slopes_divergence_official.py`).

## 3. The conditioning arithmetic

| Quantity | Value |
|---|---|
| corpus stride | 1 book frame per 10 raw messages |
| model window | 500 frames (`seq_len` in the checkpoint) |
| conditioning | 250 frames = 2,500 raw messages |
| generated and decoded | the first 50 inpainted frames = 500 messages of event time |

`main()` and `generate()` in `scripts/baselines/gen_bookdit_on_lobbench_windows.py` read these off
the checkpoint and the CLI. The model runs on one full 500-frame window with the 250-frame
conditioning prefix observed and the 250-frame suffix inpainted (`flow_rollout(...,
observe_frac=0.5)`). Only the first `--horizon-msgs / --stride` = 50 generated frames are kept; the
rest of the suffix exists because the model's window length is fixed, and is discarded.

**What is scored.** The decoder runs over the conditioning frames followed by the 50 generated
frames, so that orders resting before the horizon have real lifetimes, and every decoded message
after the last conditioning timestamp is written and scored. Nothing caps the count at 500: the
number of messages varies by window and is generally not the real continuation's count. A window
whose generated part decodes to fewer than 8 messages is dropped. The output horizon is therefore
the same in event time (500 messages' worth of book states) as the message models' 500 messages,
not the same in message count.

A message model on this benchmark sees a 500-message prompt and emits 500 messages. The BookDiT
window is located in the real stream by timestamp, not by index: the last timestamp of the
conditioning file is found in the filtered stream and the context steps back from there by the
conditioning length above (`day_windows`, same file); `docs/REPRODUCE.md` §3.2 gives the rule.

**Evaluation grid vs training grid.** The corpus takes a frame on the `k % 10 == 0` grid over the
raw message stream (`docs/REPRODUCE.md` §3.1); the official-window driver steps by 10 over the
*filtered* stream. A frame therefore spans slightly more raw messages at evaluation time than at
training time, by the rate of hidden executions and halts (the grid note in `day_windows`).

## 4. The IDM decoder

BookDiT generates book states; LOB-Bench scores message streams. The bridge is an inverse-dynamics
(IDM) decoder: a snapshot-diff procedure that reads a pair of consecutive book frames, works out
which limit-order events must have occurred to turn the first into the second, and emits them as
LOBSTER messages with types, sides, prices, sizes and timestamps (`decode_book_sequence` in
`scripts/lib/idm_decoder.py`; the FIFO queue tracker and order-level type rule are in
`scripts/lib/idm_annotator.py`). The decode has a cost, and the repo measures it: the book-state
decode floor row of Table 1 is the IDM decode of the *real* frames, scored the same way. It
reflects both information absent from the retained states and the limits of the deterministic
reconstruction. The two floor rows sit on their own window sets; §7 gives the counts.

*Known simplification.* A cancel that matches no resting order by size and is larger than the front
resting order removes only that order, so the synthetic FIFO queue can exceed the level size until the level empties.
This affects only synthetic order-id attribution (the lifecycle statistics in L1²¹), not book
states, and the reported numbers were produced with this behaviour (`QueueBook.reduce` docstring).

**The standard decoder: two switches, held fixed, `IDM_GAP_PROFILE=1` and `IDM_DEADBAND_ABS=50`.**

*Gap profile* (intra-frame timing). One book transition implies *n* messages, and the decoder places
them inside the elapsed frame interval `(t[i-1], t[i]]`. Real intra-frame gaps are bursty rather
than evenly spaced, so the decoder samples each frame's composition from an empirical gap marginal;
a marginal is the right object because the statistic it feeds, `log_inter_arrival_time`, is itself
a marginal. The table ships at `scripts/lib/intraframe_gap_profile.npz` and was fitted on six days
of GOOG 2024 message times, a different year from the 2023 evaluation windows. Seeded per frame, so
it is reproducible. (`GAP_PROFILE`, `_gap_weights`.)

*Deadband* (resting-size sparsity). Real book sizes are piecewise constant; an MSE-trained model
moves nearly every size cell by a small amount, and the decoder turns each of those into a message.
A carry-forward deadband restores the sparsity: a size is held until the pending change exceeds 50
shares, then the whole accumulated change is emitted at once, so suppressed displacement is
conserved rather than discarded. The decoder's built-in band (`DEADBAND_ABS`,
`_carry_forward_deadband`) acts on the raw size columns.

Both switches are the code defaults where `idm_decoder.py` reads them from the environment, so a
fresh clone reproduces the standard decoder with nothing set.

**Refinement, and the two Table 1 BookDiT rows.** Refinement is a local median filter (window 3)
over the deep-level sizes (levels 4–10, the `VERYDEEP_IDX` column set), applied to the generated
frames before decoding (`refine()` in `gen_bookdit_on_lobbench_windows.py`, behind `--refine`);
prices, spread and mid are unchanged. It is applied to the 40-column level-10 prefix even when a
model emits more levels, so the operation is identical across depth arms. The two Table 1 rows
share the checkpoint, the generation settings, the decoder and the deadband, and differ only in the
median filter:

| Table 1 row | median filter | where the 50-share deadband acts | how it is run |
|---|---|---|---|
| BookDiT + refinement (`bookdit_refine__aggregates.csv`, `abs50`) | yes | before decoding, in the decoder's quantized (side, price) space | `--refine --dump-frames`, then `scripts/baselines/apply_deadband.py --settings abs50` |
| BookDiT, without refinement (`bookdit_raw__aggregates.csv`, `nomed`) | no | the same | `--dump-frames` without `--refine`, then the same `apply_deadband.py --settings abs50` |

`apply_deadband.py` sets `IDM_DEADBAND_ABS=0` before importing the decoder, so the decoder's
built-in band is off and the 50-share band is applied once, in quantized space. Generated prices
are off the tick grid and the decoder quantizes them before diffing, so a size column does not
track one price from frame to frame; the quantized-space band follows prices. The `raw` row of
`bookdit_refine__aggregates.csv` is the refined generation with no deadband at all, and the `raw`
row of `bookdit_raw__aggregates.csv` is the unrefined generation decoded directly with the
decoder's built-in column-space band (the submitted version's "without refinement" row); neither
is reported in the camera-ready. The long-horizon lane (§8) uses `--refine` with the decoder's
built-in band.

## 5. The level-ten filter

The filter is LOB-Bench's own `data_loading.cut_data_to_lvl(in_dir, out_dir, 10)`, applied to every
arm's generation directory before scoring (`filtered_keys` in `scripts/baselines/score_official.py`).
For each message row it reads the book *after* that message, truncated to ten levels, and keeps
the message only if its price lies between that book's level-10 bid price and level-10 ask price
(`filter_by_lvl`); the book rows of dropped messages go with them. Only a `-1` level price is
mapped to the LOBSTER sentinel before the comparison. LOBSTER's own files mark empty levels with
the `±9999999999` sentinel (§6), so on the real side a shallow book never blocks a message. The IDM
decoder instead writes empty levels as price 0 (`idm_annotator.QueueBook.snapshot` starts from
zeros), so for every decoded arm a book whose ask side has fewer than ten levels gets an upper bound
of 0 and the message emitted into it is dropped; a shallow bid side passes, because its lower bound
of 0 admits every price. The filter therefore removes messages beyond level ten for any generator,
and removes everything a generator emits while its ask side is shallower than ten levels. That is
why a shallow generator such as Painting the Market, which fills 4.8 of ten levels on average,
loses most of its windows here (§7).

Two effects then change the window count, which is why the *n* column varies across Table 1 rows.
Filtering can leave a message file empty, and an empty CSV crashes LOB-Bench's `Simple_Loader`
time parser, so `filtered_keys` deletes empty files after filtering and that window is gone for that
arm. Each arm also loses a few windows upstream, to alignment failures: a window too close to the
start or end of the day to supply the history plus the horizon. `score_official.py` scores the
intersection of the surviving sets by default (`--no-intersect` scores each arm on its own); where
Table 1 lists rows side by side they come from separate runs, and §7 says which share a set.

## 6. The LOBSTER empty-level sentinel

LOBSTER marks an empty book level with `+9999999999` on the ask side and `-9999999999` on the bid
side. The ask sentinel is positive, so a `price > 0` populated-level test admits it as a real quote
and reads a spurious inter-level gap as book geometry. The correct mask is

```python
0 < price < 1e8      # 1e8 LOBSTER units = $10,000
```

which separates real US-equity prices from the sentinel. Reference implementation: `PX_MAX` and
`_px_ok` in `scripts/analysis/book_native_facts.py`.

## 7. Window counts

Table 1 is assembled from separate runs on different window counts. Which rows share a set:

| Table 1 row | n | Shares its set with | Lane |
|---|---|---|---|
| BookDiT + refinement | 3,127 | BookDiT, without refinement | book-state |
| BookDiT, without refinement | 3,127 | BookDiT + refinement | book-state |
| BookDiT, data-matched | 3,111 | re-scores of BookDiT + refinement, LOB-S5 compute-matched and six training seeds of the released configuration, in the same file | book-state |
| LOB-S5, released | 3,124 | LOB-S5 compute-matched, LOB-Bench baseline | message |
| LOB-S5, compute-matched | 3,124 | LOB-S5 released, LOB-Bench baseline | message |
| LOB-Bench baseline | 3,124 | the two LOB-S5 rows | message |
| Painting the Market | 844 | a re-score of BookDiT + refinement, in the same file | book-state |
| RWKV-4 / RWKV-6 | 582 | each other, and re-scores of BookDiT, LOB-S5 released and both floors in the same file | message |
| book-state decode floor | 3,132 | none | book-state |
| real-vs-real floor | 1,562 | disjoint pairs of real windows | real vs real |

Rows on different *n* are not directly comparable: a window-set change moves these quantities on
its own, independently of the model. For a same-set BookDiT-versus-LOB-S5 comparison use the
paper's same-set check: `bookdit_refine` (the checkpoint reported in the paper) and
`lobs5_retrain`, on the 3,111 windows both keep together with the three data-matched seeds and the
six seeds `seed0`–`seed5`; `n_survive_filter` is each arm's own count before the intersection.

The Painting row's 844 is an intersection: 847 of its 3,136 windows survive the level-ten filter,
and 844 of those are also in the BookDiT re-score it is compared against (`n_survive_filter` and
`n_windows` in the `aggregates.csv` that `run_painting.sh` writes, which holds both arms).

## 8. The long-horizon and divergence lanes

Both are the same official lane (real-LOBSTER reference, standard decoder, level-ten filter),
resolved by horizon instead of collapsed to one number.

**Long-horizon drift (paper §5; Figure 2, right; `horizon_bands.pdf`).** The
reported quantity is an OLS slope of L1¹⁸ across blocks; `bootstrap.json` stores it as
`slope_per_1k` (L1 per 1,000 generated messages; × 1000 gives the paper's 10⁻³ units).

* BookDiT is run with `--refine` and the standard decoder; LOB-S5 is the compute-matched retrain;
  CST is the Cont–Stoikov–Talreja model.
* 39 windows, 32 successive 500-message blocks out to 16,000 messages, B = 1,000 bootstrap
  replicates, R = 24, seed `20260804`.
* The window set is held fixed across blocks: the driver's own `profile.csv` counts windows entering
  the level-ten filter rather than windows actually scored, and the surviving set changes block to
  block, so a slope fitted across blocks of moving composition is confounded.
  `scripts/baselines/rescore_horizon_fixedset.py` recomputes on the per-arm all-block set, and
  `horizon_matched_bands.py --population matched` intersects across arms, which is where the 39
  comes from.
* Known artefact (not fixed): at h = 15.5k (block 30) every BookDiT run loses about 50 windows,
  because the generated clock jumps backwards at frame 1,750, a rollout chunk boundary, so the
  decoded messages fall outside the block's time range. Re-running the lane reproduces this. The
  fixed 39-window set keeps only windows scored at every block, so it excludes them.
* The bootstrap resamples windows, not blocks, and is paired: one window multiset per replicate,
  applied to all arms.
* The real-vs-real control uses 76 carrier windows split 38/38 (real continuations scored against
  real continuations through the identical pipeline) and is not paired with the model arms
  (`floor_paired: false`).
* Outputs: `bootstrap.json`, with the window ids listed explicitly, and `curve_bands_l1_18.csv`,
  written by `horizon_matched_bands.py` to `runs/horizon/bands/`.

**Divergence (paper §5; Figure 3; `divergence_drift.pdf`).** LOB-Bench's own
`--div_only` mode, comparing BookDiT with the *released* LOB-S5 samples on 900 windows. The reported
quantity is an L1 slope at a given `divergence_horizon` H; headline at H = 100, with H = 50 as a
robustness check.

* The curve is the mean over the 16 well-defined statistics: the `DEGENERATE` statistics (`ofi`,
  `ofi_up`, `ofi_stay`, `ofi_down`, `vol_per_min`) are excluded because a short sub-sequence leaves
  them with no usable bins and `l1_by_group` then returns a spurious zero (the set in
  `scripts/baselines/slopes_divergence_official.py`, subtracted per arm). This 16-statistic
  aggregate is a different view from L1²¹, L1¹⁸ and book-10, so its levels are not compared with
  Table 1.
* Arms are depth-matched: at each horizon every compared arm is truncated to the shallowest depth
  any of them reaches, otherwise each slope is fitted over a different span (the `--match` arms in
  `slopes_divergence_official.py`). A horizon group is kept only if at least the `--min-full 0.98`
  fraction of windows supplies a full sub-sequence there. The reported slopes are fitted over a
  matched span of 300 generated messages (3 groups at H = 100, 6 at H = 50; `span_msgs`).
* Excess is measured against a real-vs-real control arm (`realgen`, whose generated side is the real
  continuation), same protocol. Outputs: `runs/divergence/headline.csv` and
  `runs/divergence/curve_bands.csv`.
* The paper's statement that real books passed through the decoder produce no measurable drift is
  the `ditfloor` row of `headline.csv`: H = 50 only, 3 groups (150 messages), fitted over its own
  depth and not part of the matched set, decoded with an earlier revision of the IDM decoder than
  the `bookdit` arm; slope −0.00098 per 1,000 messages, 95% interval [−0.0301, +0.0242], `sig = no`.
