# artifacts/

Metadata the pipeline needs to line up with the paper's runs. Nothing here is a LOBSTER record:
the files list tickers, days, splits and window positions, not prices, sizes or messages. No
results, figures or weights ship; the scripts regenerate them under `runs/`
([`../docs/REPRODUCE.md`](../docs/REPRODUCE.md)). Everything in this directory is MIT
([`../LICENSE`](../LICENSE)).

| File | What it is | Read by |
|---|---|---|
| `corpus/corpus_manifest_2024.csv` | the training corpus: 5,142 ticker-days, 21 names, 2024, split 5,087 train / 44 val / 11 test | documentation of the day list and split; train from the manifest your own `build_corpus.py` run writes (REPRODUCE §3.1) |
| `tab1_baselines/lobs5_starts/lobs5_starts_<YYYYMMDD>.json` | LOB-S5 window starts for Table 1, one file per January-2023 evaluation day (nine days, 3,130 windows) | `scripts/baselines/gen_lobs5_on_lobbench_windows.py --starts-json` (REPRODUCE §3.7) |
| `fig_horizon/lobs5_starts_20230104_shard{0,1,2,3}.json` | LOB-S5 window starts for the 16,000-message horizon run on 2023-01-04, four shards (107 windows) | `gen_lobs5_on_lobbench_windows.py --starts-json` (REPRODUCE §3.9) |

**Manifest.** Columns `ticker,day,year,split,tensor_path,status`, one row per ticker-day, the
format `scripts/data/build_corpus.py --manifest-out` writes. `tensor_path` holds
`${LOBSTER_CACHE}/tensors_event_contig/<TICKER>_<YYYYMMDD>_event10_contig.pt` placeholders where a
fresh build writes absolute paths; `status` is `ready` on every row.

**Window-start files.** A JSON list of `{"rid": <LOB-Bench real_id>, "start": <row>}` objects.
`rid` names the LOB-Bench window; `start` is the row of that day's LOB-S5 proc array
(`GOOG_<YYYYMMDD>_message_proc.npy`) at which the scored continuation begins, the conditioning
window being the `--cond` rows before it. The rows only line up with a proc array built as
REPRODUCE §3.7 describes; check its row counts there before sampling.
