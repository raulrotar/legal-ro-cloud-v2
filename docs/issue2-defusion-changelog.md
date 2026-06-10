# Issue 2 — Word De-fusion: Change Log & Revert Guide

## What was changed

This document records every change made for Issue 2 (word de-fusion for broken_2007 era)
so the feature can be reverted safely at any time.

### Files added
- `packages/core/src/legalro_core/data/ro_unigrams.txt`
  Vendored frozen Romanian unigram frequency list (4,287 words, built from clean modern-era
  MD corpus). Used by the Viterbi segmenter. **No runtime dependency added.**

- `docs/issue2-defusion-changelog.md` (this file)

### Files modified

#### `packages/core/src/legalro_core/normalize.py`
Added functions and constants (all in a clearly delimited block):
- `_fold_ro(s)` — diacritic-folding helper for dictionary lookup.
- `_load_ro_lexicon()` — lazy loader for `ro_unigrams.txt` into `_RO_WORD_COSTS` / `_RO_WORD_SET`.
- `_viterbi_segment(token)` — Norvig/Viterbi max-probability segmenter with acceptance gate.
- `_DEFUSE_SKIP_LINE` — regex for lines to skip (signatures, headings, act numbers).
- `defuse_words(text, audit_log)` — public entry point; gate + segment + audit.
- `DEFUSE_WORDS_ENABLED` — boolean flag, defaults True; set `LEGALRO_DEFUSE_WORDS=0` to disable.

#### `packages/processing/src/legalro_processing/extract/pipeline.py`
Added step 7 to `_normalize_gazette_md` (~line 510):
- Calls `defuse_words()` when `era == Era.BROKEN_2007` and `DEFUSE_WORDS_ENABLED`.
- Writes per-run audit JSONL to `db/defuse_audit/defuse_<timestamp>.jsonl`.

## How to disable without reverting code

Set the environment variable before running any extraction command:
```bash
LEGALRO_DEFUSE_WORDS=0 uv run legalro-process extract ...
```

Or set it permanently in your shell / systemd unit.

## How to verify impact

1. Check the audit log after a full extract:
   ```bash
   cat db/defuse_audit/defuse_*.jsonl | python -c "import sys,json; [print(json.loads(l)) for l in sys.stdin]"
   ```
2. Confirm every logged `token` equals `''.join(parts)` (character-preservation invariant).
3. Spot-check suspicious splits (proper nouns, long legal words) in the audit log.
4. Run: `uv run pytest tests/test_normalize.py -k defuse -v`

## How to revert completely

Option A — disable via env var (instant, no code change):
```
LEGALRO_DEFUSE_WORDS=0
```

Option B — revert the commit that introduced Issue 2:
```bash
git log --oneline | grep -i "defus\|word.fus"   # find commit SHA
git revert <SHA>
```

Option C — manual revert:
1. In `normalize.py`: remove everything from `# ── Romanian word de-fusion` to the end of `DEFUSE_WORDS_ENABLED`.
2. In `pipeline.py`: remove step 7 block (`# 7. Word de-fusion for broken_2007 era.`).
3. Delete `packages/core/src/legalro_core/data/ro_unigrams.txt`.

After any revert, delete `db/defuse_audit/` (optional) and re-run extraction.
