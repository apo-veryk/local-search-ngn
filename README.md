# Local CSV ↔ Image Fuzzy Matcher (v2)

A Python tool that matches rows from a CSV (e.g., product names + IDs) to image files in a local folder.
It combines **RapidFuzz** fuzzy search with quantity parsing (ml/l/g), diacritic-insensitive tokenization (Greek + Latin), and an **optional semantic dictionary** to boost precision.

The app uses simple **Tkinter** dialogs for selecting inputs/outputs and writes a detailed summary for review.

---

## Highlights (What’s new in v2)

* **Smarter text normalization**: Unicode cleanup + accent/diacritic stripping + stopword filtering.
* **Promo scrubber**: strips “3+1”, percentage discounts (e.g., `20% off`, `20% εκπτωση`), and common promo words.
* **Quantity awareness**: detects `ml`, `l`, `g` and requires near-equal quantities between CSV name and image filename.
* **Semantic dictionary (optional)**:

  * Map aliases/phrases (incl. hyphen/space variants and gapped phrases) to **canonical** terms.
  * Enforce **critical category** symmetry (e.g., both sides must agree on `drink`, `topping`, etc.).
  * Forbid certain concept combinations.
* **Two-pass matcher**:

  1. **Fuzzy** (token-set) with tie-breaking and adaptive thresholds.
  2. **Prefix-overlap** ("startswith") that ignores tiny/non‑alpha tokens.
* **Anchor-word fallback** when no semantic concepts detected.
* **Usage caps** per image key to avoid over-reuse.
* **Review-first output**: copies images using safe human‑readable filenames; after you review the folder, a one-click prompt **renames to `item_id`**.
* **Progress bar** via `tqdm`.

---

## Requirements

* Python **3.10+**
* `pandas`
* `rapidfuzz`
* `tqdm`

> Tkinter ships with most Python builds. If it’s missing on Linux, install your distro’s Tk packages.

Create a `requirements.txt`:

```
pandas>=2.0
rapidfuzz>=3.0
tqdm>=4.66
```

Install:

```
pip install -r requirements.txt
```

---

## Expected CSV schema

CSV must contain at least:

* `name` — the item/product display name
* `item_id` — the unique identifier used for final renaming

Example:

| item_id | name                                      |
| ------: | ----------------------------------------- |
|     123 | Espresso Single Lungo 0.33 l              |
|     456 | Sauces – BBQ Topping 250 g (2+1 προσφορα) |

---

## Optional: Semantic dictionary CSV

Provide a CSV with these columns (strings; empty allowed):

| group_id | canonical | tags            | phrase_aliases         | category | forbid_with |
| -------- | --------- | --------------- | ---------------------- | -------- | ----------- |
| drink    | drink     | beverage;drinks | αναψυκτικο; soft drink | drink    | topping     |
| topping  | topping   | sauce; dressing | σως; bbq sauce         | topping  | drink       |

* **group_id**: stable ID for a concept (referenced in `forbid_with`).
* **canonical**: normalized token to inject/compare.
* **tags**: semicolon‑separated single‑word aliases.
* **phrase_aliases**: semicolon‑separated phrases (hyphen/space variants, gapped matching supported).
* **category**: free text; used to enforce **critical** categories (see config flags).
* **forbid_with**: semicolon‑separated `group_id`s that should not co‑occur.

You will be prompted to load this file when the app starts (optional).

---

## How matching works

1. **Index images**

   * Walks the selected image folder (recursively) and indexes `*.jpg|*.jpeg|*.png`.
   * For each filename (without extension):

     * Normalizes text (unicode → lower → no accents → stopword removal → symbol cleanup).
     * Extracts quantities (`ml`, `l`→`ml`, `g`).
     * Detects semantic concepts.

2. **First pass: Fuzzy**

   * `rapidfuzz.process.extract` using **token_set_ratio**.
   * Adaptive threshold: 80 (≤2 tokens), 70 (≤4), 60 (>4).
   * Tie‑break by closeness in normalized length.
   * **Gates** (must all pass): quantities compatible ±0.5, concept/anchor check, image usage under cap.

3. **Second pass: Prefix overlap**

   * Counts pairs of tokens where `it.startswith(im)` or `im.startswith(it)` using only alphabetic tokens with length ≥3.
   * Tries overlap levels 3 → 2 → 1, still gated by quantities and concept/anchor.

4. **Output**

   * Copies matched images to the output folder using a **safe, human-readable** filename derived from `name`.
   * Writes `match_summary.csv` and `review_index.csv`.
   * After manual review, an in-app prompt can **rename files to `<item_id>.<ext>`**.

---

## Running the tool

```
python local_search_engine.py
```

You’ll be asked to:

1. Pick your CSV (defaults to `Downloads`).
2. Pick the images folder.
3. Optionally load a semantic dictionary CSV.
4. Provide an output folder name. The app creates `<HOME>/search-results/<your-name>` (or similar) and drops results there.

> If files with the same safe name exist, the tool will append `(2)`, `(3)`, …

---

## Config flags (top of script)

* `MIN_PREFIX_LEN` (default 3): ignore tiny/non-alpha tokens in prefix matching.
* `ANCHOR_RATIO_THRESHOLD` (default 75): similarity required for fallback anchor matches.
* `MAX_IMAGE_USAGE` (default 10): limit re-use of a single image key.
* `SEMANTIC_REQUIRE_SHARED_CONCEPT` (default True): require ≥1 shared concept when the product has any.
* `CRITICAL_CATEGORIES` + `STRICT_SYMMETRIC_CRITICAL` + `REQUIRE_ALL_PRODUCT_CRITICAL`: enforce category-level symmetry.
* `PHRASE_MAX_GAP` (default 2): allow up to N in-between tokens for phrase matches.

---

## Tips & troubleshooting

* **Quantities**: only `ml`, `l`, and `g` are detected. Extend `extract_quantities` to support more units (e.g., `kg`, `oz`).
* **Performance**: lots of nested folders/images are supported; the indexer scans recursively. SSDs will help.
* **Encodings**: if your CSV isn’t UTF-8, pass `encoding=...` to `pandas.read_csv`.
* **Images not found**: ensure filenames are descriptive (e.g., include the same language and key terms).
* **GUI missing on Linux**: install `python3-tk` or your distribution’s Tk packages.

---

## CLI/Automation (optional)

For headless use, you can refactor dialog prompts into CLI flags (e.g., `--csv`, `--images`, `--out`, `--semdict`).

---

## Project structure

```
.
├─ local_search_engine.py        # main script
├─ requirements.txt
├─ README.md
└─ sample_data/                  # (optional) add tiny demo CSV/images here
```

Add a `.gitignore` like:

```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
venv/

# OS
.DS_Store
Thumbs.db

# Outputs
search-results/
review*/
*.csv
*.log

# IDE
.vscode/
.idea/
```

---

## License

MIT License — see `LICENSE`.

---

## Quick GitHub publish (new or existing repo)

```bash
# from your project folder
git init
python -m venv .venv
. .venv/Scripts/activate  # Windows
# or: source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# make sure README.md, requirements.txt, .gitignore, LICENSE are present
git add .
git commit -m "feat: v2 matcher (semantics, quantities, two-pass, review-first)"

# create the repo on GitHub first, then:
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

To tag a release:

```
git tag -a v2.0.0 -m "v2 stable"
git push origin v2.0.0
```

---

## Safety / privacy notes

* Avoid hardcoding company paths, names, or identifiers in the script. Keep output under a generic folder (e.g., `~/search-results`).
* Don’t commit real customer CSVs or proprietary images. Use `sample_data/` with synthetic examples for the repo.
* If your comments mention internal systems or codenames, delete/rewrite them before pushing.
