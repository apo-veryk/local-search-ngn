import os
import pandas as pd
from rapidfuzz import process, fuzz
import shutil
import re
import unicodedata                   
from functools import cmp_to_key 
from tqdm import tqdm

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# CONFIG
# min length of tokens considered in STARTSWITH matching (ignore 1-2 char tokens)
MIN_PREFIX_LEN = 3

# Anchor-word verification tunes (used as fallback when no semantic concept is detected)
# ignore first words shorter than this or non-alpha
ANCHOR_MIN_LEN = 3                    
ANCHOR_RATIO_THRESHOLD = 75            

# usage cap per image key
MAX_IMAGE_USAGE = 10

# dictionary behavior
# if product has ≥1 concepts, candidate must share ≥1
SEMANTIC_REQUIRE_SHARED_CONCEPT = True   
SEMANTIC_DICT_PATH = None

# categories that must be symmetric , present on both sides with the SAME concept
CRITICAL_CATEGORIES = {"prepared_food", "topping", "tag", "drink"}
STRICT_SYMMETRIC_CRITICAL = True     
REQUIRE_ALL_PRODUCT_CRITICAL = True  

# allow up to N extra tokens between words in a phrase (e.g., "σε … πιτα")
PHRASE_MAX_GAP = 2 
# 1 is enough for "σε Κυπριακή πιτα" - 2 gives extra robustness

# UI pick inputs 
root = tk.Tk()
root.withdraw()

download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
while True:
    print("Select CSV FILE!")
    csv_path = filedialog.askopenfilename(
        initialdir=download_dir,
        title="select your CSV file",
        filetypes=(("CSV Files", "*.csv"),)
    )
    if csv_path:
        break
    else:
        if messagebox.askyesno("Quit?", "no CSV file selected. wanna quit?"):
            exit()

home_dir = os.path.expanduser("~")
print("Select IMAGES FOLDER!")
images_folder = filedialog.askdirectory(
    initialdir=home_dir,
    title="select your Images Folder"
)

# ask for an optional semantic dictionary CSV
if messagebox.askyesno("Semantic dictionary", "Load a semantic dictionary CSV now? (optional)"):
    SEMANTIC_DICT_PATH = filedialog.askopenfilename(
        initialdir=download_dir,
        title="select your semantic dictionary CSV",
        filetypes=(("CSV Files", "*.csv"),)
    ) or None

while True:
    output_folder_name = simpledialog.askstring(
        "Output Folder Name",
        "Enter the output folder name (NO SPACES):"
    )
    if output_folder_name:
        invalid_chars = set(r'\/:*?"<>|')
        if " " in output_folder_name or any(c in invalid_chars for c in output_folder_name):
            messagebox.showerror(
                "Invalid Folder Name",
                "invalid characters detected!!!\ndo NOT include spaces or any of these characters:\n/ \\ : * ? \" < > |\n\nre-enter a valid folder name"
            )
            continue
        else:
            break
    else:
        if messagebox.askyesno("Quit?", "no folder name entered. wanna quit?"):
            exit()

base_path = os.path.join(home_dir, "generix-photos", "0.generix-search-results")
output_folder = os.path.join(base_path, output_folder_name)
os.makedirs(output_folder, exist_ok=True)
messagebox.showinfo("Folder Created", f"Output folder created at:\n{output_folder}")

print("CSV file selected:", csv_path)
print("Images folder selected:", images_folder)
print("Output folder:", output_folder)
print("Semantic dictionary:", SEMANTIC_DICT_PATH or "(none)")


#stopwords & promo words

STOPWORDS = {
    "και", "με", "σε", "για", "απο", "που",
    "ο", "του", "τον", "η", "της", "την", "το",
    "οι", "των", "τους", "τις", "τα",
    "στον", "στην", "στη", "στο", "στις", "στους", "στων"
}

PROMO_WORDS = {
    "free", "gift", "offer", "offers", "promo", "deal", "discount", "off",
    "δωρο", "δωρα", "προσφορα", "προσφορες", "εκπτωση", "εκπτωσεις"
}

# helpers: accents, promos, symbols 
def strip_accents(text: str) -> str:
    return ''.join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def remove_promos(text: str) -> str:
    # removing promotional phrases & patterns + leaving quantities intact
    t = strip_accents(normalize_unicode(text)).lower()

    # remove % discounts like "30%", "30 % off", "20% εκπτωση"
    t = re.sub(r'\b\d+\s*%\s*(off|deal|discount|εκπτωση|προσφορα|προσφορες)?\b', ' ', t, flags=re.IGNORECASE)

    # remove bundles like "3 + 1", optionally followed by δωρο/free/gift
    t = re.sub(r'\b\d+\s*\+\s*\d+\b(?:\s*(δωρο|free|gift))?', ' ', t, flags=re.IGNORECASE)

    # remove standalone promo words
    for w in PROMO_WORDS:
        t = re.sub(rf'\b{re.escape(w)}\b', ' ', t, flags=re.IGNORECASE)

    # collapse spaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def symbol_cleanup_keep_decimals(text: str) -> str:
    
    # treat almost all symbols as spaces, while preserving decimal separators when between digits
    # convert comma decimal to dot & '_' and '-' into spaces
    t = strip_accents(normalize_unicode(text)).lower()

    # protect decimals: replace 1,5 or 1.5 > 1⟂5
    t = re.sub(r'(?<=\d)[\.,](?=\d)', '⟂', t)

    # replace symbols (including _ and -) with spaces (hyphen placed at end to avoid range)
    t = re.sub(r"[+\/\"'&%#_!()*|?<>\[\]{}:;=~^`-]+", " ", t)

    # restore decimals to dot, collapse whitespace
    t = t.replace('⟂', '.')
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# SEMANTIC DICTIONARY 
SEMANTICS = None

class SemanticRow:
    __slots__ = ("group_id","canonical","aliases","phrases","category","forbid")
    def __init__(self, group_id, canonical, aliases, phrases, category, forbid):
        self.group_id = group_id
        self.canonical = canonical
        self.aliases = aliases  # list[str]
        self.phrases = phrases  # list[str]
        self.category = category
        self.forbid = forbid    # set[str] of group_ids

def hyphen_space_gapped_pattern(phrase: str, max_gap: int) -> re.Pattern:

    # building a regex for a phrase where internal separators are hyphen/space, allowing up to 'max_gap' extra tokens between words
    # ex: "σε πιτα" => r"\bσε(?:[-\\s]+\\w+){0,max_gap}[-\\s]+πιτα\\b"
    
    s = strip_accents(phrase).lower().strip()
    s = re.sub(r"\s+", " ", s)
    parts = re.split(r"[-\s]+", s)
    if len(parts) == 1:
        # single-token phrases: just word-boundary match
        pat = rf"\b{re.escape(parts[0])}\b"
    else:
        # allow up to max_gap extra tokens between parts
        gap = rf"(?:[-\s]+\w+){{0,{max_gap}}}[-\s]+"
        pat = r"\b" + gap.join(map(re.escape, parts)) + r"\b"
    return re.compile(pat, re.IGNORECASE)

def load_semantic_dict(path: str):
    if not path:
        return None
    df = pd.read_csv(path, dtype=str).fillna("")

    rows = []
    for _, r in df.iterrows():
        group_id   = strip_accents(str(r.get("group_id",""))).lower().strip()
        canonical  = strip_accents(str(r.get("canonical",""))).lower().strip()
        aliases    = [strip_accents(a).lower().strip() for a in str(r.get("tags","" )).split(";") if a.strip()]
        phrases    = [strip_accents(p).lower().strip() for p in str(r.get("phrase_aliases","" )).split(";") if p.strip()]
        category   = strip_accents(str(r.get("category",""))).lower().strip()
        if not group_id or not canonical:
            continue
        # include canonical as an alias of itself to detect presence
        if canonical not in aliases:
            aliases.append(canonical)
        forbid = set(
            strip_accents(x).lower().strip()
            for x in str(r.get("forbid_with","")).split(";")
            if x.strip()
        )
        rows.append(SemanticRow(group_id, canonical, aliases, phrases, category, forbid))

    # build helpers
    alias_to_row = {}
    # list[(compiled_regex, row)]
    phrase_patterns = []  
    # allows up to PHRASE_MAX_GAP extra tokens
    phrase_gapped_patterns = []

    for row in rows:
        for a in row.aliases:
            # exact word alias match
            pat = re.compile(rf"\b{re.escape(a)}\b", re.IGNORECASE)
            alias_to_row[a] = (pat, row)
        for ptxt in sorted(row.phrases, key=len, reverse=True):
            # phrase match with word boundaries; collapse internal spaces in input first
            ptxt_norm = re.sub(r"\s+", " ", ptxt)
            # contiguous
            pat_contig = re.compile(rf"\b{re.escape(ptxt_norm)}\b", re.IGNORECASE)
            phrase_patterns.append((pat_contig, row))
            # gapped (only helpful if phrase has >= 2 tokens)
            if len(re.split(r"[-\s]+", ptxt_norm)) >= 2 and PHRASE_MAX_GAP >= 1:
                pat_gapped = hyphen_space_gapped_pattern(ptxt_norm, PHRASE_MAX_GAP)
                phrase_gapped_patterns.append((pat_gapped, row))

    rows_by_id = {row.group_id: row for row in rows}
    return {
        "rows": rows,
        "rows_by_id": rows_by_id,
        "alias_to_row": alias_to_row,
        "phrase_patterns": phrase_patterns,
        "phrase_gapped_patterns": phrase_gapped_patterns    
    }

def concept_gate_ok(prod_con: set, img_con: set) -> bool:
    # require at least one shared concept if product has any
    if SEMANTIC_REQUIRE_SHARED_CONCEPT and prod_con:
        if len(prod_con & img_con) < 1:
            return False

    if not SEMANTICS:
        return True

    rows_by_id = SEMANTICS.get("rows_by_id", {})

    # forbid_with (both directions)
    for gid in prod_con:
        row = rows_by_id.get(gid)
        if row and (row.forbid & img_con):
            return False
    for gid in img_con:
        row = rows_by_id.get(gid)
        if row and (row.forbid & prod_con):
            return False

    # helpers: filter concepts by critical categories
    def crit_set(gids: set) -> set:
        out = set()
        for g in gids:
            r = rows_by_id.get(g)
            if r and r.category in CRITICAL_CATEGORIES:
                out.add(g)
        return out

    prod_crit = crit_set(prod_con)
    img_crit  = crit_set(img_con)

    # symmetric rule: if either side mentions any critical concept,
    # both sides must share at least one SAME critical concept
    if STRICT_SYMMETRIC_CRITICAL:
        if prod_crit or img_crit:
            if not (prod_crit and img_crit and (prod_crit & img_crit)):
                return False

    # if product has multiple critical concepts, require ALL of them on image
    if REQUIRE_ALL_PRODUCT_CRITICAL and prod_crit:
        if not prod_crit.issubset(img_crit):
            return False

    return True

if SEMANTIC_DICT_PATH:
    SEMANTICS = load_semantic_dict(SEMANTIC_DICT_PATH)

# apply semantics: replace phrases/aliases with canonicals and collect concepts

def apply_semantics(text: str):
    if not SEMANTICS:
        t = strip_accents(normalize_unicode(text)).lower()
        return t, set()

    t = strip_accents(normalize_unicode(text)).lower()
    concepts = set()

    # 1) contiguous phrase replacement (as you already do)
    for pat, row in SEMANTICS["phrase_patterns"]:
        if pat.search(t):
            t = pat.sub(" " + row.canonical + " ", t)
            concepts.add(row.group_id)

    # 2) gapped phrase detection: if matched, append canonical token once
    for pat, row in SEMANTICS.get("phrase_gapped_patterns", []):
        if row.group_id in concepts:
            continue  # already tagged via contiguous
        if pat.search(t):
            concepts.add(row.group_id)
            t += " " + row.canonical + " "

    # 3) aliases (single-word) as before
    for a, (pat, row) in SEMANTICS["alias_to_row"].items():
        if pat.search(t):
            t = pat.sub(row.canonical, t)
            concepts.add(row.group_id)

    # 4) direct canonical presence
    for row in SEMANTICS["rows"]:
        if re.search(rf"\b{re.escape(row.canonical)}\b", t):
            concepts.add(row.group_id)

    t = re.sub(r"\s+", " ", t).strip()
    return t, concepts

# tokenization (semantic-aware) 
def basic_tokens_and_concepts(text: str):

    # promo scrub > semantics > symbol cleanup > tokens, return tokens + concepts

    t = remove_promos(text)
    t, concepts = apply_semantics(t)
    t = symbol_cleanup_keep_decimals(t)
    tokens = re.split(r"\s+", t)
    tokens = [tok for tok in tokens if tok and tok not in STOPWORDS]
    return tokens, concepts


def normalise(text: str) -> str:
    tokens, _concepts = basic_tokens_and_concepts(text)
    return " ".join(tokens)


def tokenize(s: str) -> list:
    tokens, _concepts = basic_tokens_and_concepts(s)
    return tokens


def concepts_from_text(s: str) -> set:
    _toks, conc = basic_tokens_and_concepts(s)
    return conc

# quantities: extract & compare 
def extract_quantities(text: str):

    # returns vol_ml_list, mass_g_list from text / floats
    # accepts decimals with dot or comma, optional space before unit - units: ml, l, g
    
    t = strip_accents(normalize_unicode(text)).lower()
    # normalize decimal comma to dot
    t = re.sub(r'(?<=\d),(?=\d)', '.', t)

    vol_ml = []
    mass_g = []

    pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(ml|l|g)\b', flags=re.IGNORECASE)
    for m in pattern.finditer(t):
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == 'l':
            vol_ml.append(val * 1000.0)
        elif unit == 'ml':
            vol_ml.append(val)
        elif unit == 'g':
            mass_g.append(val)

    # deduplicate with small rounding to avoid 0.330 vs 0.33 noise
    def dedup(nums):
        return sorted({round(x, 3) for x in nums})

    return dedup(vol_ml), dedup(mass_g)


def quantities_compatible(q1, q2, tol=0.5):
    
    # if both sides specify quantity, require near-equality within tolerance
    # if one side doesn't have that type, don't constrain
    
    v1, g1 = q1
    v2, g2 = q2

    def lists_match(a, b, tol):
        if not a or not b:
            return True
        if len(a) != len(b):
            return False
        for x, y in zip(sorted(a), sorted(b)):
            if abs(x - y) > tol:
                return False
        return True

    return lists_match(v1, v2, tol) and lists_match(g1, g2, tol)

# token filters for prefix + anchor 
def alpha_tokens_minlen(text: str, min_len: int) -> list:
    # alphabetic-only tokens / no digits, with length >= min_len, NOT stopwords
    toks = tokenize(text)
    out = []
    for t in toks:
        if len(t) < min_len:
            continue
        if not t.isalpha():
            continue
        if t in STOPWORDS:
            continue
        if t in {"ml", "g", "l"}:
            continue
        out.append(t)
    return out

# STARTSWITH similarity that ignores short / non-alpha tokens entirely
def startswith_overlap_count(item_str, image_str):
    item_tokens = set(alpha_tokens_minlen(item_str, MIN_PREFIX_LEN))
    image_tokens = set(alpha_tokens_minlen(image_str, MIN_PREFIX_LEN))
    overlap = 0
    for it in item_tokens:
        for im in image_tokens:
            if it.startswith(im) or im.startswith(it):
                overlap += 1
    return overlap

# first-word (anchor) discovery from left-to-right order - fallback only
def first_anchor_word(text: str, min_len: int) -> str | None:
    ordered = symbol_cleanup_keep_decimals(remove_promos(text))
    for tok in ordered.split():
        if len(tok) < min_len:
            continue
        if not tok.isalpha():
            continue
        if tok in STOPWORDS:
            continue
        if tok in {"ml", "g", "l"}:
            continue
        return tok
    return None

# does other_text contain a token similar to anchor above a threshold?? - fallback
def anchor_match_ok(anchor: str | None, other_text: str, threshold: int) -> bool:
    if not anchor:
        return True
    candidates = alpha_tokens_minlen(other_text, ANCHOR_MIN_LEN)
    if not candidates:
        return False
    best = max((fuzz.ratio(anchor, c) for c in candidates), default=0)
    return best >= threshold

# safe filename for windows (keep readable spaces)
def safe_filename(name: str) -> str:
    s = strip_accents(normalize_unicode(name)).strip()
    s = re.sub(r'[<>:"/\\|?*\n\r\t]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:180] if len(s) > 180 else s

def _count_images(folder: str) -> int:
    cnt = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                cnt += 1
    return cnt

# core 
def find_best_matches(csv_path, images_folder, output_folder):
    
    df = pd.read_csv(csv_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    df_len = len(df)
    img_count = _count_images(images_folder) 

    pbar = tqdm(
        total=img_count + df_len,
        desc="Indexing images",
        unit="step",
        ncols=80,
        bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        leave=False,
    )

    #  index images , store normalized key and semantic concepts
    image_index = {}           # norm_key -> full_path
    image_rawbase = {}         # norm_key -> raw base filename (no ext)
    image_qty = {}             # norm_key -> (vol_ml_list, mass_g_list)
    image_concepts = {}        # norm_key -> set of concept group_ids

    for rootdir, _, files in os.walk(images_folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                base, ext = os.path.splitext(f)
                norm_key = normalise(base)
                full_path = os.path.join(rootdir, f)
                # keep first occurrence
                if norm_key not in image_index:  
                    image_index[norm_key] = full_path
                    image_rawbase[norm_key] = base
                    image_qty[norm_key] = extract_quantities(base)
                    image_concepts[norm_key] = concepts_from_text(base)
                if pbar:
                    # one step per image indexed
                    pbar.update(1)             

    # first pass fuzzy matching 
    if pbar:
        pbar.set_description("Fuzzy matching")

    # first pass: fuzzy matching with quantity + concept/anchor gating
    # (product_name, item_id, matched_image, match_score, match_method, quantity_ok, concept_ok)
    matches = []  
    matched_item_ids = set()
    # image_key -> usage count
    matched_image_usage = {}  

    # for later rename join: product_name, item_id, copied_name, ext
    review_rows = []  

    for _, row in df.iterrows():
        product_name = str(row["name"])
        item_id = str(row["item_id"])

        product_tokens = tokenize(product_name)
        # threshold adapts to length
        if len(product_tokens) <= 2:
            thresh = 80
        elif len(product_tokens) <= 4:
            thresh = 70
        else:
            thresh = 60

        prod_concepts = concepts_from_text(product_name)

        # fuzzy candidates
        top_matches = process.extract(
            normalise(product_name),
            image_index.keys(),
            scorer=fuzz.token_set_ratio,
            limit=3
        )

        if not top_matches:
            matches.append((product_name, item_id, None, 0, "NO_MATCH", False, False))
            continue

        best_score = top_matches[0][1]
        tied = [m for m in top_matches if m[1] == best_score]

        norm = normalise(product_name)

        def tie_cmp(a, b):
            key_a, key_b = a[0], b[0]
            len_diff = abs(len(norm) - len(key_a)) - abs(len(norm) - len(key_b))
            if len_diff != 0:
                return len_diff
            return -1 if key_a < key_b else 1 if key_a > key_b else 0

        tied.sort(key=cmp_to_key(tie_cmp))
        best_key, final_score, _ = tied[0]

        # gates
        qty_ok = quantities_compatible(extract_quantities(product_name), image_qty.get(best_key, ([], [])))

        # concept gate: if product has any concepts and flag enabled > require overlap >=1
        img_con = image_concepts.get(best_key, set())
        concept_ok = concept_gate_ok(prod_concepts, img_con)
        # if no concepts on product, fall back to your anchor rule
        if not prod_concepts:
            anchor = first_anchor_word(product_name, ANCHOR_MIN_LEN)
            concept_ok = anchor_match_ok(anchor, image_rawbase.get(best_key, ""), ANCHOR_RATIO_THRESHOLD)

        if final_score >= thresh and qty_ok and concept_ok and matched_image_usage.get(best_key, 0) < MAX_IMAGE_USAGE:
            src_path = image_index[best_key]
            _, ext = os.path.splitext(src_path)

            out_name = safe_filename(product_name) or f"item_{item_id}"
            candidate = out_name + ext.lower()
            i = 2
            while os.path.exists(os.path.join(output_folder, candidate)):
                candidate = f"{out_name} ({i}){ext.lower()}"
                i += 1

            target_path = os.path.join(output_folder, candidate)
            shutil.copyfile(src_path, target_path)

            matches.append((product_name, item_id, src_path, final_score, "FUZZY", True, concept_ok))
            matched_item_ids.add(item_id)
            matched_image_usage[best_key] = matched_image_usage.get(best_key, 0) + 1
            review_rows.append((product_name, item_id, candidate, ext.lower()))
        else:
            matches.append((product_name, item_id, None, 0, "NO_MATCH", qty_ok, concept_ok))
        if pbar:
            pbar.update(1)


    # second pass: prefix-overlap with quantity + concept/anchor gating
    unmatched_items = df[~df["item_id"].isin(matched_item_ids)].copy()
    
    # extend total for second pass
    if pbar:
        pbar.total += len(unmatched_items)
        pbar.set_description("Prefix matching")
        pbar.refresh()

    # available images (respect usage limit)
    available = [(k, v) for k, v in image_index.items() if matched_image_usage.get(k, 0) < MAX_IMAGE_USAGE]

    def concept_or_anchor_ok(prod_con: set, img_key: str, prod_name: str) -> bool:
        img_con = image_concepts.get(img_key, set())
        ok = concept_gate_ok(prod_con, img_con)
        if prod_con:
            return ok
        # fallback if no concepts on product
        anchor = first_anchor_word(prod_name, ANCHOR_MIN_LEN)
        return ok and anchor_match_ok(anchor, image_rawbase.get(img_key, ""), ANCHOR_RATIO_THRESHOLD)

    def try_find_with_level(product_name: str, level: int):
        for key, full in available:
            c = startswith_overlap_count(product_name, key)
            if c >= level:
                if not quantities_compatible(extract_quantities(product_name), image_qty.get(key, ([], []))):
                    continue
                if not concept_or_anchor_ok(prod_con, key, product_name):
                    continue
                return key, full
        return None, None

    for _, row in unmatched_items.iterrows():
        product_name = str(row["name"])
        item_id = str(row["item_id"])
        prod_con = concepts_from_text(product_name) 

        found_key, found = None, None

        # step 1: overlap >=3
        found_key, found = try_find_with_level(product_name, 3)
        # step 2: overlap >=2
        if not found:
            found_key, found = try_find_with_level(product_name, 2)
        # step 3: overlap >=1 (still respects MIN_PREFIX_LEN and concept/anchor gate)
        if not found:
            found_key, found = try_find_with_level(product_name, 1)

        if found and found_key:
            src_path = found
            _, ext = os.path.splitext(src_path)

            out_name = safe_filename(product_name) or f"item_{item_id}"
            candidate = out_name + ext.lower()
            i = 2
            while os.path.exists(os.path.join(output_folder, candidate)):
                candidate = f"{out_name} ({i}){ext.lower()}"
                i += 1
            target_path = os.path.join(output_folder, candidate)
            shutil.copyfile(src_path, target_path)

            matched_image_usage[found_key] = matched_image_usage.get(found_key, 0) + 1

            method_lvl = 3 if startswith_overlap_count(product_name, found_key) >= 3 else (2 if startswith_overlap_count(product_name, found_key) >= 2 else 1)
            matches.append((product_name, item_id, src_path, 100, f"STARTSWITH_{method_lvl}", True, True))
            review_rows.append((product_name, item_id, candidate, ext.lower()))
        else:
            matches.append((product_name, item_id, None, 0, "NO_MATCH", False, False))
        if pbar:
            pbar.update(1)
    
    # write summary + review index
    if pbar:
        pbar.close()

    matches_df = pd.DataFrame(
        matches,
        columns=["product_name", "item_id", "matched_image", "match_score", "match_method", "quantity_ok", "concept_ok"]
    )
    summary_file = os.path.join(output_folder, "match_summary.csv")
    matches_df.to_csv(summary_file, index=False)

    review_df = pd.DataFrame(review_rows, columns=["product_name", "item_id", "copied_filename", "ext"])
    review_index_file = os.path.join(output_folder, "review_index.csv")
    review_df.to_csv(review_index_file, index=False)

    print(f"Matching complete! Check '{output_folder}'.")
    print("Detailed match summary:", summary_file)
    print("Review mapping:", review_index_file)

    #ask to finalize rename to item_id
    if review_rows:
        if messagebox.askyesno("Finalize?", "Finished manual review?\nRename kept files to item_id now?"):
            finalize_rename(output_folder, review_index_file)
            messagebox.showinfo("Done", "Files renamed to item_id.")
        else:
            messagebox.showinfo("Later", "You can run finalize_rename() later from code, or rerun script.")


def finalize_rename(output_folder: str, review_index_csv: str):
    
    # renaming files currently in output_folder from <safe product name>.* to <item_id>.<ext>
    # using the review_index created during matching. only renames files that still exist
    # allowing for manual review/check & deletion of wrong results
    
    idx = pd.read_csv(review_index_csv)
    renamed = 0
    for _, r in idx.iterrows():
        copied = str(r["copied_filename"])
        item_id = str(r["item_id"])
        ext = str(r["ext"])

        src = os.path.join(output_folder, copied)
        if os.path.exists(src):
            dst = os.path.join(output_folder, f"{item_id}{ext}")
            if os.path.exists(dst):
                base_dst = os.path.join(output_folder, f"{item_id}")
                i = 2
                candidate = f"{base_dst} ({i}){ext}"
                while os.path.exists(candidate):
                    i += 1
                    candidate = f"{base_dst} ({i}){ext}"
                dst = candidate
            os.replace(src, dst)
            renamed += 1
    print(f"Final rename completed. Renamed {renamed} files.")


def main():
    find_best_matches(csv_path, images_folder, output_folder)


if __name__ == "__main__":
    main()
