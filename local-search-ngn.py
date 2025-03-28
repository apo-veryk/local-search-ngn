import os
import pandas as pd
from rapidfuzz import process, fuzz
import shutil
import re
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

root = tk.Tk()
root.withdraw()  

download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
while True:
    print("Select CSV FILE!")
    csv_path = filedialog.askopenfilename(
        initialdir=download_dir,
        title="select your CSV file",
        filetypes=[("CSV Files", "*.csv")]
    )
    if csv_path:
        break  # Valid CSV file selected, exit loop
    else:
        quit_confirm = messagebox.askyesno("Quit?", "no CSV file selected. wanna quit?")
        if quit_confirm:
            exit() 

home_dir = os.path.expanduser("~")
print("Select IMAGES FOLDER!")
images_folder = filedialog.askdirectory(
    initialdir=home_dir,
    title="select your Images Folder"
)

while True:
    output_folder_name = simpledialog.askstring(
        "Output Folder Name",
        "Enter the output folder name (NO SPACES):"
    )
    if output_folder_name:
        # Check for spaces or invalid characters
        invalid_chars = set(r'\/:*?"<>|')
        if " " in output_folder_name or any(char in invalid_chars for char in output_folder_name):
            messagebox.showerror(
                "Invalid Folder Name",
                "invalid characters detected!!!\ndo NOT include spaces or any of these characters:\n/ \\ : * ? \" < > |\n\nre-enter a valid folder name"
            )
            continue  # Loop back for re-entry
        else:
            break  # Valid folder name provided
    else:
        quit_confirm = messagebox.askyesno("Quit?", "no folder name entered. wanna quit?")
        if quit_confirm:
            exit()  # User confirmed exit

base_path = os.path.join(home_dir, "generix-photos", "0.generix-search-results")
output_folder = os.path.join(base_path, output_folder_name)

# Create the directory (including any missing intermediate folders)
os.makedirs(output_folder, exist_ok=True)

messagebox.showinfo("Folder Created", f"Output folder created at:\n{output_folder}")

# Print out what was selected for debugging/confirmation
print("CSV file selected:", csv_path)
print("Images folder selected:", images_folder)
print("Output folder selected or created:", output_folder)

def tokenize(s):
    """
    Splits a string on spaces and dots (and can easily be extended).
    Returns lowercase tokens with whitespace trimmed.
    """
    return [t.strip() for t in re.split(r'[ .]+', s.lower()) if t.strip()]

def startswith_overlap_count(item_str, image_str):
    """
    Returns the number of token pairs (it, im) where either it.startswith(im)
    or im.startswith(it). Using sets to avoid repeatedly matching the same token
    multiple times if it appears more than once (feel free to adjust).
    """
    item_tokens = set(tokenize(item_str))
    image_tokens = set(tokenize(image_str))

    overlap = 0
    for it in item_tokens:
        for im in image_tokens:
            if it.startswith(im) or im.startswith(it):
                overlap += 1
    return overlap

def find_best_matches(csv_path, images_folder, output_folder):
    # === 1) Read CSV ===
    df = pd.read_csv(csv_path)

    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # === 2) Gather all files in images_folder ===
    # Map from name_without_ext -> actual_filename
    image_files = [
        f for f in os.listdir(images_folder)
        if os.path.isfile(os.path.join(images_folder, f))
    ]
    image_names = {os.path.splitext(f)[0]: f for f in image_files}

    # === 3) First pass: fuzzy matching ===
    matches = []  # (product_name, item_id, matched_image, match_score, match_method)
    matched_item_ids = set()
    matched_image_keys = set()

    for _, row in df.iterrows():
        product_name = row["name"]
        item_id = row["item_id"]  # Adjust if your CSV uses a different column name

        # Find up to the top 3 fuzzy matches using token_set_ratio
        top_matches = process.extract(
            product_name,
            image_names.keys(),
            scorer=fuzz.token_set_ratio,
            limit=3
        )

        # If no match found at all
        if not top_matches:
            matches.append((product_name, item_id, None, 0, "NO_MATCH"))
            continue

        # The highest token_set_ratio score
        best_score = top_matches[0][1]
        # Gather all that share the same best score
        tied_candidates = [m for m in top_matches if m[1] == best_score]

        if len(tied_candidates) == 1:
            # No tie, straightforward best match
            best_match_key, final_score, _ = tied_candidates[0]
        else:
            # We have multiple candidates with the same top score.
            # Break the tie using token_sort_ratio
            best_tiebreak_score = -1
            best_tiebreak_key = None
            for candidate_key, _, _ in tied_candidates:
                # Evaluate token_sort_ratio for each candidate
                tiebreak_score = fuzz.token_sort_ratio(product_name, candidate_key)
                if tiebreak_score > best_tiebreak_score:
                    best_tiebreak_score = tiebreak_score
                    best_tiebreak_key = candidate_key

            best_match_key = best_tiebreak_key
            final_score = best_score

        # Check against your match threshold
        if final_score > 70:
            best_image_name = image_names[best_match_key]
            source_path = os.path.join(images_folder, best_image_name)
            target_path = os.path.join(output_folder, f"{item_id}.jpg")

            shutil.copyfile(source_path, target_path)

            matches.append((product_name, item_id, best_image_name, final_score, "FUZZY"))
            matched_item_ids.add(item_id)
            matched_image_keys.add(best_match_key)
        else:
            matches.append((product_name, item_id, None, 0, "NO_MATCH"))

    # === 4) Identify unmatched items and unmatched images for the second pass ===
    unmatched_items = df[~df["item_id"].isin(matched_item_ids)].copy()
    unmatched_image_names = {
        k: v for k, v in image_names.items() if k not in matched_image_keys
    }

    # We'll store second-pass matches here, then merge with the main matches list.
    second_pass_matches = []
    
    # Convert unmatched_image_names dict to a (key, filename) list so we can remove used images.
    unmatched_images_list = list(unmatched_image_names.items())

    # === 5) Second pass: multi-step "startswith" matching ===
    for _, row in unmatched_items.iterrows():
        product_name = row["name"]
        item_id = row["item_id"]

        found_match = None
        found_overlap_level = 0
        matched_index = None

        # --- Step 1: Look for overlap >= 3 ---
        for i, (image_key, actual_image_file) in enumerate(unmatched_images_list):
            c = startswith_overlap_count(product_name, image_key)
            if c >= 3:
                found_match = actual_image_file
                found_overlap_level = 3
                matched_index = i
                break

        # --- Step 2: If still no match, look for overlap >= 2 ---
        if not found_match:
            for i, (image_key, actual_image_file) in enumerate(unmatched_images_list):
                c = startswith_overlap_count(product_name, image_key)
                if c >= 2:
                    found_match = actual_image_file
                    found_overlap_level = 2
                    matched_index = i
                    break

        # --- Step 3: If still no match, look for overlap >= 1 ---
        if not found_match:
            for i, (image_key, actual_image_file) in enumerate(unmatched_images_list):
                c = startswith_overlap_count(product_name, image_key)
                if c >= 1:
                    found_match = actual_image_file
                    found_overlap_level = 1
                    matched_index = i
                    break

        # If we have a match, copy the file and remove from unmatched_images_list
        if found_match:
            source_path = os.path.join(images_folder, found_match)
            target_path = os.path.join(output_folder, f"{item_id}.jpg")
            shutil.copyfile(source_path, target_path)

            # Remove matched image from list so it's not used again
            unmatched_images_list.pop(matched_index)

            second_pass_matches.append((
                product_name,
                item_id,
                found_match,
                100,  # any score you like, or found_overlap_level
                f"STARTSWITH_{found_overlap_level}"
            ))
        else:
            # No match found in second pass
            second_pass_matches.append((product_name, item_id, None, 0, "NO_MATCH"))

    # === 6) Combine both sets of matches and write out final summary CSV ===
    matches_df = pd.DataFrame(
        matches + second_pass_matches,
        columns=["product_name", "item_id", "matched_image", "match_score", "match_method"]
    )

    summary_file = os.path.join(output_folder, "match_summary.csv")
    matches_df.to_csv(summary_file, index=False)

    print(f"Matching complete! Check the folder '{output_folder}' for results.")
    print("Detailed match summary written to:", summary_file)

def main():
    find_best_matches(csv_path, images_folder, output_folder)

if __name__ == "__main__":
    main()
