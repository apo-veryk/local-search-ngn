import os
import pandas as pd
from rapidfuzz import process, fuzz
import shutil

# # Update these paths as needed:
# csv_path = r"C:\Users\annan\Downloads\[coffee-square-convenience-store]-[2025-03-10]-[15_38] - generix.csv"
# images_folder = r"C:\Users\annan\generic photos\coffeeee4\resized"
# output_folder = r"C:\Users\annan\generix-search\coffee-squarezz1"

import tkinter as tk
from tkinter import filedialog

# Initialize Tkinter
root = tk.Tk()
root.withdraw()  # Hides the small root window (we only want dialogs)

# 1) Ask user for CSV file, defaulting to user's Downloads folder
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
print("Select CVS FILE!")
csv_path = filedialog.askopenfilename(
    initialdir=download_dir,
    title="Select your CSV file",
    filetypes=[("CSV Files", "*.csv")]
)

# 2) Ask user for the images folder, defaulting to user's home directory
home_dir = os.path.expanduser("~")
print("Select IMAGES FOLDER!")
images_folder = filedialog.askdirectory(
    initialdir=home_dir,
    title="Select your Images Folder"
)

# 3) Ask user for (or create) the output folder, defaulting to user's home directory
print("TYPE OUTPUT FOLDER NAME!")
output_folder = filedialog.askdirectory(
    initialdir=home_dir,
    title="Type Output Folder name"
)
os.makedirs(output_folder, exist_ok=True)

# Print out what was selected for debugging/confirmation
print("CSV file selected:", csv_path)
print("Images folder selected:", images_folder)
print("Output folder selected or created:", output_folder)

def find_best_matches(csv_path, images_folder, output_folder=output_folder):
    df = pd.read_csv(csv_path)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Gather all files in images_folder, map from name_without_ext -> full_filename
    image_files = [
        f for f in os.listdir(images_folder)
        if os.path.isfile(os.path.join(images_folder, f))
    ]
    image_names = {os.path.splitext(f)[0]: f for f in image_files}

    matches = []  # To store (product_name, item_id, matched_image, match_score)

    for _, row in df.iterrows():
        product_name = row["name"]
        item_id = row["id"]   # Adjust if your CSV uses a different field name

        # Find up to the top 3 fuzzy matches using token_set_ratio
        top_matches = process.extract(
            product_name,
            image_names.keys(),
            scorer=fuzz.token_set_ratio,
            limit=3
        )

        # If no match found at all
        if not top_matches:
            matches.append((product_name, item_id, None, 0))
            continue

        # The highest token_set_ratio score
        best_score = top_matches[0][1]
        # Gather all that share that same best score
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
            final_score = best_score  # Keep the original top score from token_set_ratio

        # Check against your match threshold (85 in this example)
        if final_score > 70:
            best_image_name = image_names[best_match_key]
            source_path = os.path.join(images_folder, best_image_name)
            target_path = os.path.join(output_folder, f"{item_id}.jpg")

            shutil.copyfile(source_path, target_path)
            matches.append((product_name, item_id, best_image_name, final_score))
        else:
            matches.append((product_name, item_id, None, 0))

    # Write out a summary CSV of all match attempts
    matches_df = pd.DataFrame(matches, columns=["product_name", "item_id", "matched_image", "match_score"])
    summary_file = os.path.join(output_folder, "match_summary.csv")
    matches_df.to_csv(summary_file, index=False)

    print(f"Matching complete! Check the folder '{output_folder}' for results.")

def main():
    find_best_matches(csv_path, images_folder)

if __name__ == "__main__":
    main()
