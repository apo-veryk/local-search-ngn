# Local CSV-Image Fuzzy Matching Search Engine

A Python-based project that helps you match a list of items (from a CSV) to images in a folder using [RapidFuzz](https://github.com/maxbachmann/RapidFuzz).  
It provides a **Tkinter** GUI for file/folder selection, then applies fuzzy-matching algorithms to find the best matching image for each item name in the CSV.

## Features
- Cross-platform file dialogs for selecting CSV and image folders.
- Fuzzy-matching using RapidFuzz (`token_set_ratio` and `token_sort_ratio`) to handle complex product name variations.
- Automatic folder creation if the output directory doesn’t exist.
- Detailed `match_summary.csv` that reports how each CSV entry was matched.

## Prerequisites
- Python 3.10+ (or higher)
- `pandas`
- `rapidfuzz`

## Installation

1. **Clone** or **download** this repository:
   ```bash
   git clone https://github.com/<your-username>/my-local-search-engine.git

2. Navigate to the project folder:
   ```bash
   cd my-local-search-engine

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
1. Run the script:
   ```bash
   python local_search_engine.py
2. A terminal message will prompt you to select your CSV file (defaults to your “Downloads” folder).
3. Next, you’ll be prompted for your images folder (defaults to your home directory).
4. Finally, you’ll be prompted to select (or create) an output folder.

Once the script completes:
- It copies the best-matched images into the output folder, renaming them according to the CSV’s `id` column.
- Creates a `match_summary.csv` in the same output folder, showing product_name, item_id, matched_image, and match_score.

## CSV Format Expectations
- The CSV file should have at least two columns:
   - `name` (the product or item name)
   - `item_id` (the unique identifier for that item)

- Example CSV snippet:
| item_id | name                           |
|---------|--------------------------------|
| 123     | Espresso Single Lungo          |
| 456     | Dark Roast Espresso Single Cup |

## Image Preparation
- Place the images in one folder.
- Names should be somewhat descriptive so that fuzzy matching can succeed (e.g., `Espresso Single Lungo.jpg`).
- The script will try to handle small variations by ignoring order, extra words, etc., but more accurate naming leads to better matches.

## License
This project is distributed under the MIT License. Feel free to modify, share, and adapt it to your needs!