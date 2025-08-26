# Sales Pipeline Analyzer 

A Streamlit app to analyze sales pipeline data with a clean, professional Telkomsel-red theme across charts and tables.

## Features
- Chap 1: CV by segment/quarter, top/bottom deals, open pipeline, pivot summaries
- Chap 2: Sales cycle time, stage→close durations, SE view, win rate, Top AM views
- Industry Segment normalization to Lines of Business (LoB) via mapping
- Minimal, readable table styling with subtle accents (no gradients)

## Project Structure
- `app.py` — Streamlit UI (file upload, column mapping, filters, rendering)
- `processing.py` — Data preprocessing and analysis logic (plots + tables)
- `requirements.txt` — Python dependencies

## Setup
1. Python 3.9+
2. Create/activate a virtual environment (optional but recommended)
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App
```bash
streamlit run app.py
```
Then open the local URL printed in the terminal.

## Usage
1. Upload an Excel `.xlsx` file (first sheet is used).
2. Review detected columns and map to canonical columns if needed.
3. Optionally filter by Industry Segment (LoB) in the sidebar.
4. Run Bagian 1 and Bagian 2 analyses; explore charts and styled tables.

## Data Expectations (Key Columns)
The app auto-maps common aliases to canonical names. Helpful columns include:
- Opportunity/Account: `Opportunity Name`, `Account Name`, `AM Name`, `Opportunity Owner`
- Amounts & Dates: `Schedule Amount`, `Schedule Date`, `Created Date`, `Close Date`, `Last Stage Change Date`
- Classification: `Stage`, `Industry Segment`, `Pilar`

If a dependency is missing, the app skips that output and shows a warning.

## Theming
- Primary accent: Telkomsel red `#e60000`
- Neutrals for comparison bars and table borders
- Minimalist, consistent styling for clarity and brand alignment

## Notes
- Large files may take time to process. Data is cached during a session for faster iteration.
- For best results, ensure date columns are valid Excel/ISO datetimes.

## License
Internal/Private use. Update as needed.
