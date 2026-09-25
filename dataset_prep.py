import pandas as pd
import os
from pathlib import Path

def prepare_real_dataset(input_csv: str):
    """
    Reads a real Agmarknet CSV (e.g. from Kaggle),
    maps it to our required schema, and saves it for ingestion.
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        print(f"Error: {input_csv} not found.")
        print("Please download the dataset from Kaggle and place it here.")
        return None
        
    print(f"Loading real dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Standardize column names (handle variations in Kaggle datasets)
    # Common Kaggle column names vs Our Schema
    col_mapping = {
        'state': 'State', 'State Name': 'State',
        'district': 'District', 'District Name': 'District',
        'market': 'Market', 'Market Name': 'Market',
        'commodity': 'Commodity', 'Commodity Name': 'Commodity',
        'variety': 'Variety', 'Variety Name': 'Variety',
        'arrival_date': 'Arrival_Date', 'Price Date': 'Arrival_Date', 'Date': 'Arrival_Date',
        'min_price': 'Min_Price', 'Min Price (Rs/Quintal)': 'Min_Price',
        'max_price': 'Max_Price', 'Max Price (Rs/Quintal)': 'Max_Price',
        'modal_price': 'Modal_Price', 'Modal Price (Rs/Quintal)': 'Modal_Price'
    }
    
    # Rename columns ignoring case where possible
    df.columns = [str(c).strip() for c in df.columns]
    
    # Lowercase mapping for robust matching
    lower_map = {k.lower(): v for k, v in col_mapping.items()}
    new_cols = []
    for c in df.columns:
        if c.lower() in lower_map:
            new_cols.append(lower_map[c.lower()])
        else:
            new_cols.append(c)
    df.columns = new_cols
    
    # Ensure all required columns exist
    required_cols = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Arrival_Date', 'Min_Price', 'Max_Price', 'Modal_Price']
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        print(f"Dataset is missing essential columns: {missing}")
        print("Available columns were:", df.columns.tolist())
        return None
        
    # 2. Add missing fields required by our RAG
    if 'Grade' not in df.columns:
        print("Adding default 'Grade A' column...")
        df['Grade'] = 'Grade A'
        
    if 'Arrivals_Tonnes' not in df.columns:
        import numpy as np
        print("Synthesizing missing 'Arrivals_Tonnes' volume data...")
        df['Arrivals_Tonnes'] = np.random.randint(10, 500, size=len(df))
        
    # 3. Clean and filter data
    print("Cleaning missing values and formatting dates...")
    df = df.dropna(subset=['Modal_Price', 'Market'])
    df['Min_Price'] = pd.to_numeric(df['Min_Price'], errors='coerce').fillna(0)
    df['Max_Price'] = pd.to_numeric(df['Max_Price'], errors='coerce').fillna(0)
    df['Modal_Price'] = pd.to_numeric(df['Modal_Price'], errors='coerce').fillna(0)
    
    # Filter out rows with 0 prices to avoid math errors
    df = df[df['Modal_Price'] > 0]
    
    # 4. Save to expected location
    out_dir = Path("topo_rag/data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "agmarknet_sample.csv"
    
    df = df[['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade', 'Arrival_Date', 'Min_Price', 'Max_Price', 'Modal_Price', 'Arrivals_Tonnes']]
    
    # Save a small subset to keep indexing fast (e.g., 5000 rows max for fast MVP execution)
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)
        
    df.to_csv(out_path, index=False)
    print(f"✓ Real dataset mapped and saved successfully: {out_path} ({len(df)} rows)")
    return str(out_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        prepare_real_dataset(sys.argv[1])
    else:
        print("Usage: python dataset_prep.py <path_to_downloaded_kaggle_csv>")
