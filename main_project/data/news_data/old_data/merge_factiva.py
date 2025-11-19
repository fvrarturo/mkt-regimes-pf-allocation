import pandas as pd
import re
from datetime import datetime
from pathlib import Path
import glob

def parse_date_and_author(date_str):
    """
    Parse the date column to extract:
    1. Date/time as timestamp
    2. Author if "By..." pattern exists
    """
    if pd.isna(date_str) or date_str == '':
        return None, None
    
    date_str = str(date_str)
    author = None
    
    # Extract author if "By..." pattern exists
    author_match = re.search(r'By\s+([^,]+)', date_str)
    if author_match:
        author = author_match.group(1).strip()
    
    # Try to parse date - handle two formats:
    # Format 1: "2:53 PM GMT, 19 November 2025" (newer format with time)
    # Format 2: "20 October 1992" (older format without time)
    
    # Check if string contains GMT (indicates time format)
    has_gmt = 'GMT' in date_str
    
    # Try newer format first (with time and GMT)
    if has_gmt:
        time_date_match = re.search(r'(\d{1,2}:\d{2}\s+(?:AM|PM)\s+GMT),\s+(\d{1,2}\s+\w+\s+\d{4})', date_str)
        if time_date_match:
            time_str = time_date_match.group(1)  # e.g., "2:53 PM GMT"
            date_str_only = time_date_match.group(2)  # e.g., "19 November 2025"
            try:
                # Parse time (remove GMT)
                time_part = time_str.replace(' GMT', '').strip()
                # Parse date
                date_obj = datetime.strptime(date_str_only, '%d %B %Y')
                # Parse time
                time_obj = datetime.strptime(time_part, '%I:%M %p').time()
                # Combine
                timestamp = datetime.combine(date_obj.date(), time_obj)
                return timestamp, author
            except Exception as e:
                # If time parsing fails but we have GMT, still try to get the date
                # but log the error for debugging
                try:
                    date_obj = datetime.strptime(date_str_only, '%d %B %Y')
                    return date_obj, author
                except:
                    pass
    
    # Try older format (date only) - only if no GMT was found
    if not has_gmt:
        date_only_match = re.search(r'(\d{1,2}\s+\w+\s+\d{4})', date_str)
        if date_only_match:
            date_str_only = date_only_match.group(1)
            try:
                timestamp = datetime.strptime(date_str_only, '%d %B %Y')
                return timestamp, author
            except:
                pass
    
    # If no match, return None
    return None, author

def merge_factiva_files():
    """
    Merge all factiva CSV files into full_factiva.csv
    """
    # Find all factiva CSV files
    factiva_files = sorted(glob.glob('factiva_*.csv'))
    
    print(f"Found {len(factiva_files)} factiva files")
    
    all_data = []
    
    for file_path in factiva_files:
        print(f"Processing {file_path}...")
        try:
            # Try reading with header first
            df = pd.read_csv(file_path)
            
            # Check if 'date' column exists, if not, assume no header
            if 'date' not in df.columns:
                # Read again without header and assign column names
                df = pd.read_csv(file_path, header=None)
                if len(df.columns) >= 3:
                    df.columns = ['headline', 'date', 'snippet'] + [f'col_{i}' for i in range(3, len(df.columns))]
                else:
                    print(f"  Warning: {file_path} has unexpected structure, skipping")
                    continue
            
            # Process each row
            processed_rows = []
            for idx, row in df.iterrows():
                date_ts, author = parse_date_and_author(row['date'])
                
                processed_row = {
                    'date': date_ts,
                    'headline': row['headline'],
                    'author': author if author else None,
                    'snippet': row['snippet']
                }
                processed_rows.append(processed_row)
            
            processed_df = pd.DataFrame(processed_rows)
            all_data.append(processed_df)
            print(f"  Processed {len(processed_df)} rows")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all dataframes
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        
        # Drop duplicates based on all columns
        initial_count = len(full_df)
        full_df = full_df.drop_duplicates()
        duplicates_removed = initial_count - len(full_df)
        
        # Sort by date
        full_df = full_df.sort_values('date', na_position='last')
        
        # Save to CSV
        output_file = 'full_factiva.csv'
        full_df.to_csv(output_file, index=False)
        print(f"\nMerged {initial_count} rows")
        print(f"Removed {duplicates_removed} duplicate rows")
        print(f"Final count: {len(full_df)} rows saved to {output_file}")
        print(f"Columns: {list(full_df.columns)}")
        print(f"\nDate range: {full_df['date'].min()} to {full_df['date'].max()}")
        print(f"Rows with authors: {full_df['author'].notna().sum()}")
        print(f"Rows without authors: {full_df['author'].isna().sum()}")
        
        # Show time parsing statistics
        full_df['date'] = pd.to_datetime(full_df['date'])
        midnight_count = (full_df['date'].dt.time == pd.Timestamp('00:00:00').time()).sum()
        with_times_count = len(full_df) - midnight_count
        print(f"\nTime parsing statistics:")
        print(f"  Entries with actual times: {with_times_count} ({with_times_count/len(full_df)*100:.1f}%)")
        print(f"  Entries at midnight (00:00:00): {midnight_count} ({midnight_count/len(full_df)*100:.1f}%)")
        print(f"    (Note: 00:00:00 entries are correct - they either have no time info in source or are actually midnight)")
    else:
        print("No data to merge!")

if __name__ == '__main__':
    merge_factiva_files()

