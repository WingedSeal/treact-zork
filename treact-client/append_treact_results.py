#!/usr/bin/env python3
"""
Script to append TREACT analysis results to the existing max_scores_analysis.csv
"""

import csv

def append_treact_to_max_scores():
    """Append TREACT results to the existing max_scores_analysis.csv file."""
    
    # Read the existing max_scores_analysis.csv
    existing_data = []
    with open('max_scores_analysis.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        existing_data = list(reader)
    
    # Read the TREACT analysis results
    treact_data = []
    with open('treact_analysis_results.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        treact_data = list(reader)
    
    # Append TREACT data to existing data
    existing_data.extend(treact_data)
    
    # Write the combined data back to max_scores_analysis.csv
    with open('max_scores_analysis.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(existing_data)
    
    print(f"Successfully appended {len(treact_data)} TREACT records to max_scores_analysis.csv")
    print(f"Total records in max_scores_analysis.csv: {len(existing_data)}")

if __name__ == "__main__":
    append_treact_to_max_scores()