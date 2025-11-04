#!/usr/bin/env python3
"""
Script to analyze TREACT-3-sets-New chat logs and generate metrics similar to max_scores_analysis.csv
"""

import os
import re
import csv
from pathlib import Path

def parse_chat_log(file_path):
    """Parse a single chat log file to extract metrics."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    lines = content.split('\n')
    
    # Initialize metrics
    max_score = 0
    moves_at_max_score = 0
    session_start_line = None
    max_score_line = None
    total_deaths = 0
    real_deaths = 0
    fake_deaths = 0
    real_death_lines = []
    fake_death_lines = []
    
    # Find session start (first game line)
    for i, line in enumerate(lines, 1):
        if 'West of House' in line and 'Score: 0' in line and 'Moves: 0' in line:
            session_start_line = i
            break
    
    # Patterns for death detection - very specific to player deaths only
    real_death_patterns = [
        r'^\s*\*\*\*\*\s*You have died\s*\*\*\*\*\s*$',  # Exact match for "****  You have died  ****"
        r'.*[Tt]he .* eats you.*',                        # Grue eats player
        r'.*You have been eaten.*',                       # Player eaten
        r'.*You are .* killed.*'                          # Player killed
    ]
    
    fake_death_patterns = [
        r'.*Land of the Dead.*',                          # Fake death location
        r'.*Land of the Living Dead.*'                    # Fake death location
    ]
    
    # Parse lines to extract scores and deaths
    for i, line in enumerate(lines, 1):
        # Look for score and move information
        score_match = re.search(r'Score:\s*(\d+).*Moves:\s*(\d+)', line)
        if score_match:
            current_score = int(score_match.group(1))
            current_moves = int(score_match.group(2))
            
            if current_score > max_score:
                max_score = current_score
                moves_at_max_score = current_moves
                max_score_line = i
        
        # Check for real deaths
        for pattern in real_death_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                total_deaths += 1
                real_deaths += 1
                real_death_lines.append(str(i))
                break
        else:
            # Check for fake deaths (only if not a real death)
            for pattern in fake_death_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    total_deaths += 1
                    fake_deaths += 1
                    fake_death_lines.append(str(i))
                    break
    
    return {
        'max_score': max_score,
        'moves_at_max_score': moves_at_max_score,
        'session_start_line': session_start_line if session_start_line else 8,  # default fallback
        'max_score_line': max_score_line if max_score_line else session_start_line,
        'total_deaths': total_deaths,
        'real_deaths': real_deaths,
        'fake_deaths': fake_deaths,
        'real_death_lines': ';'.join(real_death_lines) if real_death_lines else '',
        'fake_death_lines': ';'.join(fake_death_lines) if fake_death_lines else ''
    }

def analyze_treact_logs():
    """Main function to analyze all TREACT logs and generate CSV."""
    base_path = Path("c:/zork-tool/treact-client/logs/TREACT-3-sets-New")
    
    results = []
    
    # Process each set
    for set_name in ['set1', 'set2', 'set3']:
        set_path = base_path / set_name / "chat_log"
        
        if not set_path.exists():
            print(f"Warning: {set_path} does not exist")
            continue
        
        # Get all chat log files (excluding mcp-server logs)
        chat_files = sorted([f for f in set_path.glob("chat_log_*.txt")])
        
        print(f"Processing {set_name}: found {len(chat_files)} chat log files")
        
        for game_num, chat_file in enumerate(chat_files, 1):
            print(f"  Analyzing {chat_file.name}...")
            
            metrics = parse_chat_log(chat_file)
            
            if metrics:
                row = [
                    'TREACT-3-sets-New',  # Folder_Type
                    set_name,             # Set_Name
                    game_num,             # Game_Number
                    metrics['max_score'], # Max_Score
                    metrics['moves_at_max_score'], # Moves_at_Max_Score
                    metrics['session_start_line'], # Session_Start_Line
                    metrics['max_score_line'],     # Max_Score_Line
                    metrics['total_deaths'],       # Total_Deaths
                    metrics['real_deaths'],        # Real_Deaths
                    metrics['fake_deaths'],        # Fake_Deaths
                    metrics['real_death_lines'],   # Real_Death_Lines
                    metrics['fake_death_lines']    # Fake_Death_Lines
                ]
                results.append(row)
    
    # Write results to CSV
    output_file = "treact_analysis_results.csv"
    
    headers = [
        'Folder_Type', 'Set_Name', 'Game_Number', 'Max_Score', 'Moves_at_Max_Score',
        'Session_Start_Line', 'Max_Score_Line', 'Total_Deaths', 'Real_Deaths', 
        'Fake_Deaths', 'Real_Death_Lines', 'Fake_Death_Lines'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    
    print(f"\nAnalysis complete! Results saved to {output_file}")
    print(f"Total games analyzed: {len(results)}")
    
    # Print summary statistics
    total_real_deaths = sum(row[8] for row in results)
    total_fake_deaths = sum(row[9] for row in results)
    total_deaths = sum(row[7] for row in results)
    
    print(f"Summary:")
    print(f"  Total deaths: {total_deaths}")
    print(f"  Real deaths: {total_real_deaths}")
    print(f"  Fake deaths: {total_fake_deaths}")

if __name__ == "__main__":
    analyze_treact_logs()