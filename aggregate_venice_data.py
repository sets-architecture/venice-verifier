
import pandas as pd
import json
import os
import re
from urllib.parse import unquote
from difflib import get_close_matches

# ==========================================
# CONFIGURATION
# ==========================================
# Get the folder where THIS script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct full paths to the CSV files
FILES = {
    'date': os.path.join(BASE_DIR, 'date-columns-css.csv'),
    'archivist': os.path.join(BASE_DIR, 'archivist-mark---signature.csv'),
    'con_e': os.path.join(BASE_DIR, 'con-and-e.csv'),
    'brackets': os.path.join(BASE_DIR, 'curly-brackets-comment.csv'),
    'named': os.path.join(BASE_DIR, 'named-traveler-origin.csv'),
    'quondam': os.path.join(BASE_DIR, 'quondam-csv-2.csv')
}

# Italian Parsing Logic
ITALIAN_NUMBERS = {
    'un': 1, 'uno': 1, 'una': 1, 'due': 2, 'duo': 2, 'doi': 2,
    'tre': 3, 'tria': 3, 'quattro': 4, 'quatro': 4, 'cinque': 5,
    'sei': 6, 'sette': 7, 'otto': 8, 'nove': 9, 'dieci': 10,
    'undici': 11, 'dodici': 12
}

PERSON_KEYWORDS = [
    'moglie', 'marito', 'consorte', 'figlio', 'figli', 'figliulo', 
    'figliuli', 'figlioli', 'fratello', 'sorella', 'padre', 'madre', 
    'servo', 'servi', 'servitore', 'domestico', 'domestici', 
    'cameriera', 'famiglia', 'compagno', 'compagni', 'persona', 'persone'
]

# ==========================================
# HELPERS
# ==========================================

def clean_filename(path):
    """Extracts just the filename from gs:// paths"""
    if pd.isna(path): return "unknown"
    return os.path.basename(unquote(str(path)))

def parse_json_cell(cell_value):
    """Parses JSON string from CSV cell, returns list"""
    if pd.isna(cell_value) or cell_value == '': return []
    try:
        return json.loads(cell_value)
    except (json.JSONDecodeError, TypeError):
        return []

def get_side(x_val):
    """Determines column based on X coordinate (0-100)"""
    return 'Left' if x_val < 50 else 'Right'

def parse_people_count(text):
    """Parses 'tre figli' -> 3, 'moglie' -> 1"""
    if not text or not isinstance(text, str): return 0
    clean_text = text.lower().replace('.', ' ').replace(',', ' ').strip()
    words = clean_text.split()
    
    is_person = False
    for word in words:
        if word in PERSON_KEYWORDS:
            is_person = True
            break
        # Fuzzy match for typos
        if get_close_matches(word, PERSON_KEYWORDS, n=1, cutoff=0.8):
            is_person = True
            break
            
    if not is_person: return 0

    for w in words:
        if w in ITALIAN_NUMBERS: return ITALIAN_NUMBERS[w]
            
    return 1 # Default if person word found but no number

# ==========================================
# MAIN PROCESSING
# ==========================================

def process_all():
    dfs = []
    
    # ---------------------------------------------------------
    # 1. DATE & COLUMNS
    # ---------------------------------------------------------
    print("Processing Date & Columns...")
    df = pd.read_csv(FILES['date'])
    df['page_id'] = df['image'].apply(clean_filename)
    
    # Q1: Is there a date? (Y/N)
    df['Has_Date_Top'] = df['date'].apply(lambda x: 'Y' if pd.notna(x) and str(x).strip() != '' else 'N')
    # Q2: Year Value
    df = df.rename(columns={'date': 'Year_Value', 'columns': 'Two_Columns'})
    # Q3: Two Columns is mapped directly
    
    dfs.append(df[['page_id', 'Has_Date_Top', 'Year_Value', 'Two_Columns']])

    # ---------------------------------------------------------
    # 2. ARCHIVIST MARKS & SIGNATURES
    # ---------------------------------------------------------
    print("Processing Archivist & Signatures...")
    df = pd.read_csv(FILES['archivist'])
    df['page_id'] = df['image'].apply(clean_filename)
    
    stats = []
    for _, row in df.iterrows():
        items = parse_json_cell(row['label'])
        
        # Q4: Is signed?
        sigs = [x for x in items if 'Signature' in x.get('rectanglelabels', [])]
        is_signed = 'Y' if sigs else 'N'
        
        # Q5: Signature Location
        sig_loc = ""
        if sigs:
            s = sigs[0]
            # Simple heuristic: Top/Bottom split at 33/66%
            y_loc = "Top" if s['y'] < 33 else "Bottom" if s['y'] > 66 else "Middle"
            x_loc = "Left" if s['x'] < 50 else "Right"
            sig_loc = f"{y_loc} {x_loc}"

        # Q6: Top Left Mark? (x < 25%, y < 25%)
        marks = [x for x in items if 'Archivist mark' in x.get('rectanglelabels', [])]
        tl_mark = any(m['x'] < 25 and m['y'] < 25 for m in marks)
        
        # Q7: Additional Marks count
        count_marks = len(marks)
        if tl_mark: count_marks = max(0, count_marks - 1)

        stats.append({
            'page_id': row['page_id'],
            'Is_Signed': is_signed,
            'Signature_Location': sig_loc,
            'Has_Mark_TopLeft': 'Y' if tl_mark else 'N',
            'Count_Additional_Marks': count_marks
        })
    dfs.append(pd.DataFrame(stats))

    # ---------------------------------------------------------
    # 3. CON & E (Prepositions/Conjunctions + Unnamed Travelers)
    # ---------------------------------------------------------
    print("Processing Con & E...")
    df = pd.read_csv(FILES['con_e'])
    df['page_id'] = df['image'].apply(clean_filename)
    
    stats = []
    for _, row in df.iterrows():
        boxes = parse_json_cell(row['con-e-ed'])
        texts = parse_json_cell(row['text-after-con-e-ed'])
        
        row_data = {
            'page_id': row['page_id'],
            'Count_Con': 0, 'Con_Next_Words': [],
            'Count_E': 0, 'E_Next_Words': [],
            'Traveler_Unnamed_L': 0, 'Traveler_Unnamed_R': 0
        }
        
        # Zip safely handles if lists aren't same length (though they should be)
        for box, text_val in zip(boxes, texts):
            lbl = box.get('rectanglelabels', [''])[0]
            side = get_side(box.get('x', 0))
            text_str = str(text_val) if text_val else ""
            
            # Q9, Q10: Con
            if 'con' in lbl:
                row_data['Count_Con'] += 1
                row_data['Con_Next_Words'].append(text_str)
            
            # Q11, Q12: E
            elif any(x in lbl for x in ['e', 'ed', 'eed']):
                row_data['Count_E'] += 1
                row_data['E_Next_Words'].append(text_str)
            
            # Q14: Unnamed Travelers (Derived from Con/E text)
            count = parse_people_count(text_str)
            if count > 0:
                if side == 'Left': row_data['Traveler_Unnamed_L'] += count
                else: row_data['Traveler_Unnamed_R'] += count

        # Join lists for CSV readability
        row_data['Con_Next_Words'] = " | ".join(row_data['Con_Next_Words'])
        row_data['E_Next_Words'] = " | ".join(row_data['E_Next_Words'])
        stats.append(row_data)
    dfs.append(pd.DataFrame(stats))

    # ---------------------------------------------------------
    # 4. BRACKETS
    # ---------------------------------------------------------
    print("Processing Brackets...")
    df = pd.read_csv(FILES['brackets'])
    df['page_id'] = df['image'].apply(clean_filename)
    
    stats = []
    for _, row in df.iterrows():
        items = parse_json_cell(row['entities'])
        annot_text = str(row['bracket_annotation_text']) if pd.notna(row['bracket_annotation_text']) else ""
        
        # Q18: Total Count
        brackets = [x for x in items if any(k in x.get('rectanglelabels', []) for k in ['bracket', 'contained by bracket'])]
        
        # Q19: Per Column
        l_count = sum(1 for b in brackets if get_side(b['x']) == 'Left')
        r_count = sum(1 for b in brackets if get_side(b['x']) == 'Right')

        # Q20: Annotation Y/N + Text
        stats.append({
            'page_id': row['page_id'],
            'Count_Brackets': len(brackets),
            'Brackets_Left': l_count,
            'Brackets_Right': r_count,
            'Has_Bracket_Annotation': 'Y' if annot_text else 'N',
            'Bracket_Annotation_Words': annot_text
        })
    dfs.append(pd.DataFrame(stats))

    # ---------------------------------------------------------
    # 5. NAMED TRAVELERS & ORIGINS
    # ---------------------------------------------------------
    print("Processing Named Travelers...")
    df = pd.read_csv(FILES['named'])
    df['page_id'] = df['image'].apply(clean_filename)
    
    stats = []
    for _, row in df.iterrows():
        items = parse_json_cell(row['named travelers - origin'])
        
        # Q13: Named Travelers per Column
        travelers = [x for x in items if 'Traveler Name' in x.get('rectanglelabels', [])]
        t_l = sum(1 for t in travelers if get_side(t['x']) == 'Left')
        t_r = sum(1 for t in travelers if get_side(t['x']) == 'Right')
        
        # Q15: Places of Origin (Count Instances)
        count_origins = sum(1 for x in items if 'Place of origin' in x.get('rectanglelabels', []))

        # Q16: Unique Places of Stay
        stays = [x for x in items if 'Place of Stay' in x.get('rectanglelabels', [])]
        unique_stays = set()
        
        # Attempt to read text for uniqueness. If missing, this fallback ensures code doesn't crash.
        for s in stays:
            txt = s.get('text', '') # Assuming text might be in JSON
            if txt:
                unique_stays.add(txt.lower().strip())
        
        # Fallback: If no text found in any label, Unique Count == Instance Count
        count_unique_stays = len(unique_stays) if unique_stays else len(stays)

        # Q17: Places of Stay per Column
        s_l = sum(1 for s in stays if get_side(s['x']) == 'Left')
        s_r = sum(1 for s in stays if get_side(s['x']) == 'Right')

        stats.append({
            'page_id': row['page_id'],
            'Traveler_Named_L': t_l,
            'Traveler_Named_R': t_r,
            'Count_Instances_Origin': count_origins,
            'Count_Unique_Stays': count_unique_stays,
            'Stays_Left': s_l,
            'Stays_Right': s_r
        })
    dfs.append(pd.DataFrame(stats))

    # ---------------------------------------------------------
    # 6. QUONDAM
    # ---------------------------------------------------------
    print("Processing Quondam...")
    df = pd.read_csv(FILES['quondam'])
    df['page_id'] = df['image'].apply(clean_filename)
    
    stats = []
    for _, row in df.iterrows():
        boxes = parse_json_cell(row['quondam instances'])
        # Q8: Count Quondam
        stats.append({
            'page_id': row['page_id'],
            'Count_Quondam': sum(1 for x in boxes if 'Quondam' in x.get('rectanglelabels', []))
        })
    dfs.append(pd.DataFrame(stats))

    # ---------------------------------------------------------
    # MERGE EVERYTHING
    # ---------------------------------------------------------
    print("Merging data...")
    final_df = dfs[0]
    for d in dfs[1:]:
        final_df = pd.merge(final_df, d, on='page_id', how='outer')

    # Cleanup NaNs
    final_df = final_df.fillna(0)
    
    # Fix Y/N columns being 0
    yn_cols = ['Has_Date_Top', 'Is_Signed', 'Has_Mark_TopLeft', 'Has_Bracket_Annotation', 'Two_Columns']
    for c in yn_cols:
        if c in final_df.columns:
            final_df[c] = final_df[c].replace(0, 'N')
            
    # Fix String columns being 0
    str_cols = ['Year_Value', 'Signature_Location', 'Con_Next_Words', 'E_Next_Words', 'Bracket_Annotation_Words']
    for c in str_cols:
        if c in final_df.columns:
            final_df[c] = final_df[c].replace(0, "")

    output_filename = 'aggregated_results.csv'
    final_df.to_csv(output_filename, index=False)
    print(f"Success! Aggregated data saved to {output_filename}")

if __name__ == "__main__":
    process_all()