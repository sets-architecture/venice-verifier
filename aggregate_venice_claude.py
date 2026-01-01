import pandas as pd
import json
import os
import re
from difflib import SequenceMatcher

def parse_bounding_boxes(bbox_string):
    """Parse the bounding box JSON string and return list of boxes with labels."""
    if pd.isna(bbox_string) or bbox_string == "":
        return []
    try:
        boxes = json.loads(bbox_string)
        return boxes if isinstance(boxes, list) else []
    except (json.JSONDecodeError, TypeError):
        return []

def calculate_side_simple(x, width):
    """
    Simple calculation: if center is on left or right half.
    """
    center_x = x + (width / 2)
    return "left" if center_x < 50 else "right"

def calculate_side_gap_detection(boxes, label_filter=None):
    """
    Use gap detection to identify left and right columns based on x-coordinates.
    This works better for off-center documents without needing sklearn.
    
    Args:
        boxes: List of bounding box dictionaries
        label_filter: Optional label type to filter (e.g., "Traveler Name")
    
    Returns:
        Dictionary mapping box index to 'left' or 'right'
    """
    if not boxes:
        return {}
    
    # Filter boxes by label if specified
    filtered_data = []
    for i, box in enumerate(boxes):
        if 'x' not in box:
            continue
        if label_filter:
            if 'rectanglelabels' in box and label_filter in box['rectanglelabels']:
                filtered_data.append((i, box['x']))
        else:
            filtered_data.append((i, box['x']))
    
    if len(filtered_data) < 2:
        # Not enough boxes to detect columns, use simple method
        return {i: calculate_side_simple(box['x'], box['width']) 
                for i, box in enumerate(boxes) if 'x' in box and 'width' in box}
    
    # Sort by x-coordinate
    filtered_data.sort(key=lambda x: x[1])
    indices, x_coords = zip(*filtered_data)
    
    # Find the largest gap between consecutive x values
    max_gap = 0
    gap_index = 0
    for i in range(len(x_coords) - 1):
        gap = x_coords[i + 1] - x_coords[i]
        if gap > max_gap:
            max_gap = gap
            gap_index = i
    
    # Calculate mean gap (excluding the max gap)
    gaps = [x_coords[i + 1] - x_coords[i] for i in range(len(x_coords) - 1)]
    gaps_without_max = [g for g in gaps if g != max_gap]
    mean_gap = sum(gaps_without_max) / len(gaps_without_max) if gaps_without_max else 0
    
    # If the max gap is significantly larger than the mean, we have two columns
    if max_gap > mean_gap * 2 and max_gap > 5:
        # Split into two groups at the gap
        threshold = (x_coords[gap_index] + x_coords[gap_index + 1]) / 2
        
        side_mapping = {}
        for idx, x in zip(indices, x_coords):
            side_mapping[idx] = "left" if x < threshold else "right"
        
        # Fill in any boxes that weren't included (non-name boxes) with simple method
        for i, box in enumerate(boxes):
            if i not in side_mapping and 'x' in box and 'width' in box:
                side_mapping[i] = calculate_side_simple(box['x'], box['width'])
        
        return side_mapping
    else:
        # All boxes are in one column or too close together, use simple method
        return {i: calculate_side_simple(box['x'], box['width']) 
                for i, box in enumerate(boxes) if 'x' in box and 'width' in box}

def count_by_label_and_side(boxes, target_label, use_gap_detection=False):
    """
    Count boxes with a specific label by side.
    
    Args:
        boxes: List of bounding box dictionaries
        target_label: The label to filter for
        use_gap_detection: Whether to use gap detection for column assignment
    
    Returns:
        (left_count, right_count, total_count)
    """
    filtered_boxes = [box for box in boxes 
                     if 'rectanglelabels' in box 
                     and target_label in box['rectanglelabels']]
    
    if not filtered_boxes:
        return 0, 0, 0
    
    left_count = 0
    right_count = 0
    
    if use_gap_detection:
        side_mapping = calculate_side_gap_detection(filtered_boxes)
        for i in side_mapping:
            if side_mapping[i] == "left":
                left_count += 1
            else:
                right_count += 1
    else:
        for box in filtered_boxes:
            if 'x' in box and 'width' in box:
                side = calculate_side_simple(box['x'], box['width'])
                if side == "left":
                    left_count += 1
                else:
                    right_count += 1
    
    return left_count, right_count, len(filtered_boxes)

def strip_gs_prefix(image_path):
    """Strip the GCS prefix from image path."""
    prefix = 'gs://venice_singlepages_37/'
    if image_path.startswith(prefix):
        return image_path[len(prefix):]
    return image_path

def extract_year_from_date(date_string):
    """Extract year from date string."""
    if pd.isna(date_string):
        return None
    # Try to find a 4-digit year in the string
    import re
    match = re.search(r'\b(1[4-9]\d{2}|20\d{2})\b', str(date_string))
    if match:
        return match.group(1)
    return None

def fuzzy_match(word, keywords, threshold=0.8):
    """
    Check if a word matches any keyword with fuzzy matching.
    Returns the matched keyword if similarity >= threshold, None otherwise.
    """
    word = word.lower().strip()
    for keyword in keywords:
        similarity = SequenceMatcher(None, word, keyword).ratio()
        if similarity >= threshold:
            return keyword
    return None

def parse_unnamed_travelers_in_text(text):
    """
    Parse Italian text to count ALL unnamed travelers mentioned in a single text entry.
    This handles entries like "moglie, una figliuola" which should count as 2 people.
    
    The function:
    1. Splits text by commas and common separators
    2. Counts people in each phrase
    3. Returns total count for the entire text entry
    
    Examples:
    - "moglie" → 1
    - "moglie, una figliuola" → 2 (moglie=1 + una figliuola=1)
    - "tre figli" → 3
    - "due domestici" → 2
    - "con valigia" → 0 (no person keywords)
    """
    if not text or pd.isna(text):
        return 0
    
    text_lower = str(text).lower().strip()
    
    # Person-related keywords (Italian)
    person_keywords = [
        # Spouse/partner
        'moglie', 'consorte', 'marito', 'sposo', 'sposa',
        # Children
        'figlio', 'figlia', 'figliuolo', 'figliuola', 'figli', 'figliuoli', 'figliuole',
        'bambino', 'bambina', 'bambini',
        # Family members
        'madre', 'padre', 'fratello', 'sorella', 'fratelli', 'sorelle',
        'nonno', 'nonna', 'nipote', 'nipoti', 'cugino', 'cugina',
        # Servants/staff
        'domestico', 'domestica', 'domestici', 'domestiche',
        'servo', 'serva', 'servi', 'serve',
        'servitore', 'servitrice', 'servitori', 'servitrici',
        'cameriere', 'cameriera',
        # Companions
        'compagno', 'compagna', 'compagni', 'compagne',
        'accompagnatore', 'accompagnatrice',
        # General
        'persona', 'persone', 'gente', 'famiglia', 'famiglie',
        'individuo', 'individui'
    ]
    
    # Italian number words mapping
    number_words = {
        'un': 1, 'una': 1, 'uno': 1,
        'due': 2,
        'tre': 3,
        'quattro': 4,
        'cinque': 5,
        'sei': 6,
        'sette': 7,
        'otto': 8,
        'nove': 9,
        'dieci': 10,
        'undici': 11,
        'dodici': 12,
        'tredici': 13,
        'quattordici': 14,
        'quindici': 15
    }
    
    # Split by common separators (comma, semicolon, "e", "ed")
    # But be careful not to split on "e" that's part of a word
    phrases = re.split(r'[,;]|\se\s|\sed\s', text_lower)
    
    total_count = 0
    
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
        
        # Split phrase into words
        words = re.findall(r'\b\w+\b', phrase)
        
        # Check for person keywords using fuzzy matching
        keyword_found = False
        for word in words:
            if fuzzy_match(word, person_keywords, threshold=0.8):
                keyword_found = True
                break
        
        # If no person keyword found in this phrase, skip it
        if not keyword_found:
            continue
        
        # Look for number words in this phrase
        phrase_count = None
        for word in words:
            if word in number_words:
                phrase_count = number_words[word]
                break
        
        # If no number word, look for digit
        if phrase_count is None:
            digit_match = re.search(r'\b(\d+)\b', phrase)
            if digit_match:
                phrase_count = int(digit_match.group(1))
        
        # If keyword found but no number, count as 1 person
        if phrase_count is None:
            phrase_count = 1
        
        total_count += phrase_count
    
    return total_count

def separate_con_e_text_with_boxes(boxes, text_list):
    """
    Separate the following text based on whether it comes after 'con' or 'e/ed/eed',
    and also track which column each text belongs to.
    
    Args:
        boxes: List of bounding box dictionaries
        text_list: List of text strings following con/e
    
    Returns:
        (con_data, e_data) where each is a list of (text, side) tuples
    """
    if not text_list or len(text_list) == 0:
        return [], []
    
    con_boxes = []
    e_boxes = []
    
    for box in boxes:
        if 'rectanglelabels' not in box:
            continue
        if 'con' in box['rectanglelabels']:
            side = calculate_side_simple(box['x'], box['width']) if 'x' in box and 'width' in box else 'unknown'
            con_boxes.append((box, side))
        elif any(label in ['e', 'ed', 'eed'] for label in box['rectanglelabels']):
            side = calculate_side_simple(box['x'], box['width']) if 'x' in box and 'width' in box else 'unknown'
            e_boxes.append((box, side))
    
    # Pair texts with boxes
    con_data = []
    e_data = []
    
    con_count = len(con_boxes)
    
    for i, text in enumerate(text_list):
        if i < con_count:
            side = con_boxes[i][1]
            con_data.append((text, side))
        else:
            e_index = i - con_count
            if e_index < len(e_boxes):
                side = e_boxes[e_index][1]
                e_data.append((text, side))
    
    return con_data, e_data

def count_unnamed_travelers_by_column(con_data, e_data):
    """
    Count unnamed travelers by column from con and e text data.
    Now counts ALL people mentioned within each text entry, not just the number of entries.
    
    Args:
        con_data: List of (text, side) tuples for con
        e_data: List of (text, side) tuples for e
    
    Returns:
        (left_count, right_count, total_count)
    """
    left_count = 0
    right_count = 0
    
    # Count from con data - parse each text entry for ALL people mentioned
    for text, side in con_data:
        count = parse_unnamed_travelers_in_text(text)
        if side == 'left':
            left_count += count
        elif side == 'right':
            right_count += count
    
    # Count from e data - parse each text entry for ALL people mentioned
    for text, side in e_data:
        count = parse_unnamed_travelers_in_text(text)
        if side == 'left':
            left_count += count
        elif side == 'right':
            right_count += count
    
    return left_count, right_count, left_count + right_count

def process_annotations(csv_files):
    """
    Process all CSV files and aggregate data by image filename.
    
    Args:
        csv_files: Dictionary with keys as identifiers and values as file paths
    
    Returns:
        DataFrame with aggregated data
    """
    
    # Read all CSV files and strip GCS prefix from image column
    dataframes = {}
    for name, filepath in csv_files.items():
        try:
            df = pd.read_csv(filepath)
            # Strip the GCS prefix from image paths
            df['image'] = df['image'].apply(strip_gs_prefix)
            dataframes[name] = df
            print(f"Loaded {name}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    # Get unique images from first dataframe (they should all have the same images)
    base_df = list(dataframes.values())[0]
    images = base_df['image'].unique()
    print(f"\nFound {len(images)} unique images")
    
    # Create aggregated data structure
    aggregated_data = []
    
    for image in images:
        image_data = {
            'image_filename': os.path.basename(image)
        }
        
        # ===================================================================
        # DATE AND COLUMNS
        # ===================================================================
        if 'date_columns' in dataframes:
            df = dataframes['date_columns']
            row = df[df['image'] == image]
            if not row.empty:
                date_value = row.iloc[0]['date']
                year = extract_year_from_date(date_value)
                
                image_data['has_date'] = 'Y' if pd.notna(date_value) and date_value != '' else 'N'
                image_data['date_full_text'] = date_value if pd.notna(date_value) else ''
                image_data['date_year_extracted'] = year if year else ''
                image_data['has_two_columns'] = row.iloc[0]['columns']
        
        # ===================================================================
        # ARCHIVIST MARKS AND SIGNATURES
        # ===================================================================
        if 'archivist_mark' in dataframes:
            df = dataframes['archivist_mark']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['label'])
                
                # Count signatures
                signatures = [box for box in boxes 
                            if 'rectanglelabels' in box 
                            and 'Signature' in box['rectanglelabels']]
                
                # Count archivist marks
                archivist_marks = [box for box in boxes 
                                 if 'rectanglelabels' in box 
                                 and 'Archivist mark' in box['rectanglelabels']]
                
                image_data['is_signed'] = 'Y' if len(signatures) > 0 else 'N'
                
                # Determine signature location(s)
                if len(signatures) > 0:
                    sig_locations = []
                    for sig in signatures:
                        if 'x' in sig and 'width' in sig:
                            side = calculate_side_simple(sig['x'], sig['width'])
                            sig_locations.append(side)
                    image_data['signature_location'] = ', '.join(sig_locations)
                else:
                    image_data['signature_location'] = ''
                
                # Check for top left corner mark (y < 10 and x < 20)
                top_left_marks = [mark for mark in archivist_marks 
                                 if 'x' in mark and 'y' in mark 
                                 and mark['y'] < 10 and mark['x'] < 20]
                
                image_data['has_top_left_mark'] = 'Y' if len(top_left_marks) > 0 else 'N'
                image_data['additional_archivist_marks_count'] = max(0, len(archivist_marks) - len(top_left_marks))
                image_data['total_archivist_marks_count'] = len(archivist_marks)
        
        # ===================================================================
        # QUONDAM
        # ===================================================================
        if 'quondam' in dataframes:
            df = dataframes['quondam']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['quondam instances'])
                
                # Count only "Quondam" labels by column
                left, right, total = count_by_label_and_side(boxes, 'Quondam', use_gap_detection=False)
                
                image_data['quondam_total'] = total
                image_data['quondam_left'] = left
                image_data['quondam_right'] = right
                
                # Extract names after quondam
                if 'Name after quondam' in row.columns:
                    names = row.iloc[0]['Name after quondam']
                    if pd.notna(names):
                        try:
                            name_list = json.loads(names) if isinstance(names, str) else names
                            image_data['quondam_names'] = ' | '.join(name_list) if isinstance(name_list, list) else str(name_list)
                        except:
                            image_data['quondam_names'] = str(names)
                    else:
                        image_data['quondam_names'] = ''
                else:
                    image_data['quondam_names'] = ''
        
        # ===================================================================
        # CON AND E - SEPARATED WITH UNNAMED TRAVELER COUNTS
        # ===================================================================
        if 'con_e' in dataframes:
            df = dataframes['con_e']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['con-e-ed'])
                
                # Count "con" instances by column (instances, not people)
                left, right, total = count_by_label_and_side(boxes, 'con', use_gap_detection=False)
                image_data['con_total'] = total
                image_data['con_left'] = left
                image_data['con_right'] = right
                
                # Count "e" or "ed" or "eed" instances (all variants of the conjunction)
                e_boxes = [box for box in boxes 
                          if 'rectanglelabels' in box 
                          and any(label in ['e', 'ed', 'eed'] for label in box['rectanglelabels'])]
                
                # Count e conjunction by column (instances, not people)
                left_e = sum(1 for box in e_boxes 
                           if 'x' in box and 'width' in box 
                           and calculate_side_simple(box['x'], box['width']) == 'left')
                right_e = len(e_boxes) - left_e
                
                image_data['e_conjunction_total'] = len(e_boxes)
                image_data['e_conjunction_left'] = left_e
                image_data['e_conjunction_right'] = right_e
                
                # Extract and SEPARATE text after con/e, AND COUNT UNNAMED TRAVELERS
                if 'text-after-con-e-ed' in row.columns:
                    text_data = row.iloc[0]['text-after-con-e-ed']
                    if pd.notna(text_data):
                        try:
                            texts = json.loads(text_data) if isinstance(text_data, str) else text_data
                            if isinstance(texts, list):
                                con_data, e_data = separate_con_e_text_with_boxes(boxes, texts)
                                
                                # Store text
                                con_texts = [text for text, side in con_data]
                                e_texts = [text for text, side in e_data]
                                
                                image_data['text_after_con'] = ' | '.join(con_texts) if con_texts else ''
                                image_data['text_after_e'] = ' | '.join(e_texts) if e_texts else ''
                                
                                # Count unnamed travelers by column - NOW COUNTS ALL PEOPLE IN TEXT
                                left_unnamed, right_unnamed, total_unnamed = count_unnamed_travelers_by_column(con_data, e_data)
                                image_data['unnamed_travelers_left'] = left_unnamed
                                image_data['unnamed_travelers_right'] = right_unnamed
                                image_data['unnamed_travelers_total'] = total_unnamed
                            else:
                                image_data['text_after_con'] = str(texts)
                                image_data['text_after_e'] = ''
                                image_data['unnamed_travelers_left'] = 0
                                image_data['unnamed_travelers_right'] = 0
                                image_data['unnamed_travelers_total'] = 0
                        except Exception as e:
                            print(f"Error processing con/e text for {image}: {e}")
                            image_data['text_after_con'] = str(text_data)
                            image_data['text_after_e'] = ''
                            image_data['unnamed_travelers_left'] = 0
                            image_data['unnamed_travelers_right'] = 0
                            image_data['unnamed_travelers_total'] = 0
                    else:
                        image_data['text_after_con'] = ''
                        image_data['text_after_e'] = ''
                        image_data['unnamed_travelers_left'] = 0
                        image_data['unnamed_travelers_right'] = 0
                        image_data['unnamed_travelers_total'] = 0
        
        # ===================================================================
        # TRAVELERS
        # ===================================================================
        if 'travelers' in dataframes:
            df = dataframes['travelers']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['named travelers - origin'])
                
                # Count named travelers by column (using gap detection)
                left, right, total = count_by_label_and_side(boxes, 'Traveler Name', use_gap_detection=True)
                image_data['named_travelers_total'] = total
                image_data['named_travelers_left'] = left
                image_data['named_travelers_right'] = right
                
                # Count places of stay by column (using gap detection)
                left, right, total = count_by_label_and_side(boxes, 'Place of Stay', use_gap_detection=True)
                image_data['place_of_stay_total'] = total
                image_data['place_of_stay_left'] = left
                image_data['place_of_stay_right'] = right
                
                # Count places of origin
                left, right, total = count_by_label_and_side(boxes, 'Place of origin', use_gap_detection=True)
                image_data['place_of_origin_total'] = total
                image_data['place_of_origin_left'] = left
                image_data['place_of_origin_right'] = right
                
                # Manual entry fields for unique places only
                image_data['unique_places_of_origin_nationality'] = ''
                image_data['unique_places_of_stay'] = ''
                image_data['place_of_stay_with_person_name'] = ''
        
        # ===================================================================
        # CURLY BRACKETS
        # ===================================================================
        if 'brackets' in dataframes:
            df = dataframes['brackets']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['entities'])
                
                # Count only actual bracket symbols by column
                left, right, total = count_by_label_and_side(boxes, 'bracket', use_gap_detection=False)
                
                image_data['bracket_total'] = total
                image_data['bracket_left'] = left
                image_data['bracket_right'] = right
                
                # Count bracket annotations
                annotation_boxes = [box for box in boxes 
                                  if 'rectanglelabels' in box 
                                  and 'bracket annotation' in box['rectanglelabels']]
                
                image_data['has_bracket_annotation'] = 'Y' if len(annotation_boxes) > 0 else 'N'
                
                # Get annotation text
                bracket_text = row.iloc[0]['bracket_annotation_text']
                image_data['bracket_annotation_text'] = bracket_text if pd.notna(bracket_text) else ''
        
        aggregated_data.append(image_data)
    
    # Create final dataframe
    result_df = pd.DataFrame(aggregated_data)
    
    # Reorder columns in the exact order specified
    column_order = [
        'image_filename',
        'has_date',
        'date_full_text',
        'date_year_extracted',
        'has_two_columns',
        'is_signed',
        'signature_location',
        'has_top_left_mark',
        'additional_archivist_marks_count',
        'total_archivist_marks_count',
        'quondam_total',
        'quondam_left',
        'quondam_right',
        'quondam_names',
        'con_total',
        'con_left',
        'con_right',
        'text_after_con',
        'e_conjunction_total',
        'e_conjunction_left',
        'e_conjunction_right',
        'text_after_e',
        'named_travelers_total',
        'named_travelers_left',
        'named_travelers_right',
        'unnamed_travelers_total',
        'unnamed_travelers_left',
        'unnamed_travelers_right',
        'place_of_origin_total',
        'place_of_origin_left',
        'place_of_origin_right',
        'unique_places_of_origin_nationality',
        'place_of_stay_total',
        'place_of_stay_left',
        'place_of_stay_right',
        'unique_places_of_stay',
        'place_of_stay_with_person_name',
        'bracket_total',
        'bracket_left',
        'bracket_right',
        'has_bracket_annotation',
        'bracket_annotation_text'
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in result_df.columns]
    result_df = result_df[column_order]
    
    return result_df

# Main execution
if __name__ == "__main__":
    # Define your CSV file paths
    csv_files = {
        'date_columns': 'date-columns-css.csv',
        'archivist_mark': 'archivist-mark---signature.csv',
        'con_e': 'con-and-e.csv',
        'brackets': 'curly-brackets-comment.csv',
        'travelers': 'named-traveler-origin.csv',
        'quondam': 'quondam-csv-2.csv'
    }
    
    # Process all annotations
    result = process_annotations(csv_files)
    
    # Display summary statistics
    print("\n" + "="*80)
    print("AGGREGATION SUMMARY")
    print("="*80)
    print(f"\nTotal images processed: {len(result)}")
    print(f"\nColumns in aggregated dataset: {len(result.columns)}")
    print("\nColumn names:")
    for col in result.columns:
        print(f"  - {col}")
    
    # Display sample of results
    print("\n" + "="*80)
    print("SAMPLE DATA (first 2 rows)")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(result.head(2).to_string())
    
    # Save to CSV
    output_file = 'aggregated_annotations.csv'
    result.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Display what needs manual entry
    print("\n" + "="*80)
    print("FIELDS REQUIRING MANUAL ENTRY")
    print("="*80)
    print("\nThe following columns have been created for manual data entry:")
    print("  - unique_places_of_origin_nationality")
    print("  - unique_places_of_stay")
    print("  - place_of_stay_with_person_name")
    
    # Display comprehensive summary
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    
    print("\n--- DATE AND COLUMNS ---")
    print(f"Pages with dates: {result['has_date'].value_counts().get('Y', 0)}")
    print(f"Pages with two columns: {result['has_two_columns'].value_counts().get('yes', 0)}")
    
    print("\n--- SIGNATURES AND MARKS ---")
    print(f"Pages that are signed: {result['is_signed'].value_counts().get('Y', 0)}")
    print(f"Pages with top-left archivist mark: {result['has_top_left_mark'].value_counts().get('Y', 0)}")
    print(f"Total archivist marks: {result['total_archivist_marks_count'].sum()}")
    
    print("\n--- QUONDAM ---")
    print(f"Total quondam instances: {result['quondam_total'].sum()}")
    print(f"  - Left column: {result['quondam_left'].sum()}")
    print(f"  - Right column: {result['quondam_right'].sum()}")
    
    print("\n--- CON AND E ---")
    print(f"Total 'con' INSTANCES: {result['con_total'].sum()}")
    print(f"  - Left column: {result['con_left'].sum()}")
    print(f"  - Right column: {result['con_right'].sum()}")
    print(f"Total 'e' conjunction INSTANCES: {result['e_conjunction_total'].sum()}")
    print(f"  - Left column: {result['e_conjunction_left'].sum()}")
    print(f"  - Right column: {result['e_conjunction_right'].sum()}")
    
    print("\n--- TRAVELERS (People Counted in Text) ---")
    print(f"Total named travelers: {result['named_travelers_total'].sum()}")
    print(f"  - Left column: {result['named_travelers_left'].sum()}")
    print(f"  - Right column: {result['named_travelers_right'].sum()}")
    print(f"Total unnamed travelers (PEOPLE counted from text): {result['unnamed_travelers_total'].sum()}")
    print(f"  - Left column: {result['unnamed_travelers_left'].sum()}")
    print(f"  - Right column: {result['unnamed_travelers_right'].sum()}")
    print(f"Total places of origin: {result['place_of_origin_total'].sum()}")
    print(f"  - Left column: {result['place_of_origin_left'].sum()}")
    print(f"  - Right column: {result['place_of_origin_right'].sum()}")
    print(f"Total places of stay: {result['place_of_stay_total'].sum()}")
    print(f"  - Left column: {result['place_of_stay_left'].sum()}")
    print(f"  - Right column: {result['place_of_stay_right'].sum()}")
    
    print("\n--- BRACKETS ---")
    print(f"Total brackets: {result['bracket_total'].sum()}")
    print(f"  - Left column: {result['bracket_left'].sum()}")
    print(f"  - Right column: {result['bracket_right'].sum()}")
    print(f"Brackets with annotations: {result['has_bracket_annotation'].value_counts().get('Y', 0)}")