import pandas as pd
import json
import os

def parse_bounding_boxes(bbox_string):
    """Parse the bounding box JSON string and return list of boxes with labels."""
    if pd.isna(bbox_string) or bbox_string == "":
        return []
    try:
        boxes = json.loads(bbox_string)
        return boxes if isinstance(boxes, list) else []
    except (json.JSONDecodeError, TypeError):
        return []

def calculate_side(x, width, original_width):
    """
    Calculate if a bounding box is on the left or right side of the image.
    Uses the center point of the bounding box.
    """
    # x is in percentage, convert to actual position
    center_x = x + (width / 2)
    
    # If center is less than 50%, it's on the left
    return "left" if center_x < 50 else "right"

def count_labels_by_side(boxes):
    """Count how many labels appear on left vs right side."""
    left_count = 0
    right_count = 0
    
    for box in boxes:
        if 'x' in box and 'width' in box:
            side = calculate_side(box['x'], box['width'], 
                                 box.get('original_width', 3024))
            if side == "left":
                left_count += 1
            else:
                right_count += 1
    
    return left_count, right_count

def extract_label_types(boxes):
    """Extract all label types from bounding boxes."""
    labels = []
    for box in boxes:
        if 'rectanglelabels' in box:
            labels.extend(box['rectanglelabels'])
    return labels

def strip_gs_prefix(image_path):
    """Strip the GCS prefix from image path."""
    prefix = 'gs://venice_singlepages_37/'
    if image_path.startswith(prefix):
        return image_path[len(prefix):]
    return image_path

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
            'image': image,
            'image_filename': os.path.basename(image)
        }
        
        # Process date-columns
        if 'date_columns' in dataframes:
            df = dataframes['date_columns']
            row = df[df['image'] == image]
            if not row.empty:
                image_data['date'] = row.iloc[0]['date']
                image_data['has_columns'] = row.iloc[0]['columns']
        
        # Process archivist-mark and signature
        if 'archivist_mark' in dataframes:
            df = dataframes['archivist_mark']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['label'])
                labels = extract_label_types(boxes)
                left, right = count_labels_by_side(boxes)
                
                image_data['archivist_signature_count'] = len(boxes)
                image_data['archivist_signature_left'] = left
                image_data['archivist_signature_right'] = right
                image_data['archivist_signature_types'] = ', '.join(set(labels))
        
        # Process con-and-e
        if 'con_e' in dataframes:
            df = dataframes['con_e']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['con-e-ed'])
                labels = extract_label_types(boxes)
                left, right = count_labels_by_side(boxes)
                
                image_data['con_e_count'] = len(boxes)
                image_data['con_e_left'] = left
                image_data['con_e_right'] = right
                image_data['con_e_types'] = ', '.join(labels)
                
                # Parse text after con-e-ed
                if 'text-after-con-e-ed' in row.columns:
                    text_data = row.iloc[0]['text-after-con-e-ed']
                    if pd.notna(text_data):
                        try:
                            texts = json.loads(text_data) if isinstance(text_data, str) else text_data
                            image_data['con_e_text_labels'] = ' | '.join(texts) if isinstance(texts, list) else str(texts)
                        except:
                            image_data['con_e_text_labels'] = str(text_data)
        
        # Process curly brackets
        if 'brackets' in dataframes:
            df = dataframes['brackets']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['entities'])
                labels = extract_label_types(boxes)
                left, right = count_labels_by_side(boxes)
                
                image_data['bracket_count'] = len(boxes)
                image_data['bracket_left'] = left
                image_data['bracket_right'] = right
                image_data['bracket_types'] = ', '.join(set(labels))
                image_data['bracket_text'] = row.iloc[0]['bracket_annotation_text']
        
        # Process named travelers
        if 'travelers' in dataframes:
            df = dataframes['travelers']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['named travelers - origin'])
                labels = extract_label_types(boxes)
                left, right = count_labels_by_side(boxes)
                
                image_data['traveler_count'] = len(boxes)
                image_data['traveler_left'] = left
                image_data['traveler_right'] = right
                image_data['traveler_label_types'] = ', '.join(set(labels))
        
        # Process quondam
        if 'quondam' in dataframes:
            df = dataframes['quondam']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['quondam instances'])
                labels = extract_label_types(boxes)
                left, right = count_labels_by_side(boxes)
                
                image_data['quondam_count'] = len(boxes)
                image_data['quondam_left'] = left
                image_data['quondam_right'] = right
                
                # Parse names after quondam
                if 'Name after quondam' in row.columns:
                    names = row.iloc[0]['Name after quondam']
                    if pd.notna(names):
                        try:
                            name_list = json.loads(names) if isinstance(names, str) else names
                            image_data['quondam_names'] = ' | '.join(name_list) if isinstance(name_list, list) else str(name_list)
                        except:
                            image_data['quondam_names'] = str(names)
        
        aggregated_data.append(image_data)
    
    # Create final dataframe
    result_df = pd.DataFrame(aggregated_data)
    
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
    print(result.head(2).to_string())
    
    # Show example of stripped image paths
    print("\n" + "="*80)
    print("SAMPLE IMAGE PATHS (showing prefix removal)")
    print("="*80)
    for img in result['image'].head(3):
        print(f"  {img}")
    
    # Save to CSV
    output_file = 'aggregated_annotations.csv'
    result.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Display side distribution statistics
    print("\n" + "="*80)
    print("LEFT vs RIGHT DISTRIBUTION")
    print("="*80)
    
    for prefix in ['archivist_signature', 'con_e', 'bracket', 'traveler', 'quondam']:
        left_col = f'{prefix}_left'
        right_col = f'{prefix}_right'
        if left_col in result.columns and right_col in result.columns:
            total_left = result[left_col].sum()
            total_right = result[right_col].sum()
            print(f"\n{prefix.replace('_', ' ').title()}:")
            print(f"  Left side:  {total_left}")
            print(f"  Right side: {total_right}")
            if total_left + total_right > 0:
                left_pct = (total_left / (total_left + total_right)) * 100
                print(f"  Left %:     {left_pct:.1f}%")