import pandas as pd
import json
import os
import numpy as np
from sklearn.cluster import KMeans

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

def calculate_side_clustering(boxes, label_filter=None):
    """
    Use clustering to identify left and right columns based on x-coordinates.
    This works better for off-center documents.
    
    Args:
        boxes: List of bounding box dictionaries
        label_filter: Optional label type to filter (e.g., "Traveler Name")
    
    Returns:
        Dictionary mapping box index to 'left' or 'right'
    """
    if not boxes:
        return {}
    
    # Filter boxes by label if specified
    filtered_boxes = []
    filtered_indices = []
    for i, box in enumerate(boxes):
        if label_filter:
            if 'rectanglelabels' in box and label_filter in box['rectanglelabels']:
                filtered_boxes.append(box)
                filtered_indices.append(i)
        else:
            filtered_boxes.append(box)
            filtered_indices.append(i)
    
    if len(filtered_boxes) < 2:
        # Not enough boxes to cluster, use simple method
        return {i: calculate_side_simple(box['x'], box['width']) 
                for i, box in enumerate(boxes) if 'x' in box and 'width' in box}
    
    # Extract x-coordinates (use left edge of box)
    x_coords = []
    valid_indices = []
    for i, box in zip(filtered_indices, filtered_boxes):
        if 'x' in box:
            x_coords.append(box['x'])
            valid_indices.append(i)
    
    if len(x_coords) < 2:
        return {i: calculate_side_simple(box['x'], box['width']) 
                for i, box in enumerate(boxes) if 'x' in box and 'width' in box}
    
    # Check if there are actually two distinct columns
    # Calculate the gap between sorted x values
    sorted_x = sorted(x_coords)
    gaps = [sorted_x[i+1] - sorted_x[i] for i in range(len(sorted_x)-1)]
    
    # If there's a significant gap (suggesting two columns), use clustering
    max_gap = max(gaps) if gaps else 0
    mean_gap = np.mean(gaps) if gaps else 0
    
    if max_gap > mean_gap * 2 and max_gap > 5:  # Threshold for detecting two columns
        # Use K-means clustering with k=2
        X = np.array(x_coords).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Determine which cluster is left and which is right
        cluster_centers = kmeans.cluster_centers_.flatten()
        left_cluster = 0 if cluster_centers[0] < cluster_centers[1] else 1
        
        # Create mapping
        side_mapping = {}
        for idx, cluster in zip(valid_indices, clusters):
            side_mapping[idx] = "left" if cluster == left_cluster else "right"
        
        # Fill in any boxes that weren't clustered (non-name boxes) with simple method
        for i, box in enumerate(boxes):
            if i not in side_mapping and 'x' in box and 'width' in box:
                side_mapping[i] = calculate_side_simple(box['x'], box['width'])
        
        return side_mapping
    else:
        # All boxes are in one column or too close together, use simple method
        return {i: calculate_side_simple(box['x'], box['width']) 
                for i, box in enumerate(boxes) if 'x' in box and 'width' in box}

def count_labels_by_side(boxes, use_clustering=False, label_filter=None):
    """
    Count how many labels appear on left vs right side.
    
    Args:
        boxes: List of bounding box dictionaries
        use_clustering: If True, use clustering method; if False, use simple 50% method
        label_filter: Optional label type to filter for clustering
    """
    if not boxes:
        return 0, 0
    
    left_count = 0
    right_count = 0
    
    if use_clustering:
        side_mapping = calculate_side_clustering(boxes, label_filter)
        for i in side_mapping:
            if side_mapping[i] == "left":
                left_count += 1
            else:
                right_count += 1
    else:
        for box in boxes:
            if 'x' in box and 'width' in box:
                side = calculate_side_simple(box['x'], box['width'])
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

def analyze_column_distribution(boxes, label_filter=None):
    """
    Analyze and print the distribution of x-coordinates to help debug column detection.
    """
    if label_filter:
        x_coords = [box['x'] for box in boxes 
                   if 'x' in box and 'rectanglelabels' in box 
                   and label_filter in box['rectanglelabels']]
    else:
        x_coords = [box['x'] for box in boxes if 'x' in box]
    
    if not x_coords:
        return None
    
    sorted_x = sorted(x_coords)
    return {
        'min': min(sorted_x),
        'max': max(sorted_x),
        'mean': np.mean(sorted_x),
        'median': np.median(sorted_x),
        'values': sorted_x
    }

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
        
        # Process archivist-mark and signature (use simple method)
        if 'archivist_mark' in dataframes:
            df = dataframes['archivist_mark']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['label'])
                labels = extract_label_types(boxes)
                left, right = count_labels_by_side(boxes, use_clustering=False)
                
                image_data['archivist_signature_count'] = len(boxes)
                image_data['archivist_signature_left'] = left
                image_data['archivist_signature_right'] = right
                image_data['archivist_signature_types'] = ', '.join(set(labels))
        
        # Process con-and-e (use simple method)
        if 'con_e' in dataframes:
            df = dataframes['con_e']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['con-e-ed'])
                labels = extract_label_types(boxes)
                left, right = count_labels_by_side(boxes, use_clustering=False)
                
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
        
        # Process curly brackets (use simple method)
        if 'brackets' in dataframes:
            df = dataframes['brackets']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['entities'])
                labels = extract_label_types(boxes)
                left, right = count_labels_by_side(boxes, use_clustering=False)
                
                image_data['bracket_count'] = len(boxes)
                image_data['bracket_left'] = left
                image_data['bracket_right'] = right
                image_data['bracket_types'] = ', '.join(set(labels))
                image_data['bracket_text'] = row.iloc[0]['bracket_annotation_text']
        
        # Process named travelers (USE CLUSTERING METHOD)
        if 'travelers' in dataframes:
            df = dataframes['travelers']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['named travelers - origin'])
                labels = extract_label_types(boxes)
                
                # Use clustering method specifically for traveler names
                left, right = count_labels_by_side(boxes, use_clustering=True, 
                                                   label_filter="Traveler Name")
                
                image_data['traveler_count'] = len(boxes)
                image_data['traveler_left'] = left
                image_data['traveler_right'] = right
                image_data['traveler_label_types'] = ', '.join(set(labels))
                
                # Store x-coordinate distribution for debugging
                dist = analyze_column_distribution(boxes, "Traveler Name")
                if dist:
                    image_data['traveler_x_min'] = dist['min']
                    image_data['traveler_x_max'] = dist['max']
                    image_data['traveler_x_mean'] = dist['mean']
        
        # Process quondam (use simple method)
        if 'quondam' in dataframes:
            df = dataframes['quondam']
            row = df[df['image'] == image]
            if not row.empty:
                boxes = parse_bounding_boxes(row.iloc[0]['quondam instances'])
                labels = extract_label_types(boxes)
                left, right = count_labels_by_side(boxes, use_clustering=False)
                
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
    
    # Show traveler x-coordinate distribution for debugging
    print("\n" + "="*80)
    print("TRAVELER NAME X-COORDINATE DISTRIBUTION (sample)")
    print("="*80)
    traveler_cols = ['image', 'traveler_count', 'traveler_left', 'traveler_right', 
                     'traveler_x_min', 'traveler_x_max', 'traveler_x_mean']
    available_cols = [col for col in traveler_cols if col in result.columns]
    if available_cols:
        print(result[available_cols].head(5).to_string())
    
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
            method = "CLUSTERING" if prefix == 'traveler' else "SIMPLE (50%)"
            print(f"\n{prefix.replace('_', ' ').title()} [{method}]:")
            print(f"  Left side:  {total_left}")
            print(f"  Right side: {total_right}")
            if total_left + total_right > 0:
                left_pct = (total_left / (total_left + total_right)) * 100
                print(f"  Left %:     {left_pct:.1f}%")