#!/usr/bin/env python3
"""
Convert PLY label files to ABC dataset label format.
Automatically processes /home/cle/Work/FAIR-CONF/data/toy_dino_6d_16s/lb directory.

Input format:
-1  # non-annotated (will be kept as -1)
0   # edge point (will be converted to 1)  
1   # non-edge point (will be converted to 0)

Output format (.lb):
#/path/to/original/file.yml
-1  # non-annotated (unchanged)
1   # edge (converted from 0)
0   # non-edge (converted from 1)
...
"""

import os
import shutil
import tempfile
from pathlib import Path


def convert_label_format(input_file, output_file, source_comment=None):
    """
    Convert a label file to ABC dataset label format.
    
    Args:
        input_file (str): Path to input file
        output_file (str): Path to output .lb file
        source_comment (str, optional): Comment line for the header
    """
    try:
        # Read the input file
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # Process labels with cleaner logic
        converted_labels = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Skip comment lines that start with #
            if line.startswith('#'):
                continue
                
            try:
                label = int(line)
                if label == -1:
                    converted_labels.append('-1')  # Keep non-annotated as -1
                elif label == 0:
                    converted_labels.append('1')   # Edge: 0 -> 1
                elif label == 1:
                    converted_labels.append('0')   # Non-edge: 1 -> 0
                else:
                    print(f"Warning: Unexpected label '{label}' at line {line_num} in {input_file}, treating as non-annotated")
                    converted_labels.append('-1')
            except ValueError:
                print(f"Warning: Invalid label '{line}' at line {line_num} in {input_file}, skipping")
                continue
        
        # Write the output file
        with open(output_file, 'w') as f:
            # Write header comment
            if source_comment:
                f.write(f"#{source_comment}\n")
            else:
                # Generate a generic comment based on the filename
                base_name = Path(input_file).stem
                f.write(f"/home/ano2lb/{base_name}_features.yml\n")
            
            # Write converted labels
            for label in converted_labels:
                f.write(f"{label}\n")
        
        return len(converted_labels)
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return 0
    except Exception as e:
        print(f"Error converting file {input_file}: {e}")
        return 0


def process_directory():
    """
    Process all files in the target directory and convert them to ABC format.
    """
    target_dir = Path("/home/cle/data/dtu_results_pc/IML_scan24/lb")
    
    if not target_dir.exists():
        print(f"Error: Target directory '{target_dir}' does not exist")
        return
    
    print(f"Processing directory: {target_dir}")
    print("Conversion logic:")
    print("  -1 (non-annotated) → -1 (unchanged)")
    print("  0 (edge) → 1 (edge)")
    print("  1 (non-edge) → 0 (non-edge)")
    
    # Find all files in the directory (any extension)
    all_files = [f for f in target_dir.iterdir() if f.is_file()]
    
    if not all_files:
        print(f"No files found in '{target_dir}'")
        return
    
    print(f"Found {len(all_files)} files to process")
    
    # Create a temporary directory for conversion
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        converted_files = []
        total_points = 0
        total_edge_points = 0
        total_non_edge_points = 0
        total_non_annotated = 0
        
        # Convert each file
        for input_file in all_files:
            print(f"Processing: {input_file.name}")
            
            # Create temporary output file
            temp_output = temp_path / f"{input_file.stem}.lb"
            
            # Convert the file
            points_converted = convert_label_format(str(input_file), str(temp_output))
            
            if points_converted > 0:
                converted_files.append((temp_output, input_file.stem))
                total_points += points_converted
                
                # Count different labels in converted file
                with open(temp_output, 'r') as f:
                    lines = f.readlines()[1:]  # Skip header
                    edge_count = sum(1 for line in lines if line.strip() == '1')
                    non_edge_count = sum(1 for line in lines if line.strip() == '0')
                    non_annotated_count = sum(1 for line in lines if line.strip() == '-1')
                    
                    total_edge_points += edge_count
                    total_non_edge_points += non_edge_count
                    total_non_annotated += non_annotated_count
                
                print(f"  ✓ Converted {points_converted} points ({edge_count} edges, {non_edge_count} non-edges, {non_annotated_count} non-annotated)")
            else:
                print(f"  ✗ Failed to convert {input_file.name}")
        
        if not converted_files:
            print("No files were successfully converted")
            return
        
        # Remove all original files
        print(f"\nRemoving {len(all_files)} original files...")
        for file_path in all_files:
            try:
                file_path.unlink()
                print(f"  ✓ Removed {file_path.name}")
            except Exception as e:
                print(f"  ✗ Failed to remove {file_path.name}: {e}")
        
        # Copy converted files back to the original directory
        print(f"\nCopying {len(converted_files)} converted files...")
        for temp_file, base_name in converted_files:
            output_file = target_dir / f"{base_name}.lb"
            try:
                shutil.copy2(temp_file, output_file)
                print(f"  ✓ Created {output_file.name}")
            except Exception as e:
                print(f"  ✗ Failed to copy {base_name}.lb: {e}")
        
        # Print final statistics
        print(f"\n" + "="*50)
        print(f"CONVERSION COMPLETED")
        print(f"="*50)
        print(f"Total files processed: {len(all_files)}")
        print(f"Successfully converted: {len(converted_files)}")
        print(f"Total points: {total_points}")
        print(f"Edge points (1): {total_edge_points} ({total_edge_points/total_points*100:.2f}%)")
        print(f"Non-edge points (0): {total_non_edge_points} ({total_non_edge_points/total_points*100:.2f}%)")
        print(f"Non-annotated (-1): {total_non_annotated} ({total_non_annotated/total_points*100:.2f}%)")
        print(f"Output directory: {target_dir}")


def main():
    """
    Main function - automatically processes the target directory.
    """
    print("="*50)
    print("PLY to ABC Label Format Converter")
    print("="*50)
    print("Target directory: /home/cle/data/dtu_results_pc/IML_scan24/lb")
    print("Conversion: -1 → -1 (non-annotated), 0 → 1 (edge), 1 → 0 (non-edge)")
    print("="*50)
    
    # Ask for confirmation
    response = input("This will replace ALL files in the target directory. Continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    # Process the directory
    process_directory()
    
    print("\nConversion process completed!")


if __name__ == "__main__":
    main()