#!/usr/bin/env python3
"""
Reverse Converter: ABC (.lb) back to original PLY format

Input format (.lb):
#/path/to/original/file.yml
-1  # non-annotated (unchanged)
1   # edge (will be converted back to 0)
0   # non-edge (will be converted back to 1)

Output format (original):
-1  # non-annotated (unchanged)
0   # edge point (converted from 1)  
1   # non-edge point (converted from 0)
"""

import os
import shutil
import tempfile
from pathlib import Path


def reverse_label_format(input_file, output_file):
    """
    Convert ABC dataset label format back to original PLY format.
    
    Args:
        input_file (str): Path to input .lb file
        output_file (str): Path to output file (original format)
    """
    try:
        # Read the input .lb file
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # Process labels with reverse conversion logic
        converted_labels = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Skip header comment lines that start with #
            if line.startswith('#'):
                continue
                
            try:
                label = int(line)
                if label == -1:
                    converted_labels.append('-1')  # Keep non-annotated as -1
                elif label == 1:
                    converted_labels.append('0')   # Edge: 1 -> 0 (reverse)
                elif label == 0:
                    converted_labels.append('1')   # Non-edge: 0 -> 1 (reverse)
                else:
                    print(f"Warning: Unexpected label '{label}' at line {line_num} in {input_file}, treating as non-annotated")
                    converted_labels.append('-1')
            except ValueError:
                print(f"Warning: Invalid label '{line}' at line {line_num} in {input_file}, skipping")
                continue
        
        # Write the output file (without header comment)
        with open(output_file, 'w') as f:
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
    Process all .lb files in the target directory and convert them back to original format.
    """
    target_dir = Path("/home/cle/data/dtu_results_pc/IML_scan105/lb")
    
    if not target_dir.exists():
        print(f"Error: Target directory '{target_dir}' does not exist")
        return
    
    print(f"Processing directory: {target_dir}")
    print("Reverse conversion logic:")
    print("  -1 (non-annotated) → -1 (unchanged)")
    print("  1 (edge) → 0 (edge)")
    print("  0 (non-edge) → 1 (non-edge)")
    
    # Find all .lb files in the directory
    lb_files = [f for f in target_dir.iterdir() if f.is_file() and f.suffix == '.lb']
    
    if not lb_files:
        print(f"No .lb files found in '{target_dir}'")
        return
    
    print(f"Found {len(lb_files)} .lb files to process")
    
    # Create a temporary directory for conversion
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        converted_files = []
        total_points = 0
        total_edge_points = 0
        total_non_edge_points = 0
        total_non_annotated = 0
        
        # Convert each .lb file
        for input_file in lb_files:
            print(f"Processing: {input_file.name}")
            
            # Create temporary output file (remove .lb extension)
            base_name = input_file.stem  # filename without .lb extension
            temp_output = temp_path / base_name
            
            # Convert the file
            points_converted = reverse_label_format(str(input_file), str(temp_output))
            
            if points_converted > 0:
                converted_files.append((temp_output, base_name))
                total_points += points_converted
                
                # Count different labels in converted file
                with open(temp_output, 'r') as f:
                    lines = f.readlines()
                    edge_count = sum(1 for line in lines if line.strip() == '0')
                    non_edge_count = sum(1 for line in lines if line.strip() == '1')
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
        
        # Remove all original .lb files
        print(f"\nRemoving {len(lb_files)} original .lb files...")
        for file_path in lb_files:
            try:
                file_path.unlink()
                print(f"  ✓ Removed {file_path.name}")
            except Exception as e:
                print(f"  ✗ Failed to remove {file_path.name}: {e}")
        
        # Copy converted files back to the original directory
        print(f"\nCopying {len(converted_files)} converted files...")
        for temp_file, base_name in converted_files:
            output_file = target_dir / base_name  # No extension added
            try:
                shutil.copy2(temp_file, output_file)
                print(f"  ✓ Created {output_file.name}")
            except Exception as e:
                print(f"  ✗ Failed to copy {base_name}: {e}")
        
        # Print final statistics
        print(f"\n" + "="*50)
        print(f"REVERSE CONVERSION COMPLETED")
        print(f"="*50)
        print(f"Total .lb files processed: {len(lb_files)}")
        print(f"Successfully converted: {len(converted_files)}")
        print(f"Total points: {total_points}")
        print(f"Edge points (0): {total_edge_points} ({total_edge_points/total_points*100:.2f}%)")
        print(f"Non-edge points (1): {total_non_edge_points} ({total_non_edge_points/total_points*100:.2f}%)")
        print(f"Non-annotated (-1): {total_non_annotated} ({total_non_annotated/total_points*100:.2f}%)")
        print(f"Output directory: {target_dir}")


def main():
    """
    Main function - automatically processes the target directory.
    """
    print("="*50)
    print("ABC to PLY Label Format Converter (REVERSE)")
    print("="*50)
    print("Target directory: /home/cle/data/dtu_results_pc/IML_scan105/lb")
    print("Reverse conversion: -1 → -1 (non-annotated), 1 → 0 (edge), 0 → 1 (non-edge)")
    print("="*50)
    
    # Ask for confirmation
    response = input("This will replace ALL .lb files in the target directory with original format files. Continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    # Process the directory
    process_directory()
    
    print("\nReverse conversion process completed!")


if __name__ == "__main__":
    main()