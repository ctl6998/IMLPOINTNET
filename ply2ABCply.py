import struct
import datetime

def convert_ply_ascii_to_binary(input_file, output_file):
    """
    Convert PLY file from ASCII format with RGB to binary format without RGB.
    
    Args:
        input_file (str): Path to input ASCII PLY file
        output_file (str): Path to output binary PLY file
    """
    
    vertices = []
    
    # Read the ASCII PLY file
    with open(input_file, 'r') as f:
        # Skip header until we reach vertex data
        line = f.readline().strip()
        vertex_count = 0
        
        while line != 'end_header':
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            line = f.readline().strip()
        
        # Read vertex data
        for i in range(vertex_count):
            line = f.readline().strip()
            if not line:
                break
                
            parts = line.split()
            if len(parts) >= 9:  # x, y, z, r, g, b, nx, ny, nz
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                # Skip RGB values (parts[3], parts[4], parts[5])
                nx, ny, nz = float(parts[6]), float(parts[7]), float(parts[8])
                
                # Flip the model right-side up (most common fix)
                # Option 1: Flip Y axis (try this first)
                # y = -y
                # ny = -ny
                
                # Option 2: Flip Z axis (uncomment if Option 1 doesn't work)
                z = -z
                nz = -nz
                
                # Option 3: Flip both Y and Z (uncomment if needed)
                # y = -y
                # z = -z
                # ny = -ny
                # nz = -nz
                
                vertices.append((x, y, z, nx, ny, nz))
    
    # Write binary PLY file
    with open(output_file, 'wb') as f:
        # Write header
        header = f"""ply
format binary_little_endian 1.0
comment Created by meshio v5.0.5, {datetime.datetime.now().isoformat()}
element vertex {len(vertices)}
property float x
property float y
property float z
property float nx
property float ny
property float nz
end_header
"""
        f.write(header.encode('ascii'))
        
        # Write vertex data in binary format
        for vertex in vertices:
            # Pack as little-endian floats (6 floats per vertex)
            packed_data = struct.pack('<6f', *vertex)
            f.write(packed_data)
    
    print(f"Converted {len(vertices)} vertices from {input_file} to {output_file}")
    print(f"Input format: ASCII with RGB colors")
    print(f"Output format: Binary little-endian without RGB colors")

def read_ply_header(filename):
    """
    Read and display PLY header information for verification.
    
    Args:
        filename (str): Path to PLY file
    """
    print(f"\nHeader information for {filename}:")
    print("-" * 50)
    
    with open(filename, 'rb') as f:
        line = f.readline().decode('ascii').strip()
        while line != 'end_header':
            print(line)
            line = f.readline().decode('ascii').strip()
        print(line)  # Print 'end_header'

# Example usage
if __name__ == "__main__":
    FILE_NUM = 105
    input_file = f"/home/cle/data/dtu_results_pc/scan{FILE_NUM}.ply"      # Your ASCII PLY file
    output_file = f"/home/cle/data/dtu_results_pc/IML_scan{FILE_NUM}/ply/scan{FILE_NUM}.ply"    # Output binary PLY file
    
    # Convert the file
    convert_ply_ascii_to_binary(input_file, output_file)
    
    # Optionally verify headers
    print("\nVerifying conversion:")
    read_ply_header(input_file)
    read_ply_header(output_file)