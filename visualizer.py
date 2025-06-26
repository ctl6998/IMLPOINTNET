import open3d as o3d
import numpy as np
import json

# Configuration - Only need to change this section
DATA_PATH = "/home/cle/data/dtu_results_pc/IML_scan24/"
PLY_FILE = DATA_PATH + "ply/scan24.ply"
ANNOTATION_FILE = DATA_PATH + "annotation_2.ply"
FEEDBACK_FILE = DATA_PATH + "feedback_2_512b_lr01.ply"

# Camera configuration from JSON
CAMERA_CONFIG = {
    "boundingbox_max": [ 0.45399200916290283, 0.69798702001571655, 0.58256900310516357 ],
    "boundingbox_min": [ -0.46335700154304504, -0.60707098245620728, -0.58195799589157104 ],
    "field_of_view": 59.999999999999993,
    "front": [ 0.89193081192149037, 0.048073342695748505, 0.44960914188562501 ],
    "lookat": [ -0.015990312617361824, 0.028779729434574807, 0.034514664967778746 ],
    "up": [ -0.36037335101870605, -0.52500286197556045, 0.77104023422453083 ],
    "zoom": 0.50000000000000011
}

def visualize_point_cloud_with_labels():
    try:
        # Load the main point cloud
        print("Loading point cloud...")
        pcd = o3d.io.read_point_cloud(PLY_FILE)
        num_points = len(pcd.points)
        print(f"Loaded {num_points} points from {PLY_FILE}")
        
        if num_points == 0:
            print("Error: Point cloud is empty")
            return
        
        # Read annotation labels
        print("Reading annotation labels...")
        with open(ANNOTATION_FILE, 'r') as f:
            annotation_labels = [int(line.strip()) for line in f.readlines()]
        
        # Read feedback labels
        print("Reading feedback labels...")
        with open(FEEDBACK_FILE, 'r') as f:
            feedback_labels = [int(line.strip()) for line in f.readlines()]
        
        print(f"Annotation labels: {len(annotation_labels)}")
        print(f"Feedback labels: {len(feedback_labels)}")
        print(f"Point cloud points: {num_points}")
        
        # Check if dimensions match
        if len(annotation_labels) != num_points:
            print(f"Warning: Annotation labels ({len(annotation_labels)}) don't match point count ({num_points})")
        if len(feedback_labels) != num_points:
            print(f"Warning: Feedback labels ({len(feedback_labels)}) don't match point count ({num_points})")
        
        # Create color array - default to gray
        colors = np.full((num_points, 3), [0.5, 0.5, 0.5])  # Gray for default/unprocessed
        
        # Color points based on labels
        min_len = min(num_points, len(annotation_labels), len(feedback_labels))
        
        print(f"Label scheme:")
        print(f"Annotation: 0=edge, 1=non-edge, -1=other")
        print(f"Feedback: 0=edge, -1=other/non-edge")
        
        # Debug: Let's check the first 20 points to see what's happening
        print(f"\nDebugging first 20 points:")
        print("Index | Ann | FB | Color")
        print("-" * 25)
        
        color_counts = {"yellow": 0, "red": 0, "green": 0, "blue": 0, "gray": 0}
        
        for i in range(min_len):
            ann_label = annotation_labels[i]
            fb_label = feedback_labels[i]
            
            # Debug first 20 points
            if i < 20:
                color_name = "unknown"
                if ann_label == 0 and fb_label == 0:
                    color_name = "yellow"  # Both edge
                elif ann_label == 0 and fb_label == -1:
                    color_name = "red"     # Annotation edge only
                elif fb_label == 0:
                    color_name = "green"   # Feedback edge (regardless of annotation)
                elif ann_label == 1:
                    color_name = "blue"    # Annotation non-edge
                else:
                    color_name = "gray"    # Both undefined
                print(f"{i:5d} | {ann_label:3d} | {fb_label:2d} | {color_name}")
            
            # Simplified coloring scheme with priority order:
            if ann_label == 0 and fb_label == 0:
                colors[i] = [1.0, 1.0, 0.0]  # Yellow - both edge
                color_counts["yellow"] += 1
            elif ann_label == 0 and fb_label == -1:
                colors[i] = [1.0, 0.0, 0.0]  # Red - annotation edge only
                color_counts["red"] += 1
            elif fb_label == 0:
                colors[i] = [0.0, 1.0, 0.0]  # Green - feedback edge (any annotation except handled above)
                color_counts["green"] += 1
            elif ann_label == 1:
                colors[i] = [0.0, 0.0, 1.0]  # Blue - annotation non-edge
                color_counts["blue"] += 1
            else:
                colors[i] = [0.5, 0.5, 0.5]  # Gray - both undefined
                color_counts["gray"] += 1
        
        print(f"\nColor assignment counts:")
        for color, count in color_counts.items():
            print(f"{color}: {count}")
        
        # Check specifically for feedback=0 cases
        fb_zero_count = sum(1 for label in feedback_labels[:min_len] if label == 0)
        print(f"\nTotal feedback=0 (edge) points: {fb_zero_count}")
        
        ann_zero_count = sum(1 for label in annotation_labels[:min_len] if label == 0)
        ann_one_count = sum(1 for label in annotation_labels[:min_len] if label == 1)
        print(f"Total annotation=0 (edge) points: {ann_zero_count}")
        print(f"Total annotation=1 (non-edge) points: {ann_one_count}")
        
        # Apply colors to point cloud
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Print statistics
        both_edge = sum(1 for i in range(min_len) if annotation_labels[i] == 0 and feedback_labels[i] == 0)
        ann_edge_only = sum(1 for i in range(min_len) if annotation_labels[i] == 0 and feedback_labels[i] == -1)
        feedback_edge = sum(1 for i in range(min_len) if feedback_labels[i] == 0 and not (annotation_labels[i] == 0))
        ann_non_edge = sum(1 for i in range(min_len) if annotation_labels[i] == 1)
        both_undefined = sum(1 for i in range(min_len) if annotation_labels[i] == -1 and feedback_labels[i] == -1)
        
        print(f"\nStatistics:")
        print(f"Both edges (yellow): {both_edge}")
        print(f"Annotation edges only (red): {ann_edge_only}")
        print(f"Feedback edges (green): {feedback_edge}")
        print(f"Annotation non-edges (blue): {ann_non_edge}")
        print(f"Both undefined (gray): {both_undefined}")
        print(f"Total processed: {both_edge + ann_edge_only + feedback_edge + ann_non_edge + both_undefined}")
        
        # Display legend
        print(f"\nColor Legend:")
        print(f"🟡 Yellow: Both annotation and feedback define as edges")
        print(f"🔴 Red: Only annotation define as edge")
        print(f"🟢 Green: Feedback define as edges")
        print(f"🔵 Blue: Annotation non-edge")
        print(f"⚪ Gray: Both undefined")
        
        # Create visualizer with custom camera
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud with Edge Labels", width=1200, height=800)
        
        # Add point cloud to visualizer
        vis.add_geometry(pcd)
        
        # Set camera parameters
        view_control = vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        
        # Set camera parameters from config
        camera_params.extrinsic = np.array([
            [0.73153823, -0.37018477, 0.64911765, CAMERA_CONFIG["lookat"][0]],
            [0.20856196, -0.67800899, 0.20856196, CAMERA_CONFIG["lookat"][1]],
            [0.64911765, 0.63503310, 0.40000000, CAMERA_CONFIG["lookat"][2]],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Apply camera configuration
        view_control.convert_from_pinhole_camera_parameters(camera_params)
        view_control.set_front(CAMERA_CONFIG["front"])
        view_control.set_lookat(CAMERA_CONFIG["lookat"])
        view_control.set_up(CAMERA_CONFIG["up"])
        view_control.set_zoom(CAMERA_CONFIG["zoom"])
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
        
        print("\nDisplaying point cloud with fixed camera...")
        print("Controls:")
        print("- Mouse: Rotate view")
        print("- Mouse wheel: Zoom")
        print("- Ctrl + Mouse: Pan")
        print("- Press 'Q' or close window to exit")
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")

def analyze_labels():
    """Analyze the label files to understand the data better"""
    try:
        # Read annotation labels
        with open(ANNOTATION_FILE, 'r') as f:
            annotation_labels = [int(line.strip()) for line in f.readlines()]
        
        # Read feedback labels  
        with open(FEEDBACK_FILE, 'r') as f:
            feedback_labels = [int(line.strip()) for line in f.readlines()]
        
        print("Label Analysis:")
        print(f"Annotation file: {len(annotation_labels)} labels")
        print(f"Feedback file: {len(feedback_labels)} labels")
        
        # Count unique values
        ann_unique = np.unique(annotation_labels, return_counts=True)
        fb_unique = np.unique(feedback_labels, return_counts=True)
        
        print(f"\nAnnotation labels distribution:")
        for val, count in zip(ann_unique[0], ann_unique[1]):
            print(f"  {val}: {count} points")
            
        print(f"\nFeedback labels distribution:")
        for val, count in zip(fb_unique[0], fb_unique[1]):
            print(f"  {val}: {count} points")
            
        # Cross-tabulation
        if len(annotation_labels) == len(feedback_labels):
            print(f"\nCross-tabulation (Annotation vs Feedback):")
            min_len = min(len(annotation_labels), len(feedback_labels))
            
            # Create a mapping of all combinations
            combinations = {}
            for i in range(min_len):
                combo = (annotation_labels[i], feedback_labels[i])
                combinations[combo] = combinations.get(combo, 0) + 1
            
            print("  (Annotation, Feedback): Count")
            for combo, count in sorted(combinations.items()):
                print(f"  {combo}: {count}")
            
            return annotation_labels, feedback_labels, combinations
            
    except Exception as e:
        print(f"Error analyzing labels: {e}")
        return None, None, None

if __name__ == "__main__":
    print(f"Configuration:")
    print(f"Data path: {DATA_PATH}")
    print(f"PLY file: {PLY_FILE}")
    print(f"Annotation file: {ANNOTATION_FILE}")
    print(f"Feedback file: {FEEDBACK_FILE}")
    print("-" * 50)
    
    # First analyze the labels to understand what values we have
    ann_labels, fb_labels, combinations = analyze_labels()
    
    if combinations:
        print(f"\nDetected label combinations: {list(combinations.keys())}")
        
        # Check specifically for green case (1,0)
        if (1, 0) in combinations:
            print(f"Green points available: {combinations[(1, 0)]} points where annotation=1 and feedback=0")
        else:
            print("No green points found: No cases where annotation=1 (non-edge) and feedback=0 (edge)")
            print("This means the feedback model never detected edges that annotation labeled as non-edge")
        
    print("\nStarting visualization...")
    visualize_point_cloud_with_labels()