import open3d as o3d
import numpy as np
import json
import os

# Configuration - Only need to change this section
FILE_NUM = 24
DIALOGUE_NUM = 2
DESCRIPTOR_NUM = 16
NUM_DESCRIPTOR = 20
DATA_PATH = f"/home/cle/data/dtu_results_pc/IML_scan{FILE_NUM}/IML_Loop/scenario_3/"

# PLY_FILE = f"/home/cle/data/dtu_results_pc/IML_scan{FILE_NUM}/ply/scan{FILE_NUM}.ply"
PLY_FILE = f"/home/cle/data/dtu_results_pc/scan{FILE_NUM}.ply"
ANNOTATION_FILE = DATA_PATH + f"annotation_{DIALOGUE_NUM}.ply"
FEEDBACK_FILE = DATA_PATH + f"feedback_{DIALOGUE_NUM}_512b_lr01_{DESCRIPTOR_NUM}f_scale8.ply"

# View options
SHOW_ANNOTATION = False   # Set to False to hide annotation labels
SHOW_FEEDBACK = False    # Set to False to hide feedback labels

# Sreenshot
AUTO_SCREENSHOT = True
SCREENSHOT_PATH = "/home/cle/Work/ABC-Challenge/IMLPointNet/DTU_original" #DATA_PATH
SCREENSHOT_NAME = f"scan_{FILE_NUM}_original.png"

#SCREENSHOT_PATH = "/home/cle/data/dtu_results_pc/IML_scan24/IML_Loop/scenario_3" #DATA_PATH
#SCREENSHOT_NAME = f"{DIALOGUE_NUM}.feedback_{DIALOGUE_NUM}_512b_lr01_{DESCRIPTOR_NUM}f_scale8.png"
#SCREENSHOT_NAME = f"{DIALOGUE_NUM}.annotation_{DIALOGUE_NUM}_512b_lr01.png"

# Flip
FLIP_HORIZONTAL = False # Mimic RollingDot View

# Camera configuration from JSON
CAMERA_CONFIG = {
			"boundingbox_max" : [ 0.45399200916290283, 0.69798702001571655, 0.58256900310516357 ],
			"boundingbox_min" : [ -0.46335700154304504, -0.60707098245620728, -0.58195799589157104 ],
			"field_of_view" : 59.999999999999993,
			"front" : [ 0.89193081192149037, 0.048073342695748505, 0.44960914188562501 ],
			"lookat" : [ -0.015990312617361824, 0.028779729434574807, 0.034514664967778746 ],
			"up" : [ -0.36037335101870605, -0.52500286197556045, 0.77104023422453083 ],
			"zoom" : 0.56000000000000016
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
        
        # Apply horizontal flip if enabled
        if FLIP_HORIZONTAL:
            print("Applying horizontal flip (mirror effect)...")
            points = np.asarray(pcd.points)
            points[:, 0] = -points[:, 0]  # Negate X coordinates
            pcd.points = o3d.utility.Vector3dVector(points)
            print("✓ Point cloud flipped horizontally")
        
        # Read annotation labels (if enabled)
        annotation_labels = []
        if SHOW_ANNOTATION:
            print("Reading annotation labels...")
            with open(ANNOTATION_FILE, 'r') as f:
                annotation_labels = [int(line.strip()) for line in f.readlines()]
        else:
            print("Annotation labels disabled - using default values")
            annotation_labels = [-1] * num_points  # All undefined
        
        # Read feedback labels (if enabled)
        feedback_labels = []
        if SHOW_FEEDBACK:
            print("Reading feedback labels...")
            with open(FEEDBACK_FILE, 'r') as f:
                feedback_labels = [int(line.strip()) for line in f.readlines()]
        else:
            print("Feedback labels disabled - using default values")
            feedback_labels = [-1] * num_points  # All undefined
        
        print(f"Annotation labels: {len(annotation_labels)} {'(enabled)' if SHOW_ANNOTATION else '(disabled)'}")
        print(f"Feedback labels: {len(feedback_labels)} {'(enabled)' if SHOW_FEEDBACK else '(disabled)'}")
        print(f"Point cloud points: {num_points}")
        
        # Check if dimensions match (only for enabled files)
        if SHOW_ANNOTATION and len(annotation_labels) != num_points:
            print(f"Warning: Annotation labels ({len(annotation_labels)}) don't match point count ({num_points})")
        if SHOW_FEEDBACK and len(feedback_labels) != num_points:
            print(f"Warning: Feedback labels ({len(feedback_labels)}) don't match point count ({num_points})")
        
        # Create color array - default to gray
        colors = np.full((num_points, 3), [0.5, 0.5, 0.5])  # Gray for default/unprocessed
        
        # Color points based on labels
        min_len = min(num_points, len(annotation_labels), len(feedback_labels))
        
        print(f"Label scheme:")
        print(f"Annotation: {'0=edge, 1=non-edge, -1=other' if SHOW_ANNOTATION else 'DISABLED (all -1)'}")
        print(f"Feedback: {'0=edge, -1=other/non-edge' if SHOW_FEEDBACK else 'DISABLED (all -1)'}")
        
        # Adjust coloring scheme based on what's enabled
        if not SHOW_ANNOTATION and not SHOW_FEEDBACK:
            print("Both annotation and feedback disabled - showing all points in gray")
        elif not SHOW_ANNOTATION:
            print("Annotation disabled - showing only feedback edges (green) vs gray")
        elif not SHOW_FEEDBACK:
            print("Feedback disabled - showing only annotation edges (red) and non-edges (blue) vs gray")
        
        # Debug: Let's check the first 20 points to see what's happening
        # print(f"\nDebugging first 20 points:")
        # print("Index | Ann | FB | Color")
        # print("-" * 25)
        
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
                
                # Show status for disabled files
                ann_status = "DISABLED" if not SHOW_ANNOTATION else str(ann_label)
                fb_status = "DISABLED" if not SHOW_FEEDBACK else str(fb_label)
                # print(f"{i:5d} | {ann_status:>8} | {fb_status:>8} | {color_name}")
            
            # Simplified coloring scheme with priority order:
            if SHOW_ANNOTATION and SHOW_FEEDBACK:
                # Both enabled - full color scheme
                if ann_label == 0 and fb_label == 0:
                    colors[i] = [1.0, 1.0, 0.0]  # Yellow - both edge
                    color_counts["yellow"] += 1
                elif ann_label == 0 and fb_label == -1:
                    colors[i] = [1.0, 0.0, 0.0]  # Red - annotation edge only
                    color_counts["red"] += 1
                elif fb_label == 0:
                    colors[i] = [0.0, 1.0, 0.0]  # Green - feedback edge
                    color_counts["green"] += 1
                elif ann_label == 1:
                    colors[i] = [0.0, 0.0, 1.0]  # Blue - annotation non-edge
                    color_counts["blue"] += 1
                else:
                    colors[i] = [0.5, 0.5, 0.5]  # Gray - both undefined
                    color_counts["gray"] += 1
            elif SHOW_FEEDBACK and not SHOW_ANNOTATION:
                # Only feedback enabled
                if fb_label == 0:
                    colors[i] = [0.0, 1.0, 0.0]  # Green - feedback edge
                    color_counts["green"] += 1
                else:
                    colors[i] = [0.5, 0.5, 0.5]  # Gray - feedback non-edge/undefined
                    color_counts["gray"] += 1
            elif SHOW_ANNOTATION and not SHOW_FEEDBACK:
                # Only annotation enabled
                if ann_label == 0:
                    colors[i] = [1.0, 0.0, 0.0]  # Red - annotation edge
                    color_counts["red"] += 1
                elif ann_label == 1:
                    colors[i] = [0.0, 0.0, 1.0]  # Blue - annotation non-edge
                    color_counts["blue"] += 1
                else:
                    colors[i] = [0.5, 0.5, 0.5]  # Gray - annotation undefined
                    color_counts["gray"] += 1
            else:
                # Both disabled
                colors[i] = [0.5, 0.5, 0.5]  # Gray - all points
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
        print(f"🟡 Yellow: Both annotation and feedback edges")
        print(f"🔴 Red: Annotation edge only")
        print(f"🟢 Green: Feedback edges")
        print(f"🔵 Blue: Annotation non-edge")
        print(f"⚪ Gray: Both undefined")
        
        # Create visualizer with custom camera
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud with Edge Labels", width=1200, height=800, visible=False)
        
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
        
        # Take screenshot if enabled
        if AUTO_SCREENSHOT:
            # Ensure screenshot directory exists
            os.makedirs(SCREENSHOT_PATH, exist_ok=True)
            screenshot_full_path = os.path.join(SCREENSHOT_PATH, SCREENSHOT_NAME)
            
            print(f"\nTaking screenshot...")
            print(f"Screenshot will be saved to: {screenshot_full_path}")
            
            # Update the display to ensure everything is rendered
            vis.poll_events()
            vis.update_renderer()
            
            # Capture screenshot
            success = vis.capture_screen_image(screenshot_full_path)
            if success:
                print(f"✓ Screenshot saved successfully: {screenshot_full_path}")
            else:
                print(f"✗ Failed to save screenshot to: {screenshot_full_path}")
        
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
        # Read annotation labels (if enabled)
        annotation_labels = []
        if SHOW_ANNOTATION:
            with open(ANNOTATION_FILE, 'r') as f:
                annotation_labels = [int(line.strip()) for line in f.readlines()]
        else:
            annotation_labels = [-1] * 1000  # Dummy data for analysis
        
        # Read feedback labels (if enabled)
        feedback_labels = []
        if SHOW_FEEDBACK:
            with open(FEEDBACK_FILE, 'r') as f:
                feedback_labels = [int(line.strip()) for line in f.readlines()]
        else:
            feedback_labels = [-1] * 1000  # Dummy data for analysis
        
        print("Label Analysis:")
        print(f"Annotation file: {len(annotation_labels)} labels {'(enabled)' if SHOW_ANNOTATION else '(disabled)'}")
        print(f"Feedback file: {len(feedback_labels)} labels {'(enabled)' if SHOW_FEEDBACK else '(disabled)'}")
        
        if not SHOW_ANNOTATION and not SHOW_FEEDBACK:
            print("Both files disabled - skipping detailed analysis")
            return None, None, None
        
        # Count unique values (only for enabled files)
        if SHOW_ANNOTATION:
            ann_unique = np.unique(annotation_labels, return_counts=True)
            print(f"\nAnnotation labels distribution:")
            for val, count in zip(ann_unique[0], ann_unique[1]):
                print(f"  {val}: {count} points")
        
        if SHOW_FEEDBACK:
            fb_unique = np.unique(feedback_labels, return_counts=True)
            print(f"\nFeedback labels distribution:")
            for val, count in zip(fb_unique[0], fb_unique[1]):
                print(f"  {val}: {count} points")
            
        # Cross-tabulation (only if both are enabled)
        if SHOW_ANNOTATION and SHOW_FEEDBACK and len(annotation_labels) == len(feedback_labels):
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
        else:
            return annotation_labels, feedback_labels, None
            
    except Exception as e:
        print(f"Error analyzing labels: {e}")
        return None, None, None

if __name__ == "__main__":
    print(f"Configuration:")
    print(f"Data path: {DATA_PATH}")
    print(f"PLY file: {PLY_FILE}")
    print(f"Annotation file: {ANNOTATION_FILE} {'(enabled)' if SHOW_ANNOTATION else '(disabled)'}")
    print(f"Feedback file: {FEEDBACK_FILE} {'(enabled)' if SHOW_FEEDBACK else '(disabled)'}")
    print(f"Horizontal flip: {'ENABLED' if FLIP_HORIZONTAL else 'DISABLED'}")
    if AUTO_SCREENSHOT:
        print(f"Auto screenshot: ENABLED -> {os.path.join(SCREENSHOT_PATH, SCREENSHOT_NAME)}")
    else:
        print(f"Auto screenshot: DISABLED")
    print("-" * 50)
    
    # First analyze the labels to understand what values we have
    ann_labels, fb_labels, combinations = analyze_labels()
    
    if combinations and SHOW_ANNOTATION and SHOW_FEEDBACK:
        print(f"\nDetected label combinations: {list(combinations.keys())}")
        
        # Check specifically for green case (1,0)
        if (1, 0) in combinations:
            print(f"Green points available: {combinations[(1, 0)]} points where annotation=1 and feedback=0")
        else:
            print("No green points found: No cases where annotation=1 (non-edge) and feedback=0 (edge)")
            print("This means the feedback model never detected edges that annotation labeled as non-edge")
        
    print("\nStarting visualization...")
    visualize_point_cloud_with_labels()