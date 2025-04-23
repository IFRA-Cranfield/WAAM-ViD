# Import necessary libraries
import numpy as np
import open3d as o3d  # For 3D point cloud processing
import cv2  # For video processing
import pandas as pd  # For reading/writing CSV data
import imagehash  # For perceptual image hashing
from PIL import Image  # For image operations with imagehash

# Load the point cloud data from .npy file
file_path = r"C:\SUBSTRATE_PATH_HERE"
file_name = "SUBSTRATE_NAME_HERE"
point_cloud = np.load(file_path + "\\" + file_name)

print(file_name)
print("Shape of point cloud:", point_cloud.shape)

# Load associated CSV data
csv_path = "C:\\CSV_PATH_HERE"
df = pd.read_csv(csv_path)

# Define paths to processed and raw videos
video_path = r"C:\VIDEO_FOLDER_PATH_HERE"
raw_video_path = r"C:\RAW_VIDEO_FOLDER_PATH_HERE"

# Set key parameters for the specific video / weld seam
index = 90 # Video number
x_shift = 205 # Units - displasment in x 
y_shift = 60 # Units - displasment in y 
end_ofset = 4 # Units - distance from weld seam end to final welder position 
start_ofset = 2.5 # Units - distance from weld start end to begining welder position 
speed =  df.loc[index-1, 'Travel_speed']
actual_distance = 70 # mm - distance of weld seam
video_name = df.loc[index-1, 'Video_filepath']
print(video_name)
min_z_ = -231.2 # lowerbound height of weld seam
max_z_ = -200 # upperbound height of weld seam
window = 50 # window size for frame comparison set 0 for infinate 


#%% Global analysis of full point cloud

# Extract x, y, z coordinates
x_ = point_cloud[:, 0]
y_ = point_cloud[:, 1]
z_ = point_cloud[:, 2]

# Apply Z-filter to isolate region of interest
mask = (z_ >= -235) & (z_ <= -200)
filtered_point_cloud = point_cloud[mask]

# Convert filtered point cloud to Open3D format
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(filtered_point_cloud[:, :3])

# Create and translate coordinate frame
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
coord_frame.translate(pcd1.get_min_bound())

# Get and color bounding box
aabb = pcd1.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)

# Visualize
o3d.visualization.draw_geometries([pcd1, aabb, coord_frame])

# Extract physical dimensions from bounding box
min_bound = aabb.get_min_bound()
max_bound = aabb.get_max_bound()
width  = max_bound[0] - min_bound[0]
length = max_bound[1] - min_bound[1]
height = max_bound[2] - min_bound[2]

# Print measured and actual dimensions
print(f"cloud width: {width:.2f} Units")
print(f"cloud length: {length:.2f} Units")
actual_width = 166
actual_length = 325
print(f"Actual width: {actual_width} mm")
print(f"Actual length: {actual_length} mm")

# Calculate scaling factors
scale_x = actual_width / width
scale_y = actual_length / length

#%% Local ROI analysis

# Define region of interest in point cloud
ROI = (x_ >= 0 + x_shift) & (x_ <= 15 + x_shift) & (y_ >= 0 + y_shift) & (y_ <= 80 + y_shift) & (z_ >= min_z_) & (z_ <= max_z_)
ROI_point_cloud = point_cloud[ROI]

# Convert and clean point cloud
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(ROI_point_cloud[:, :3])
ROI_point_cloud, ind = pcd2.remove_radius_outlier(nb_points=30, radius=1.5)
pcd2 = pcd2.select_by_index(ind)
ROI_point_cloud = np.asarray(pcd2.points)

# Coordinate frame and bounding box
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)
coord_frame.translate(pcd2.get_min_bound()-np.array([0, 2, 0]))
aabb = pcd2.get_axis_aligned_bounding_box()

# Shrink bounding box to welding segment
min_bound = aabb.get_min_bound()
max_bound = aabb.get_max_bound()
min_shrunk = min_bound + np.array([0, end_ofset, 0])
max_shrunk = max_bound - np.array([0, start_ofset, 0])
shrunken_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_shrunk, max_bound=max_shrunk)
shrunken_box.color = (1, 0, 0)

# Visualize ROI
o3d.visualization.draw_geometries([pcd2, coord_frame, shrunken_box])

# Calculate weld segment bounds and scaling
y = ROI_point_cloud[:, 1]
y_min = np.min(y) + end_ofset
y_max = np.max(y) - start_ofset
distance = y_max - y_min
scale_y = actual_distance / distance

print(f"weld start: {y_max:.2f} Units")
print(f"weld end: {y_min:.2f} Units")
print(f"point cloud weld length: {distance:.2f} Units")
print(f"actual weld length: {actual_distance:.2f} mm")
print(f"length scale: {scale_y:.2f}")




#%% Corrupted frame hashing and mapping

def get_blank_frame_hash(resolution=(640, 480)):
    """Generate a hash for a blank gray image (for fallback)."""
    blank = Image.new("RGB", resolution, color=(128, 128, 128))
    return imagehash.phash(blank)

def extract_frame_hashes(video_path):
    """Extract perceptual hashes from video frames."""
    print("\n" + video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames: {total_frames}")
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    hashes = []
    frame_idx = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fallback_hash = get_blank_frame_hash((width, height))
    f_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            hashes.append((frame_idx, fallback_hash))
            f_i += 1
        else:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_hash = imagehash.phash(pil_img)
                hashes.append((frame_idx, frame_hash))
            except Exception:
                hashes.append((frame_idx, fallback_hash))
                f_i += 1
        frame_idx += 1
        if frame_idx >= total_frames:
            break

    cap.release()
    print(f"recorded frames: {len(hashes)}")
    print(f"corrupted frames: {f_i}")
    return hashes

def map_frames_sequential(original_hashes, reduced_hashes):
    """Map frames from reduced video to original using hash similarity."""
    print("mapping")
    mapping = []
    start_idx = 0

    for i, (red_idx, red_hash) in enumerate(reduced_hashes):
        best_match_idx = None
        best_distance = float('inf')
        end_idx = len(original_hashes) if window == 0 else min(start_idx + window, len(original_hashes))
        search_range = original_hashes[start_idx:end_idx]

        for orig_idx, orig_hash in search_range:
            dist = red_hash - orig_hash
            if dist < best_distance:
                best_distance = dist
                best_match_idx = orig_idx
            if dist == 0:
                break

        if best_match_idx is not None:
            mapping.append((red_idx, best_match_idx))
            start_idx = best_match_idx + 1

    print(len(mapping))
    print(orig_idx)
    return mapping

# Define and adjust video paths
video_full = video_path + "\\" + video_name
raw_video_full = raw_video_path + "\\" + video_name.removeprefix("video\\")
if index % 10 != 10:
    raw_video_full = raw_video_full.replace(str(index) + ".mp4", str(index % 10) + ".avi")
else:
    raw_video_full = raw_video_full.replace(str(index) + ".mp4", "10.avi")

# Extract and match frame hashes
original_hashes = extract_frame_hashes(raw_video_full)
reduced_hashes = extract_frame_hashes(video_full)
frame_mapping = map_frames_sequential(original_hashes, reduced_hashes)

#%% Frame-by-frame weld width measurement

cap1 = cv2.VideoCapture(video_full)
fps = cap1.get(cv2.CAP_PROP_FPS)
spf = 1 / fps
total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
cap1.release()

print(f"Frame rate: {fps} FPS")
print(f"travel speed: {speed} mm/s")

# Measure weld width frame by frame
mesure_slice = y_min
mesure_width = []
tolerance = 0.1
y__ROI = ROI_point_cloud[:, 1]
frame_index = total_frames - 1
frame_distance_array = []

while mesure_slice <= y_max and frame_index > 0:    
    slice_mask = (y__ROI >= mesure_slice - tolerance) & (y__ROI <= mesure_slice + tolerance)
    slice_points = ROI_point_cloud[slice_mask]
    try:
        slice_x_min = np.min(slice_points[:, 0])
        slice_x_max = np.max(slice_points[:, 0])
        slice_width = (slice_x_max - slice_x_min) * scale_x
        mesure_width.append(round(slice_width, 1))
    except:
        raise ValueError(mesure_slice)
        break

    current_frame, current_Tframe = frame_mapping[frame_index]
    next_frame, next_Tframe = frame_mapping[frame_index - 1]
    time_diff = (current_Tframe - next_Tframe) * spf
    frame_distance = speed * time_diff
    frame_distance_array.append((frame_distance / scale_y))
    mesure_slice += (frame_distance / scale_y)
    frame_index -= 1

# Fill in missing frames
frame_difference = total_frames - len(mesure_width)
print(f"Total number of frames: {total_frames}")
print(f"measured frames: {len(mesure_width)}")
print(f"difference in frames: {frame_difference}, time: {frame_difference/fps:.2f} seconds")

for i in range(frame_difference):
    mesure_width.append(0)

mesure_width.reverse()

#%% Save results to CSV

df.at[index-1, 'Width'] = str(mesure_width)
df.to_csv(csv_path, index=False)
print("saved")
