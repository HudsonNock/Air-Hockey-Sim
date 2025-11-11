import pandas as pd
import cv2
import numpy as np

# Configuration
csv_file = 'system_loop_data_N2.csv' # Replace with your CSV filename
mallet_r = 0.1011 / 2
puck_r = 0.0629 / 2

# Playing field dimensions (meters)
field_width = 1.993
field_height = 0.992

# Pixel ratio (pixels per meter)
px_per_meter = 500

# Calculate screen dimensions
screen_width = int(field_width * px_per_meter)
screen_height = int(field_height * px_per_meter)

# Convert radii to pixels
puck_r_px = int(puck_r * px_per_meter)
mallet_r_px = int(mallet_r * px_per_meter)

# Load CSV data
df = pd.read_csv(csv_file)

# Calculate average time between frames for FPS
if len(df) > 1:
    avg_dt = df['dt'].mean()
    print("A")
    print(avg_dt)
    print(np.max(df['dt']))
    calculated_fps = int(1.0 / avg_dt) if avg_dt > 0 else 30
else:
    calculated_fps = 30

print(f"Average dt: {avg_dt:.4f}s")
print(f"Calculated FPS: {calculated_fps}")
print(f"Screen dimensions: {screen_width}x{screen_height}")
print(f"Total frames: {len(df)}")

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter("gameplay_new2.avi", fourcc, calculated_fps, (screen_width, screen_height))

frame_count = 0
prev_mx = None
prev_my = None

for idx, row in df.iterrows():
    if idx < 7500:
        continue
    # Create blank image (dark green background for air hockey table)
    img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    img[:] = (0, 100, 0)  # Green background
    
    # Draw center line
    cv2.line(img, (screen_width // 2, 0), (screen_width // 2, screen_height), (255, 255, 255), 2)
    
    # Draw center circle
    center_x = screen_width // 2
    center_y = screen_height // 2
    cv2.circle(img, (center_x, center_y), int(0.15 * px_per_meter), (255, 255, 255), 2)
    
    # Convert puck coordinates from meters to pixels (current frame)
    px_x = int(row['Px'] * px_per_meter)
    px_y = int((field_height - row['Py']) * px_per_meter)  # Flip Y
    
    # Use previous frame's mallet position (to sync with puck)
    if prev_mx is not None and prev_my is not None:
        mx_x = int(prev_mx * px_per_meter)
        mx_y = int((field_height - prev_my) * px_per_meter)  # Flip Y
    else:
        # For first frame, use current mallet position
        mx_x = int(row['Mx'] * px_per_meter)
        mx_y = int((field_height - row['My']) * px_per_meter)
    
    # Store current mallet position for next frame
    prev_mx = row['Mx']
    prev_my = row['My']
    
    # Draw puck (white)
    cv2.circle(img, (px_x, px_y), puck_r_px, (255, 255, 255), -1)
    
    # Draw mallet (red)
    cv2.circle(img, (mx_x, mx_y), mallet_r_px, (0, 0, 255), -1)
    
    # Add frame info
    cv2.putText(img, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Time: {row['dt']:.3f}s", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display frame
    cv2.imshow('Air Hockey', img)
    
    # Write to video
    out.write(img)
    
    # Wait for key press (1ms delay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1
    
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

    #if frame_count == 6000:
    #    out.release()
    #    cv2.destroyAllWindows()
    #    break

# Release resources

print(f"\nVideo saved as 'gameplay.avi'")
print(f"Total frames processed: {frame_count}")