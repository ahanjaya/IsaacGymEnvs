import numpy as np
from pybullet_utils import transformations
import json
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <motion-fn>")
    sys.exit(1)

motion_fn = sys.argv[1]

print(motion_fn)

with open(motion_fn, "r") as f:
    motion_data = json.load(f)

original_frames = motion_data["Frames"]
filtered_frames = []

for frame in original_frames:
    quat = frame[3:7]

    roll, pitch, yaw = transformations.euler_from_quaternion(quat)
    filtered_quat = transformations.quaternion_from_euler(roll, pitch, 0.0)
    frame[3:7] = filtered_quat
    frame[1] = 0.0

    filtered_frames.append(frame)

motion_data["Frames"] = filtered_frames

with open(motion_fn.split(".txt")[0] + "_remove_yaw.txt", "w") as f:
    json.dump(motion_data, f)

