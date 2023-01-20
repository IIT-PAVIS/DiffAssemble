# Visualize in 3D

Tested with blender(4.0), needs numpy as dependncy (should already be included in blender's internal packaging)

## Usage

1. Launch Blender
2. load 3d_viz.blender (include glass materials, and simple camera and light setup)
3. In blender go to scripting tab and load **blender_script.py**
4. Inside the change the dp in line 6 to point to a folder where we save the ply and npz (extracted with train_3d script, setting the evaluate to true)
5. Run the script
6. The script will load the mesh files in blender and set the animation keyframes to follow the inference
7. Render the video
