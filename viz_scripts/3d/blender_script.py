import bpy
from pathlib import Path
import numpy as np

current_scene = bpy.context.scene
dp = "results/3d-breaking-bad-301-kkqlet-backbone:pointnet-category:-max_num_partes:8-architecture-transformer/viz_3d/everyday/DrinkBottle/7d41c6018862dc419d231837a704886d/3-parts/30.105802536010742rot-0.011143021285533905tras-1.0pa-fractured_42"
data_path = Path(dp)
objs = list(data_path.glob("init_*_origin.ply"))


for ob in objs:
    id = int(ob.stem.split("_")[1])
    bpy.ops.wm.ply_import(filepath=str(ob))
    dt = np.load(ob.parent / f"pred_{id}.npy.npz")

    pos = dt["pos"]
    rot = dt["rot"]
    quat = dt["quat"]

    selected_obj = current_scene.objects[ob.stem]
    selected_obj.rotation_mode = "QUATERNION"
    # selected_obj.active_material.name = f"Material.00{id%4+1}"
    mat = bpy.data.materials.get(f"Material.00{(id%4)+1}")

    selected_obj.data.materials.append(mat)

    for i, inverse in enumerate(list(range(pos.shape[0])[::-1])):
        px = pos[i]
        pr = rot[i]
        pr_quat = quat[i]
        # pr_quat=[pr_quat[-1], *pr_quat[:-1]]

        selected_obj.location = px
        selected_obj.rotation_quaternion = pr_quat
        # Set the keyframe with that location, and which frame.
        selected_obj.keyframe_insert(data_path="location", frame=i * 10)
        selected_obj.keyframe_insert(data_path="rotation_quaternion", frame=i * 10)
