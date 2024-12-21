import blenderproc as bproc

import os
import bpy
import json
import random
import imageio
import argparse
import numpy as np

context = bpy.context
scene = context.scene
render = scene.render

bproc.init()
bproc.renderer.set_output_format(file_format="PNG", enable_transparency=True)


render.engine = 'BLENDER_EEVEE' # default is CYCLES for blenderproc, but EEVEE is faster
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

    
def _add_lighting():
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[2] = 1.3
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100
    
    bpy.data.worlds["World"].use_nodes = True
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0.2

def _setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 3, 0)
    bpy.data.cameras["Camera"].lens_unit = "FOV"
    bpy.data.cameras["Camera"].angle = 40 * np.pi / 180

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty


def _sample_camera_loc(phi=None, theta=None, r=3.0):
    '''
    phi: inclination angle
    theta: azimuth angle
    r: radius
    '''
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z], dtype=np.float32)
        
def _load_objs(tree, data_root):
    for node in tree:
        for obj_path in node['objs']:
            objs = bproc.loader.load_obj(f'{data_root}/{obj_path}')
    

def _render_scene(n_imgs=10):
    _setup_camera()
    _add_lighting()
    
    # backface culling
    for material in bpy.data.materials:
        material.use_backface_culling = True
    
    phi_seg = np.linspace(np.pi/3, np.pi/2, n_imgs+1)
    theta_seg = np.linspace(-5*np.pi/6, -np.pi/6, n_imgs+1)
    
    phis = [np.random.uniform(phi_seg[i], phi_seg[i+1]) for i in range(n_imgs)] # angle relative to z-axis
    thetas = [np.random.uniform(theta_seg[i], theta_seg[i+1]) for i in range(n_imgs)] # front view: -np.pi/2
    random.shuffle(phis)
    random.shuffle(thetas)
    
    # sample camera locations
    for i in range(n_imgs):
        r = np.random.uniform(3, 3.5)
        location = _sample_camera_loc(phis[i], thetas[i], r)
        rotation_matrix = bproc.camera.rotation_from_forward_vec([0, 0, 0] - location)
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
    
    data = bproc.renderer.render()
    data.update(
        bproc.renderer.render_segmap(map_by=["instance"])
    )
    
    return data


def _write_imgs(src_dir, data):
    save_img_dir = os.path.join(src_dir, 'imgs')
    os.makedirs(save_img_dir, exist_ok=True)

    # rendered images
    rgbs = data['colors']
    # save images
    for i, rgb in enumerate(rgbs):
        fname = str(i).zfill(2)
        imageio.imwrite(f'{save_img_dir}/{fname}.png', rgb)

    
def render_imgs(src_dir, n_imgs=20, incremental=False):
    if incremental:
        img_dir = os.path.join(src_dir, 'imgs')
        if not os.path.exists(img_dir):
            return
        if len(os.listdir(img_dir)) < n_imgs:
            return

    # load json file
    with open(os.path.join(src_dir, 'object.json'), 'r') as f:
        src = json.load(f)
    # load textured objs into blender (w/ semantic+instance ids)
    _load_objs(src['diffuse_tree'], src_dir)
    # render images
    raw_data = _render_scene(n_imgs)
    # write images
    _write_imgs(src_dir, raw_data)
    
        
if __name__ == '__main__':
    '''
    Script to render images for the specified data.
    
    To run this script, use the following command:
    ```
    blenderproc run scripts/preprocess/render_script.py --data <path_to_data>
    ```
    
    <path_to_data> should contain the following files:
        - object.json: the json file that contains the part hierarchy and the paths to the textured objs
        - objs: the directory that contains the textured objs
    
    The rendered images will be saved under <path_to_data>/imgs.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='path to the data directory')
    parser.add_argument('--n_imgs', type=int, default=20, help='number of images to render for each model')
    parser.add_argument('--incremental', action='store_true', help='whether to render images incrementally')

    args = parser.parse_args()
     
    try:
        render_imgs(args.data, args.n_imgs, args.save_seg_img, args.incremental)
    except Exception as e:
        with open('render_err.log', 'a') as f:
            f.write(f'{args.data}\n')
            f.write(f'{e}\n')