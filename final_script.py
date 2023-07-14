
import bpy
import bpy_extras
from mathutils import Euler
import numpy as np
import os
import re
import uuid
from bpy_extras.object_utils import world_to_camera_view
from datetime import datetime
from math import pi
from mathutils import Vector

class BlenderObjectManipulator:
    OUTPUT_DIR = 'C:/chatgptv2/synthdata/Tools/dataset/v4'
    LABELS_DIR = 'labels/'
    IMAGES_DIR = 'images/'
    BACKGROUND_IMAGES_PATH = 'C:/chatgptv2/synthdata/Tools/assets/hdri'

    def __init__(self, mesh_names, mesh2material, mesh2class, spiral_radius_range, spiral_height_range, spiral_n_points, background_only, sample):
        self.mesh_names = mesh_names
        self.mesh2material = mesh2material
        self.mesh2class = mesh2class
        self.spiral_radius_range = spiral_radius_range
        self.spiral_height_range = spiral_height_range
        self.spiral_n_points = spiral_n_points
        self.background_only = background_only
        self.sample = sample

        self.ensure_directory_exists(os.path.join(self.OUTPUT_DIR, self.IMAGES_DIR))
        self.ensure_directory_exists(os.path.join(self.OUTPUT_DIR, self.LABELS_DIR))
        
        self.bg_images = [f for f in os.listdir(self.BACKGROUND_IMAGES_PATH) 
                          if os.path.splitext(f)[-1].lower() in ['.hdr']]

    @staticmethod
    def ensure_directory_exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def hide_objects(mesh_names, hide):
        for mesh_name in mesh_names:
            obj = bpy.data.objects.get(mesh_name)
            if obj is not None:
                obj.hide_viewport = hide
                obj.hide_render = hide

    @staticmethod
    def render_still(filepath: str, file_format='PNG'):
        scene = bpy.context.scene
        resolutions = {
            "4032x3024":0.1,
            "1920 x 1080": 0.1,
            "1280x720": 0.2,  # 16:9 aspect ratio
            "1024x768": 0.1,  # 4:3 aspect ratio
            "800x600": 0.1,
            "720x480": 0.1,  # 3:2 aspect ratio
            "640x640":0.1,
            "640x480": 0.1,  # 4:3 aspect ratio
            "280x280": 0.1 
        }


        # Normalize the probabilities so they sum to 1
        probabilities = np.array(list(resolutions.values()))
        probabilities /= probabilities.sum()

        # Choose a resolution randomly, according to the provided probabilities
        chosen_resolution = np.random.choice(list(resolutions.keys()), p=probabilities)

        # Split the resolution string into separate width and height values
        resolution_x, resolution_y = map(int, chosen_resolution.split('x'))

        scene.render.resolution_x = resolution_x
        scene.render.resolution_y = resolution_y
        scene.render.resolution_percentage = 100
        scene.render.filepath = filepath
        scene.render.image_settings.file_format = file_format
        bpy.ops.render.render(write_still=True)

    def set_world_image(self, image_path, bg_strength_range=[0.2, 0.4]):
        world = bpy.data.worlds["World"].node_tree

        if 'random_texture_node' in world.nodes:
            tex_node = world.nodes['random_texture_node']
            world.nodes.remove(tex_node)

        selected_image_path = os.path.join(self.BACKGROUND_IMAGES_PATH, image_path)

        tex_image_node = world.nodes.new('ShaderNodeTexEnvironment')
        tex_image_node.name = 'random_texture_node'
        tex_image_node.image = bpy.data.images.load(selected_image_path)

        world.links.new(tex_image_node.outputs[0], world.nodes["Background"].inputs[0])
        rand_bg = np.random.uniform(low=bg_strength_range[0], high=bg_strength_range[1])
        world.nodes["Background"].inputs[1].default_value = rand_bg

    def orient_camera(self, target_mesh_name, camera_name):
        target_object = bpy.data.objects[target_mesh_name]
        camera_object = bpy.data.objects[camera_name]

        direction = target_object.location - camera_object.location
        # point the camera to the target
        rot_quat = direction.to_track_quat('-Z', 'Y')

        camera_object.rotation_euler = rot_quat.to_euler()



    def get_all_coordinates(self, mesh_name):
        b_box = self.find_bounding_box(bpy.data.objects[mesh_name])

        if b_box:
            return self.format_coordinates(b_box, self.mesh2class[mesh_name])

        return ''

    def format_coordinates(self, coordinates, _class):
        if coordinates: 
            ## Change coordinates reference frame
            x1 = (coordinates[0][0])
            x2 = (coordinates[1][0])
            y1 = (1 - coordinates[1][1])
            y2 = (1 - coordinates[0][1])

            ## Get final bounding box information
            width = (x2-x1)  # Calculate the absolute width of the bounding box
            height = (y2-y1) # Calculate the absolute height of the bounding box
            # Calculate the absolute center of the bounding box
            cx = x1 + (width/2) 
            cy = y1 + (height/2)

            ## Formulate line corresponding to the bounding box of one class
            txt_coordinates = str(_class) + ' ' + str(cx) + ' ' + str(cy) + ' ' + str(width) + ' ' + str(height) + '\n'

            return txt_coordinates
        # If the current class isn't in view of the camera, then pass
        else:
            pass

    def find_bounding_box(self, obj):
        camera_object = bpy.data.objects['Camera']
        matrix = camera_object.matrix_world.normalized().inverted()
        """ Create a new mesh data block, using the inverse transform matrix to undo any transformations. """
        mesh = obj.to_mesh(preserve_all_data_layers=True)
        mesh.transform(obj.matrix_world)
        mesh.transform(matrix)
        """ Get the world coordinates for the camera frame bounding box, before any transformations. """
        frame = [-v for v in camera_object.data.view_frame(scene=bpy.context.scene)[:3]]

        lx = []
        ly = []

        for v in mesh.vertices:
            co_local = v.co
            z = -co_local.z

            if z <= 0.0:
                """ Vertex is behind the camera; ignore it. """
                continue
            else:
                """ Perspective division """
                frame = [(v / (v.z / z)) for v in frame]

            min_x, max_x = frame[1].x, frame[2].x
            min_y, max_y = frame[0].y, frame[1].y

            x = (co_local.x - min_x) / (max_x - min_x)
            y = (co_local.y - min_y) / (max_y - min_y)

            lx.append(x)
            ly.append(y)

        """ Image is not in view if all the mesh verts were ignored """
        if not lx or not ly:
            return None

        min_x = np.clip(min(lx), 0.0, 1.0)
        min_y = np.clip(min(ly), 0.0, 1.0)
        max_x = np.clip(max(lx), 0.0, 1.0)
        max_y = np.clip(max(ly), 0.0, 1.0)

        """ Image is not in view if both bounding points exist on the same side """
        if min_x == max_x or min_y == max_y:
            return None

        """ Figure out the rendered image size """
        render = bpy.context.scene.render
        fac = render.resolution_percentage * 0.01
        dim_x = render.resolution_x * fac
        dim_y = render.resolution_y * fac
        
        ## Verify there's no coordinates equal to zero
        coord_list = [min_x, min_y, max_x, max_y]
        if min(coord_list) == 0.0:
            indexmin = coord_list.index(min(coord_list))
            coord_list[indexmin] = coord_list[indexmin] + 0.0000001

        return (min_x, min_y), (max_x, max_y)
    
    def write_annotations(self, mesh_name, filename_img):
        annotations = self.get_all_coordinates(mesh_name)

        filename_no_ext = os.path.splitext(filename_img)[0]
        label_filename = f"{filename_no_ext}.txt"
        label_filepath = os.path.join(self.OUTPUT_DIR, self.LABELS_DIR, label_filename)

        with open(label_filepath, 'w') as file:
            if annotations and not self.background_only:
                file.write(annotations)


    # Remaining methods for bounding box calculation, formatting and annotations writing...
    def create_3d_spiral(self, center):
        radius_range = self.spiral_radius_range
        height_range = self.spiral_height_range
        num_points = self.spiral_n_points
        radius = np.random.uniform(*radius_range)
        start_height, end_height = [height_range[0], height_range[1]]
        start_angle = np.random.uniform(0, 2.*np.pi)  # random starting angle

        theta = np.linspace(start_angle, start_angle + 2.*np.pi, num_points)  # angle array
        z = np.linspace(start_height, end_height, num_points)  # height array
        x = center[0] + radius * np.sin(theta)  # x coordinates
        y = center[1] + radius * np.cos(theta)  # y coordinates
        return list(zip(x, y, z))

    @staticmethod
    def randomize_materials_rgb(materials = ['plain', 'transparent']):
        for _material in materials:
            material = bpy.data.materials[_material]
            rand_rgb = np.random.uniform(low=0., high=1., size=3)
            rand_rgb = np.append(rand_rgb, 1.)
            material.node_tree.nodes["RGB"].outputs[0].default_value = tuple(rand_rgb)

    def process(self):
        camera_object = bpy.data.objects['Camera']  # Get the camera object once
        render_counter = 0

        if self.background_only:
            for bg_image in self.bg_images:
                self.set_world_image(bg_image)
                for _ in range(self.spiral_n_points):
                    self.hide_objects(self.mesh_names, True)  # Hide all meshes
                    target_object = bpy.data.objects[self.mesh_names[0]]  # Get the target object once
                    camera_positions = np.array(self.create_3d_spiral(target_object.location))  # Convert to numpy array
                    camera_object.location = Vector(camera_positions[np.random.choice(len(camera_positions))])  # Select random row
                    self.orient_camera(self.mesh_names[0], 'Camera')
                    filename_img = f"{render_counter}-{uuid.uuid4()}.png"
                    filepath_img = os.path.join(self.OUTPUT_DIR, self.IMAGES_DIR, filename_img)
                    self.render_still(filepath_img)
                    self.write_annotations(self.mesh_names[0], filename_img)
                    render_counter += 1
                    if sample and render_counter > 10:
                        return

        else:
            
            for bg_image in self.bg_images:
                self.set_world_image(bg_image)
                for mesh_name in self.mesh_names:
                    self.hide_objects(self.mesh_names, True)  # Hide all meshes
                    self.hide_objects([mesh_name], False)  # Unhide the current mesh
                    target_object = bpy.data.objects[mesh_name]  # Get the target object once
                    camera_positions = self.create_3d_spiral(target_object.location)
                    for pos in camera_positions:
                        self.randomize_materials_rgb([self.mesh2material[mesh_name]])
                        camera_object.location = Vector(pos)
                        self.orient_camera(mesh_name, 'Camera')
                        filename_img = f"{render_counter}-{uuid.uuid4()}.png"
                        filepath_img = os.path.join(self.OUTPUT_DIR, self.IMAGES_DIR, filename_img)
                        self.render_still(filepath_img)
                        self.write_annotations(mesh_name, filename_img)
                        render_counter+=1
                        if sample and render_counter > 10:
                            return
        self.hide_objects(self.mesh_names, False)
    

if __name__ == "__main__":
    mesh_names = ['Lighter1', 'Lighter2']
    # mesh_names = ['Lighter1']

    mesh2material = {
        'Lighter1':'transparentv2',
        'Lighter2':'plain'
    }
    mesh2class = {
        'Lighter1':'0',
        'Lighter2':'0'
    }
    spiral_radius_range = [145,160]
    spiral_height_range = [-150,150]
    spiral_n_points = 7
    background_only = True
    sample = False
    manipulator = BlenderObjectManipulator(mesh_names=mesh_names, mesh2material=mesh2material, mesh2class=mesh2class,
                                           spiral_radius_range=spiral_radius_range, spiral_height_range=spiral_height_range, 
                                           spiral_n_points=spiral_n_points, background_only=background_only, sample=sample)
    manipulator.process()
