# Blender Synthetic Dataset Generation for Object Detection

This repository contains a Python script that leverages the Blender API to generate a synthetic dataset for object detection tasks. The main object of interest for this project is a regular lighter, but the code can easily be adapted to any 3D object. 

## Overview

The Python script is written in Blender Python (bpy), and should be run inside the Blender environment. It automates a series of 3D rendering tasks, including positioning the camera, rendering objects, and generating annotated images for object detection models.

## Key Features

1. **Camera Positioning:** The camera navigates along a 3D spiral path to introduce variability in the viewpoint for the rendered images.
2. **World Image Setting:** The script sets the background image of the 3D scene, adding more variability to the dataset.
3. **Material Color Randomization:** The color of the materials is randomly changed for each render, making the dataset more diverse and the model robust against changes in color.
4. **Rendering and Annotations:** The script renders the scene, saves the result as a PNG image, and writes the bounding box coordinates of the 3D object in the rendered image into a text file.

## Usage

The main class, `BlenderObjectManipulator`, should be initialized with parameters such as mesh names, mappings for materials and classes, spiral path for the camera parameters, and a flag indicating whether only the background should be rendered.

Please refer to the provided example for more details on how to use the script.

## Requirements

This script requires Blender version 2.93 or later. 

## Contributing

Contributions to this project are welcome. Please open an issue or submit a pull request.

## References

1. [federicoarenasl/Data-Generation-with-Blender](https://github.com/federicoarenasl/Data-Generation-with-Blender)
2. [Synthetic Data Generation for Computer Vision in Blender (Part 1)](https://betterprogramming.pub/synthetic-data-generation-for-computer-vision-in-blender-part-1-6926819b11e6?gi=1f3e5f2f5bca)

---

This README is a template and might need modifications to fit your project needs. Let me know if you need any changes!
