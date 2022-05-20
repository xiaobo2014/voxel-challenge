from scene import Scene
import taichi as ti
from taichi.math import *


scene = Scene(exposure=1)


@ti.func
def create_olympic(pos_2d, radius, color):
    for i, j in ti.ndrange((-radius, radius), (-radius, radius)):
        point_2d = vec2(i, j)
        if i*i + j*j < radius*radius and i*i + j*j > radius*radius*0.75: 
            scene.set_voxel(vec3(pos_2d[0]+point_2d[0], pos_2d[1]+point_2d[1], 0), 2, color)

@ti.kernel
def initialize_voxels():
    create_olympic(ivec2(10, 30), 10, vec3(0, 107.0/255.0, 176.0/255.0))
    create_olympic(ivec2(32, 30), 10, vec3(29.0/255.0, 24.0/255.0, 21.0/255.0))
    create_olympic(ivec2(54, 30), 10, vec3(220.0/255.0, 47.0/255.0, 31.0/255.0))
    create_olympic(ivec2(21, 20), 10, vec3(239.0/255.0, 169.0/255.0, 13.0/255.0))
    create_olympic(ivec2(43, 20), 10, vec3(5.0/255.0, 147.0/255.0, 65.0/255.0))
initialize_voxels()

scene.finish()
