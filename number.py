from scene import Scene
import taichi as ti
from taichi.math import *

BLACK = vec3(0.1, 0.1, 0.1)
RED = vec3(1.0, 0, 0)
ORANGE = vec3(1.0, 165 / 255.0, 0)
YELLOW = vec3(1.0, 1.0, 0)
GREEN = vec3(0, 128 / 255.0, 0)
BLUE = vec3(0, 0, 1.0)
CYAN = vec3(0, 1.0, 1.0)
PURPLE = vec3(128 / 255.0, 0, 128 / 255.0)
GOLD = vec3(1.0 , 215 / 255.0, 0)
WHITE = vec3(1.0, 1.0, 1.0)
KHAKI = vec3(237 / 255.0, 189 / 255.0, 101 / 255.0)

height = 9; rad = 1; hgtHalf = (height - rad) / 2 + rad * 2
width = round((height + 2 * rad)  * 2 / (ti.sqrt(5) + 1))
depth = 4; ori_radius = 8
light = False # 改变光照

scene = Scene(voxel_edges = 0.01, exposure = 2 - light)
scene.set_floor(-0.85, (204 / 255.0, 229 / 225.0, 1.0))
scene.set_background_color((0.9, 0.98, 1) if light else (0.01, 0.01, 0.02))
scene.set_directional_light((0.5, 1, 1), 0.2, (0.9, 0.98, 1) if light else (0.01, 0.01, 0.02)) 

@ti.func
def create_block(pos, size, mat, color):
    for I in ti.grouped(
            ti.ndrange((pos[0], pos[0] + size[0]), (pos[1], pos[1] + size[1]), (pos[2], pos[2] + size[2]))):
        scene.set_voxel(I, mat, color)

@ti.func
def create_random_block(pos, size, mat):
    for I in ti.grouped(
            ti.ndrange((pos[0], pos[0] + size[0]), (pos[1], pos[1] + size[1]), (pos[2], pos[2] + size[2]))):
        scene.set_voxel(I, mat, vec3(ti.random(), ti.random(), ti.random()))
        
@ti.func
def draw_number(list, pos, color, color_noise):
    mat = 1 if light else 2
    if list[0] == 1:
        create_block(pos, ivec3(width, rad, rad), mat, color)
    if list[1] == 1:
        create_block(pos + ivec3(width - rad, 0, 0), ivec3(rad, hgtHalf, rad), mat, color)
    if list[2] == 1:
        create_block(pos + ivec3(width - rad, hgtHalf - rad, 0), ivec3(rad, hgtHalf, rad), mat, color)
    if list[3] == 1:
        create_block(pos + ivec3(0, height + rad, 0), ivec3(width, rad, rad), mat, color)
    if list[4] == 1:
        create_block(pos + ivec3(0, hgtHalf - rad, 0), ivec3(rad, hgtHalf, rad), mat, color)
    if list[5] == 1:
        create_block(pos, ivec3(rad, hgtHalf, rad), mat, color)
    if list[6] == 1:
        create_block(pos + ivec3(0, hgtHalf - rad, 0), ivec3(width, rad, rad), mat, color)

@ti.func
def create_heap(pos, color):
    mat = 1 if light else 2
    for j in range(depth):
        radius = ori_radius - depth + j
        for i, k in ti.ndrange((-radius, radius), (-radius, radius)):
            prob = max((radius - vec2(i, k).norm()) / radius, 0)
            getMat, _ = scene.get_voxel(pos + ivec3(i, -j, k))
            if (ti.random() < prob**(depth - j)) or (j > 0 and (getMat == 1 or getMat == 2)):
                if ti.random() > 0.618:
                    scene.set_voxel(pos + ivec3(i, -(j + 1), k), 1, KHAKI)
                else:
                    scene.set_voxel(pos + ivec3(i, -(j + 1), k), mat, color)
                       
@ti.kernel
def initialize_voxels():
    floor_pos = -40
    number_pos = floor_pos + 5
    '''
    draw_number([1, 1, 1, 1, 1, 1, 0], ivec3(-53, number_pos, 0), BLACK, vec3(0.01)) # 0
    draw_number([0, 1, 1, 0, 0, 0, 0], ivec3(-42, number_pos, 0), RED, vec3(0.01)) # 1
    draw_number([1, 0, 1, 1, 0, 1, 1], ivec3(-31, number_pos, 0), ORANGE, vec3(0.01)) # 2
    draw_number([1, 1, 1, 1, 0, 0, 1], ivec3(-20, number_pos, 0), YELLOW, vec3(0.01)) # 3
    draw_number([0, 1, 1, 0, 1, 0, 1], ivec3(-9, number_pos, 0), GREEN, vec3(0.01)) # 4
    draw_number([1, 1, 0, 1, 1, 0, 1], ivec3(2, number_pos, 0), BLUE, vec3(0.01))# 5
    draw_number([1, 1, 0, 1, 1, 1, 1], ivec3(13, number_pos, 0), CYAN, vec3(0.01))# 6
    draw_number([0, 1, 1, 1, 0, 0, 0], ivec3(24, number_pos, 0), PURPLE, vec3(0.01))# 7
    draw_number([1, 1, 1, 1, 1, 1, 1], ivec3(35, number_pos, 0), GOLD, vec3(0.01))# 8
    draw_number([1, 1, 1, 1, 1, 0, 1], ivec3(46, number_pos, 0), WHITE, vec3(0.01))# 9
    '''
    draw_number([1, 0, 1, 1, 0, 1, 1], ivec3(-53, number_pos, 0), ORANGE, vec3(0.01)) # 2
    draw_number([1, 1, 1, 1, 1, 1, 0], ivec3(-42, number_pos, 0), BLACK, vec3(0.01)) # 0
    draw_number([1, 0, 1, 1, 0, 1, 1], ivec3(-31, number_pos, 0), ORANGE, vec3(0.01)) # 2
    draw_number([1, 0, 1, 1, 0, 1, 1], ivec3(-20, number_pos, 0), ORANGE, vec3(0.01)) # 2
    draw_number([1, 1, 1, 1, 1, 1, 0], ivec3(-9, number_pos, 0), BLACK, vec3(0.01)) # 0
    draw_number([1, 1, 0, 1, 1, 0, 1], ivec3(2, number_pos, 0), BLUE, vec3(0.01))# 5
    draw_number([1, 0, 1, 1, 0, 1, 1], ivec3(13, number_pos, 0), ORANGE, vec3(0.01)) # 2
    draw_number([1, 1, 1, 1, 1, 1, 0], ivec3(24, number_pos, 0), BLACK, vec3(0.01)) # 0
    

initialize_voxels()
scene.finish()