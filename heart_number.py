from scene import Scene
import taichi as ti
from taichi.math import *

night_mode = True
exposure = 1.0 + night_mode * 9.

scene = Scene(exposure=10)
scene.set_floor(-1, (0.0, 0.3, 0.5))
scene.set_directional_light((1, 1, 0), 0.2, vec3(1.0, 1.0, 1.0) / exposure)
scene.set_background_color(vec3(0.6, 0.8, 1.0) / exposure)

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

@ti.func
def create_heart(pos, sz, color):
    '''
    3-3*sin(@theta)+sin(@theta)*sqrt(abs(cos(@theta)))/(sin(@theta)+1.6)
    @theta=[0, 2 @ pi, 33], color = red, shade = middle)
    '''
    for I in ti.grouped(ti.ndrange((-sz, sz), (-sz, sz))):
        tmp = vec2([I[0]+0.5,I[1]+0.5])
        tmp[0] = tmp[0] / sz * 4
        tmp[1] = tmp[1] / sz * 4.5 - 1.5
        pol_r = tmp.norm()
        pol_theta = ti.asin(tmp[1] / pol_r)
        sdf = pol_r-(3-3*ti.sin(pol_theta)+(ti.sin(pol_theta)*ti.sqrt(ti.abs(ti.cos(pol_theta)))/(ti.sin(pol_theta)+1.6)))
        if sdf < 0:
            z_axis = sz * ti.exp(-(ti.pow(I[0]/sz,2)/0.25+ti.pow(I[1]/sz,2)/0.25)/2) / (0.5 * ti.math.pi)
            for i in range(-z_axis, z_axis):
                scene.set_voxel(pos + ivec3(I, i), 2, color/exposure)


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
    if night_mode:
        # create_heart(ivec3(40, 40, -40), 20, vec3(0.8, 0.1, 0.2))
        # create_heart(ivec3(40, 0, -40), 20, vec3(0.8, 0.1, 0.2)) # move ahead
        # create_heart(ivec3(-40, 40, -40), 20, vec3(0.8, 0.1, 0.2)) # move left
        create_heart(ivec3(-35, 30, -30), 25, vec3(0.8, 0.1, 0.2)) # move left, ahead

    # number_pos = -35
    # number_pos = 0
    number_pos = 20 # move back
    x_pos = -10
    y_pos = -40

    draw_number([1, 0, 1, 1, 0, 1, 1], ivec3(x_pos, number_pos, y_pos), ORANGE, vec3(0.01)) # 2
    draw_number([1, 1, 1, 1, 1, 1, 0], ivec3(x_pos+9, number_pos, y_pos), BLACK, vec3(0.01)) # 0
    draw_number([1, 0, 1, 1, 0, 1, 1], ivec3(x_pos+18, number_pos, y_pos), ORANGE, vec3(0.01)) # 2
    draw_number([1, 0, 1, 1, 0, 1, 1], ivec3(x_pos+27, number_pos, y_pos), ORANGE, vec3(0.01)) # 2
    draw_number([1, 1, 1, 1, 1, 1, 0], ivec3(x_pos+36, number_pos, y_pos), BLACK, vec3(0.01)) # 0
    draw_number([1, 1, 0, 1, 1, 0, 1], ivec3(x_pos+45, number_pos, y_pos), BLUE, vec3(0.01))# 5
    draw_number([1, 0, 1, 1, 0, 1, 1], ivec3(x_pos+54, number_pos, y_pos), ORANGE, vec3(0.01)) # 2
    draw_number([1, 1, 1, 1, 1, 1, 0], ivec3(x_pos+63, number_pos, y_pos), BLACK, vec3(0.01)) # 0

initialize_voxels()

scene.finish()
