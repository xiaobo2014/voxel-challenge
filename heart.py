from scene import Scene
import taichi as ti
from taichi.math import *

night_mode = True
exposure = 1.0 + night_mode * 9.

scene = Scene(exposure=10)
scene.set_floor(-1, (0.0, 0.3, 0.5))
scene.set_directional_light((1, 1, 0), 0.2, vec3(1.0, 1.0, 1.0) / exposure)
scene.set_background_color(vec3(0.6, 0.8, 1.0) / exposure)

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

@ti.kernel
def initialize_voxels():
    if night_mode:
        create_heart(ivec3(40, 40, -40), 20, vec3(0.8, 0.1, 0.2))

initialize_voxels()

scene.finish()
