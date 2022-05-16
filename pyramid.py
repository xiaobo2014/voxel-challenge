from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=1)
scene.set_floor(-0.05, (1.0, 1.0, 1.0))


@ti.kernel
def initialize_voxels():
    n = 50
    for k in range(1, n/2):
        for i, j in ti.ndrange(n, n):
            if min(i, j) <= k or max(i, j) >= n - 1 - k:
                continue
            else:
                scene.set_voxel(vec3(i, k, j), 2, vec3(0.6, 0.6, 0))


initialize_voxels()

scene.finish()
