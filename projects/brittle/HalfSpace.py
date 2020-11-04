import taichi as ti

@ti.data_oriented
class HalfSpace:

    def __init__(self, center, normal, collisionType):
        self.center = center
        self.normal = normal
        self.collisionType = collisionType #0 = STICK, 1 = SLIP

    def collide(self, i, j, v):
        return 0
    