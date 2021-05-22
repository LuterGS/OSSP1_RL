from classes.Maths import Vec2D


class Camera:
    def __init__(self, pos, entity,length):
        self.pos = Vec2D(pos.x, pos.y)
        self.entity = entity
        self.x = self.pos.x * 32
        self.y = self.pos.y * 32
        self.length=length

    def move(self):
        xPosFloat = self.entity.getPosIndexAsFloat().x
 #       print("xPosFloat: "+str(xPosFloat))
        if 10 < xPosFloat < (self.length-10):
            self.pos.x = -xPosFloat + 10
        self.x = self.pos.x * 32
        self.y = self.pos.y * 32
