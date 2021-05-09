import json
import random
from collections import OrderedDict
"""
바닥 구멍 -> Object : Sky, [x,13] and [x,14] 두개로 조정. 최대 연타로 4개까지만 뚫어야 함. 그이상되면 못넘음
Object : Pipe (x,height,1) height는 최대 9부터 최소 13까지만 조정 가능, 8부터 마리오가 못넘어감

"""

class MakeRandomMap:
    def __init__(self):
        self.not_exist = 6
        self.maximum_size = 160
        self.data = OrderedDict()
        self.max_jump_height = 9
        self.min_pipe_height = 13
        self.max_jump_length = 4
        self.id = 3
        self.data["id"]=self.id
        self.data["length"]=self.maximum_size
        #Obeject Data.

    def make_ObjectJson(self):
        pipe = self.make_Pipes()  # Pipe = [x,height,1], MaxHeight = 9.
        bush = []  # Bush = [x,y(=12)], y=12가 땅바닥에 붙어 있으므로 12로 고정.
        sky = []  # Sky = [x,13] and [x,14] 두 개를 한방에 뚫어야 됨.
        cloud = []  # cloud = [x,y] => 최소 9까지 y가 존재 가능. 그 밑으로 내려가면 땅바닥에 붙음
        ground = []  # ground = [x,y] => y가 9 넘어가면 마찬가지로 못넘음.
        object = {"pipe":pipe}
        return object

    def make_Pipes(self):
        temp = []
        maximum_pipes = 5
        xSample = range(self.not_exist,self.maximum_size)
        x = random.sample(xSample, maximum_pipes)
        for Repeats in range(maximum_pipes):
            height = random.randrange(self.max_jump_height, self.min_pipe_height)
            temp.append([x[Repeats],height,2])# 이 맨 뒤에꺼 대체 뭔지 모르겠음..
        return temp

    def make_LayerJson(self):
        x = [0,self.maximum_size]
        sky_y = [0,13]
        ground_y = [14,16]
        sky = {"x" : x, "y": sky_y}
        ground = {"x": x, "y": ground_y}
        LayerJson = {"sky":sky, "ground":ground}
        return LayerJson

    def do_test(self):
        self.data["level"]={"objects":self.make_ObjectJson(),"layers":self.make_LayerJson()}
        print(json.dumps(self.data,ensure_ascii=False,indent="\t"))

    def write_Json(self):
        self.data["level"]={"objects":self.make_ObjectJson(),"layers":self.make_LayerJson()}
        with open('./levels/Level-'+str(self.id)+'.json','w',encoding="utf-8") as make_file:
            json.dump(self.data,make_file,ensure_ascii=False,indent="\t")