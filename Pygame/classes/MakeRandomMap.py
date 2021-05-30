import json
import random
from collections import OrderedDict
"""
바닥 구멍 -> Object : Sky, [x,13] and [x,14] 두개로 조정. 최대 연타로 4개까지만 뚫어야 함. 그이상되면 못넘음
Object : Pipe (x,height,1) height는 최대 9부터 최소 13까지만 조정 가능, 8부터 마리오가 못넘어감

"""

class MakeRandomMap:
    def __init__(self):
        self.not_exist = 10 # Start ~ not_exist, not_exist ~ End 까지는 아무것도 안 생김.( 엔티티 프리 )
        self.maximum_size = 320 # 맵 크기.
        self.data = OrderedDict()
        self.max_jump_height = 9 # 점프로 뛰어넘을 수 있는 블럭 높이.
        self.min_pipe_height = 13 # 파이프의 최소 높이.
        self.max_jump_length = 4 #최대, 점프로 뛰어넘을 수 있는 블럭 개수.
        self.minimum_heights = 10 #공중에 떠 있는 블럭의 최소 높이.
        self.id = 3 # 맵 번호
        self.data["id"]=self.id
        self.data["length"]=self.maximum_size
        self.forbidden_X = []
        self.external_walls = []
        self.havehieghts =[]
        #Obeject Data.

    def make_ObjectJson(self):
        pipe = self.make_Pipes()  # Pipe = [x,height,1], MaxHeight = 9.
        bush = []  # Bush = [x,y(=12)], y=12가 땅바닥에 붙어 있으므로 12로 고정.
        sky = self.make_Sky()  # Sky = [x,13] and [x,14] 두 개를 한방에 뚫어야 됨.
        cloud = []  # cloud = [x,y] => 최소 9까지 y가 존재 가능. 그 밑으로 내려가면 땅바닥에 붙음
        ground = self.make_Ground()  # ground = [x,y] => y가 9 넘어가면 마찬가지로 못넘음.
        object = {"pipe":pipe,"sky":sky,"ground":ground}
        return object

    def get_Random_X_Position(self,range1,range2,counts,delta):
        temp = []
        xSample = range(range1,range2)
        #forbidden X와 중복을 찾아내서 제거한다.
        while True:
            if temp.__len__() == counts: break
            x = random.sample(xSample,counts-temp.__len__())
            t = list(set(x).intersection(self.forbidden_X))
            t = list(set(x)-set(t))
            #Sort,
            t.extend(temp)
            t.extend(self.forbidden_X)
            t.sort()
            res = []#Copied Array.
            res.extend(t)
            res2 = []#역순 비교용
            res2.extend(t)
            for index in range(len(t)-1):
                #if Bigger then delta,
                if(t[index]+delta >= t[(index+1)]):
                    res.remove(t[index+1])
                    res2.remove(t[index])
            check = list(set(res).intersection(temp))
            check2 = list(set(res).intersection(self.forbidden_X))
            check3 = list(set(res2).intersection(temp))
            check4 = list(set(res2).intersection(self.forbidden_X))
            ans1 = list(set(res)-set(check)-set(check2))
            ans2 = list(set(res2)-set(check3)-set(check4))
            temp.extend(list(set(ans1).intersection(ans2)))
        self.forbidden_X.extend(temp)
        return temp

    def make_Ground(self):
        temp=[]
        maximum_walls = 20# 몇개의 벽을 만들것인지?
        min_continuous_block = 3#최소 연속된 벽 개수.
        max_continuous_block = 5#최대 연속된 벽 개수.
        x = self.get_Random_X_Position(self.not_exist,self.maximum_size-self.not_exist,maximum_walls,delta=max_continuous_block)
        #일단 External walls부터 만들어 줌. ( 바닥에 딱붙은 가드레일 )
        for item in self.external_walls:
            temp.append([item,12])

        #그 다음 공중에 띄워진 블럭을 생성.
        for item in x:
            Count = random.randint(min_continuous_block, max_continuous_block)
            height = random.randint(self.max_jump_height,self.minimum_heights)
            for index in range (Count):
                temp.append([item+index,height])
                self.forbidden_X.append(item+index)
                self.havehieghts.append((item+index,height))

        # print("---Make Ground---")
        # print(x)
        # print("Forbidden_X")
        # print(self.forbidden_X)
        # print("haveHeight")
        # print(self.havehieghts)
        # print("-----------------")
        return temp

    def make_Sky(self):
        temp=[]
        maximum_sky = 8#몇개의 구멍을 뚫을 것인지?
        min_continuous_sky = 2# 최소 연속된 구멍 개수는 2개이고,
        max_continuous_sky = 4# 최대 연속된 구멍 개수는 4칸까지.
        x = self.get_Random_X_Position(self.not_exist,self.maximum_size-self.not_exist,maximum_sky,delta=max_continuous_sky+3)
        #3를 더해주는 이유는 그 위치에 벽을 만들것이라서 그럼.
        #ex, 4칸짜리 drop
        #벽-낭-떠-러-지-벽
        #Make_Skys.
        for Repeats in range(maximum_sky):
            Count = random.randint(min_continuous_sky, max_continuous_sky)
            for index in range(Count+1):
                if index != 0:#0은 이미 forbidden_X에 추가되어 있으므로.
                    temp.append([x[Repeats] + index, 13])
                    temp.append([x[Repeats] + index, 14])
                    self.forbidden_X.append(x[Repeats]+index)
                else:
                    self.external_walls.append(x[Repeats])
            self.forbidden_X.append(x[Repeats]+index+1)
            self.external_walls.append(x[Repeats]+index+1)

        # print("---Make Sky---")
        # print(x)
        # print("Forbidden_X")
        # print(self.forbidden_X)
        # print("External Walls")
        # print(self.external_walls)
        # print("haveHeight")
        # print(self.havehieghts)
        # print("--------------")
        #self.forbidden_X.extend(temp)
        return temp

    def make_Pipes(self):
        temp = []
        maximum_pipes = 5# 생성할 파이프 개수.
        pipe_size = 2 #파이프의 가로 길이. 즉 x하나당 2개씩 잡아먹음.
        #Make X.
        x = self.get_Random_X_Position(self.not_exist, self.maximum_size - self.not_exist, maximum_pipes,delta=pipe_size)
        for Repeats in range(maximum_pipes):
            height = random.randrange(self.max_jump_height, self.min_pipe_height)
            temp.append([x[Repeats],height,2])# 이 맨 뒤에꺼 대체 뭔지 모르겠음..
            self.forbidden_X.append(x[Repeats]+1)
            #높이 정보를 가지고 있는 애들 리스트.
            self.havehieghts.append((x[Repeats],height))

        # print("---Make Pipes---")
        # print(x)
        # print("Forbidden_X")
        # print(self.forbidden_X)
        # print("haveHeight")
        # print(self.havehieghts)
        # print("----------------")

        return temp

    def make_LayerJson(self):
        x = [0,self.maximum_size]
        sky_y = [0,13]
        ground_y = [14,16]
        sky = {"x" : x, "y": sky_y}
        ground = {"x": x, "y": ground_y}
        LayerJson = {"sky":sky, "ground":ground}
        return LayerJson

    def make_EntitiesJson(self):
        #하나라도 비었으면 Exception.
        coin = [[1,13]]
        CoinBox = [[3,5]]
        coinBrick = [[37,9]]
        Goomba = self.make_Goomba()#[y,x] 로 생성. ground = 12.
        Koopa = self.make_Koopa()#Gommba와 동일.
        RandomBox = [[4, 9, "RedMushroom"]]

        EntitiesJson = {"Goomba":Goomba,"Koopa":Koopa,"coin":coin,"CoinBox":CoinBox,"coinBrick":coinBrick,"RandomBox":RandomBox}
        return EntitiesJson

    def make_RandomBox(self):
        temp = []
        how_many_RandomBox = 5

    def make_Goomba(self):
        temp = []
        #빈 땅에 굼바 만들기.
        how_many_Goomba = 20
        x = self.get_Random_X_Position(self.not_exist,self.maximum_size-self.not_exist,how_many_Goomba,delta=0)
        for item in x:
            temp.append([12,item])
        return temp

    def make_Koopa(self):
        temp=[]
        how_many_Koopa = 20
        x = self.get_Random_X_Position(self.not_exist, self.maximum_size - self.not_exist, how_many_Koopa, delta=0)
        for item in x:
            temp.append([12, item])
        return temp

    def retMapSize(self):
        return self.maximum_size

    def do_test(self):
        self.data["level"]={"objects":self.make_ObjectJson(),"layers":self.make_LayerJson()}
        print(json.dumps(self.data,ensure_ascii=False,indent="\t"))

    def write_Json(self):
        self.data["level"]={"objects":self.make_ObjectJson(),"layers":self.make_LayerJson(),"entities":self.make_EntitiesJson()}
        with open('./Pygame/levels/Level1-'+str(self.id)+'.json','w',encoding="utf-8") as make_file:
            json.dump(self.data,make_file,ensure_ascii=False,indent="\t")