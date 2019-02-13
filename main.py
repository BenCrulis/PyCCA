import random
from collections import deque
import sys
import os
import pygame
import multiprocessing
from multiprocessing.sharedctypes import RawArray
from multiprocessing import Process
import threading
import time


class Grid:
    def __init__(self,sizeX,sizeY,grid):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.grid = grid
    def __repr__(self):
        return "({},{})\n{}\n".format(self.sizeX, self.sizeY, self.grid)

def newGrid(sizeX, sizeY, variation=0.0):
    ret = RawArray('d', sizeX*sizeY*[0.0])
    for i in range(len(ret)):
        ret[i] = 0.5 + (-random.random() if random.randint(0,1) == 1 else random.random())*variation
    return Grid(sizeX,sizeY,ret)

def wrap_pos(sizeX, sizeY):
    def internal(x,y):
        return (x % sizeX, y % sizeY)
    return internal

def get_cell(g):
    wpos = wrap_pos(g.sizeX,g.sizeY)
    def intern(x,y):
        ax, ay = wpos(x,y)
        return g.grid[ax+ay*g.sizeY]
    return intern

def set_cell(g):
    wpos = wrap_pos(g.sizeX,g.sizeY)
    def intern(x,y, state):
        ax, ay = wpos(x,y)
        g.grid[ax+ay*g.sizeY] = state
    return intern

def parent_cells(x,y):
    return [(x-1,y-1),(x,y-1),(x+1,y-1)]

def get_surrounding(dist):
    def internal(x,y):
        return [(i,j) for i in range(x-dist,x+dist+1) for j in range(y-dist,y+dist+1)]
    return internal

def cell_with_parents(g):
    cell = get_cell(g)
    def int_cp(x,y):
        return (cell(x,y), list(map(lambda t: cell(*t), parent_cells(x,y))))
    return int_cp

def get_surrounding_cells(dist):
    surr = get_surrounding(dist)
    def int1(g):
        cell = get_cell(g)
        def int2(x,y):
            return list(map(lambda t: cell(*t), surr(x,y)))
        return int2
    return int1

def prob_cell_0(g,p,dist,x,y):
    cell = get_cell(g)

    surr = deque(maxlen=3)

    posi = 0
    c = 0

    #dy = y-2
    for dy in range(y-dist-1, y+dist):
        surr.clear()
        for dx in range(x-dist-1, x+dist+2):
            surr.append(cell(dx,dy))

            if len(surr) == 3:
                if surr[0] == p[0] and surr[1] == p[1] and surr[2] == p[2]:
                    if cell(dx-1,dy+1) == 0:
                        posi += 1
                    c += 1
    if c == 0:
        return random.random()
    return float(posi)/c

def interpolate(rate, actual, target):
    if actual > target:
        return actual - (actual - target)*rate
    elif actual < target:
        return actual + (target-actual)*rate
    else:
        return actual

def cell_update_from_surrounding(dist_x, dist_y,rate):
    def intern(g,p,val,x,y,rd):
        cell = get_cell(g)

        surr = deque(maxlen=3)

        posi = 0.0
        c = 0.0

        multiplier = 0

        best_pattern_match = 0
        best_prob = 0
        best_surr = None

        i = 0
        #dy = y-1
        for dy in range(y-dist_y, y+dist_y+1):
            surr.clear()
            for dx in range(x-dist_x-1, x+dist_x+2):
                surr.append(cell(dx,dy))
                if len(surr) == 3:
                    sq_sum_diff = (1-abs(surr[0]-p[0]))*(1-abs(surr[1]-p[1]))*(1-abs(surr[2]-p[2]))
                    son_value = cell(dx-1,dy+1)
                    posi += (sq_sum_diff)*son_value
                    c += (sq_sum_diff)
                    #multiplier += (abs(surr[0]-surr[1])+abs(surr[0]-surr[2])+abs(surr[1]-surr[2]))/3
                    #if sq_sum_diff > best_pattern_match:
                    #    best_pattern_match = sq_sum_diff
                    #    best_prob = son_value
                    #    best_surr = (surr[0],surr[1],surr[2])
                    #elif sq_sum_diff == best_pattern_match:
                    #    if rd.random() < 0.5:
                    #        best_pattern_match = sq_sum_diff
                    #        best_prob = son_value
                    #        best_surr = (surr[0],surr[1],surr[2])
                    i += 1
        if c <= 0.0:
            return 0.0
        prob = float(posi)/c
        certainty = c / i
        setter = set_cell(g)

        #setter(x-1,y-1,interpolate(rate, p[0], best_surr[0]))
        #setter(x,y-1,interpolate(rate, p[1], best_surr[1]))
        #setter(x+1,y-1,interpolate(rate, p[2], best_surr[2]))

        #prob = best_prob
        diff = (1-prob)*rate if rd.random() < prob else -prob*rate
        newval = min(1.0, max(0.1, val + diff))
        #return 1 if prob > 0.5 else 0
        return newval
    return intern

def pick_random(dist_x, dist_y,rate):
    def intern(g,p,val,x,y,rd):
        cell = get_cell(g)
        setter = set_cell(g)

        rx, ry = (rd.randint(x-dist_x, x+dist_x), rd.randint(y-dist_y-1, y+dist_y-1))
        rp = (cell(rx-1,ry-1),cell(rx,ry-1),cell(rx+1,ry-1))

        rpv = [1 if rd.random() < rp[i] else 0 for i in range(3)]
        pv = [1 if rd.random() < p[i] else 0 for i in range(3)]

        if rpv == pv:
            rv = 1 if rd.random() < cell(rx,ry) else 0
            trv = 1 if rd.random() < val else 0

            strength = (abs(0.5-rp[0])+abs(0.5-rp[1])+abs(0.5-rp[2]))*2/3
            strength2 = (abs(0.5-p[0])+abs(0.5-p[1])+abs(0.5-p[2]))*2/3

            newval = val
            if trv > rv:
                newval -= rate*strength*strength2
            elif trv < rv:
                newval += rate*strength*strength2
            newval = min(1.0,max(0.0, newval))
            return newval
        return val
    return intern

#thread_num = multiprocessing.cpu_count()
thread_num = 5

pygame.init()

size = width, height = 600, 600

screen = pygame.display.set_mode(size)
black = 0,0,0
white = 255,255,255

def draw_grid(surface, grid, sizeX, sizeY):
    gsizeX = float(sizeX)/grid.sizeX
    gsizeY = float(sizeY)/grid.sizeY

    cellgetter = get_cell(g)

    for x in range(sizeX):
        for y in range(sizeY):
            cell = cellgetter(x,y)
            gpos = (x*gsizeX, y*gsizeY)
            color = (int(cell*255),int(cell*255),int(cell*255))

            pygame.draw.rect(surface, color, pygame.Rect(gpos[0],gpos[1], gsizeX, gsizeY))

sx, sy = 100,100

g = newGrid(sx,sy, 0.01)

setter = set_cell(g)


pattern = [
    [1,0,1,0,1,0,1,0,1],
    [0,1,0,1,0,1,0,1,0],
    [1,0,1,0,1,0,1,0,1],
    [0,1,0,1,0,1,0,1,0],
    [1,0,1,0,1,0,1,0,1],
    [0,1,0,1,0,1,0,1,0],
    [1,0,1,0,1,0,1,0,1],
    [0,1,0,1,0,1,0,1,0]
]

for y,l in enumerate(pattern):
    for x,v in enumerate(l):
        setter(x+40,y+40,v)


byProb = True
write_images = False

def probabilityRule(dist,byProb):
    def intern(gr, parents, cell,x,y, rd):
        prob = prob_cell_0(gr,parents,dist,x,y)

        if byProb:
            roll = rd.random()
            new = 0 if roll < prob else 1
        else:
            new = 0 if prob > 0.5 else 1

        if cell != new:
            return new
        return None
    return intern

def probabilityRule2(dist,byProb):
    def intern(gr, parents, cell,x,y, rd):
        prob = cell_update_from_surrounding(gr,parents,dist,x,y)

        if byProb:
            roll = rd.random()
            new = 0 if roll < prob else 1
        else:
            new = 0 if prob > 0.5 else 1

        if cell != new:
            return new
        return None
    return intern

rule111 = {(1,1,0): 1,
    (1,0,1): 1,
    (0,1,1): 1,
    (0,1,0): 1,
    (0,0,1): 1}

def wolframRule(wolframRuleNumber):
    def intern(gr, parents, cell,x,y, rd):
        return rule111.get(parents, 0)
    return intern

class CellUpdater(Process):
    def __init__(self,threadCont, updateRule, grid, name="no name"):
        Process.__init__(self, name=name)
        self.grid = grid
        self.name = name
        self.cont = threadCont
        self.it = multiprocessing.Value('I', False)
        self.it.value = 0
        self.start_time = None
        self.stop_time = None
        self.rd = random.Random()
        self.updateRule = updateRule

        self.start_time = time.time()

    def run(self):
        pid = self.pid

        print("starting thread '{}' ({})".format(self.name, pid))

        gr = self.grid
        cell = get_cell(gr)
        cellsetter = set_cell(gr)

        while self.cont.value:
            x,y = (self.rd.randint(0,gr.sizeX-1),self.rd.randint(0,gr.sizeY-1))

            old = cell(x,y)

            new = self.updateRule(gr, (cell(x-1,y-1),cell(x,y-1),cell(x+1,y-1)), old,x,y, self.rd)

            if new is not None:
                cellsetter(x,y,new)

            self.it.value += 1


    def avg_it_per_second(self):
        return float(self.it.value)/(time.time()-self.start_time)



prefix = "/home/ben/Images/automatas/automata_run_"
img_folder = ""
if write_images:
    suffix = 1
    while os.path.exists(prefix+str(suffix)):
        suffix += 1
    img_folder = prefix+str(suffix)
    os.makedirs(img_folder)


"""
threads = [CellUpdater(g, "thread_{}".format(i)) for i in range(1,thread_num+1)]
for thread in threads:
    thread.start()
"""

threadCont = multiprocessing.Value('I',False)
threadCont.value = 1
#threadCont = True

threads = []
for i in range(thread_num):
    threads.append(CellUpdater(threadCont,
                                    pick_random(1,3,0.1),
                                    g,
                                    name="Thread_{}".format(i)))

for thread in threads:
    thread.start()

print("all threads started")
before = time.time()

framerate = 10
clock = pygame.time.Clock()

c = 1
cont = True
i = 0
n = 0
while cont:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cont = False

    clock.tick(framerate)
    #pygame.time.wait(100)

    i += 1

    screen.fill(black)
    draw_grid(screen, g, width, height)
    if write_images:
        img_name = img_folder+"/img_{}.tga".format(i)
        print(img_name)
        pygame.image.save(screen, img_name)
    pygame.display.flip()

threadCont.value = 0

for thread in threads:
    print("stoping thread '{}'...".format(thread.name))
    thread.join()

total_it = 0
for thread in threads:
    thread.join()
    total_it += thread.it.value
    print("Average iterations per second in thread '{}': {}"
        .format(thread.name, int(thread.avg_it_per_second())))

after = time.time()

print("Total average iteration per second: {}".format(int(total_it/(after-before))))

sys.exit()
