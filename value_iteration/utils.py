import numpy as np
import random

def inBound(pos, im_size):
    (x,y) = pos
    if 0<x<=im_size and 0<y<=im_size:
        return True
    else:
        return False

def check_action(point, obstacles, im_size):
    action = []
    if (not point.left in obstacles) and (inBound(point.left, im_size)):
        action.append('Left')
    if (not point.right in obstacles) and (inBound(point.right, im_size)):
        action.append('Right')
    if (not point.up in obstacles) and (inBound(point.up, im_size)):
        action.append('Up')
    if (not point.down in obstacles) and (inBound(point.down, im_size)):
        action.append('Down')
    if (not point.leftup in obstacles) and (inBound(point.leftup, im_size)):
        action.append('LeftUp')
    if (not point.rightup in obstacles) and (inBound(point.rightup, im_size)):
        action.append('RightUp')
    if (not point.leftdown in obstacles) and (inBound(point.leftdown, im_size)):
        action.append('LeftDown')
    if (not point.rightdown in obstacles) and (inBound(point.rightdown, im_size)):
        action.append('RightDown')

    return tuple(action)

def find_way(point, V, obstacles, im_size):
    point_list = []
    acs = check_action(point, obstacles, im_size)
    dic = {}
    if 'Left' in acs:
        dic[point.left] = V[point.left]
    if 'Right' in acs:
        dic[point.right] = V[point.right]
    if 'Up' in acs:
        dic[point.up] = V[point.up]
    if 'Down' in acs:
        dic[point.down] = V[point.down]
    if 'LeftUp' in acs:
        dic[point.leftup] = V[point.leftup]
    if 'LeftDown' in acs:
        dic[point.leftdown] = V[point.leftdown]
    if 'RightUp' in acs:
        dic[point.rightup] = V[point.rightup]
    if 'RightDown' in acs:
        dic[point.rightdown] = V[point.rightdown]

    max_v = float('-inf')
    for k,v in dic.items():
        if v == 0:
            return k
        if v > max_v:
            max_v = v
    if max_v == float('-inf'):
        return (-1, -1)
    for k in sorted(dic, key=dic.__getitem__, reverse=True):
        if dic[k] == max_v:
            point_list.append(k)

    next_point = random.choice(point_list)
    return next_point



def get_info(Xval_data, Xim_data, map_num, terminal_reward, im_size):
    terminal_state = np.argwhere(Xval_data[map_num] == terminal_reward)
    terminal_state = tuple(terminal_state[0])
    terminal_state = (terminal_state[0] + 1, terminal_state[1] + 1)

    obstacles = []
    for x in range(len(Xim_data[map_num])):
        for y in range(len(Xim_data[map_num][x])):
            if Xim_data[map_num][x][y]:
                obstacles.append((x + 1, y + 1))

    idx = True
    while idx:
        start_x = np.random.randint(1, im_size+1)
        start_y = np.random.randint(1, im_size+1)
        p = position(start_x,start_y, im_size)
        acs = check_action(p,obstacles, im_size)
        if (start_x,start_y) not in obstacles and len(acs)!=0:
            idx = False
    start_state = (start_x, start_y)
    return start_state, terminal_state, obstacles

def print_values(V, g):
  for i in range(1,g.width+1):
    print("---------------------------")
    for j in range(1,g.height+1):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")
def print_policy(P, g):
  for i in range(1,g.width+1):
    print("---------------------------")
    for j in range(1,g.height+1):
      a = P.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")

def get_result(grid, obstacles, start_state, terminal_state, V, im_size, max_step):
    # print("values:")
    # print_values(V, grid)
    success = False
    trajectory = []
    p = position(start_state[0], start_state[1], im_size)
    trajectory.append(start_state)
    step = 0
    while True:
        step += 1
        if step > max_step:
            break
        (x, y) = find_way(p, V, obstacles, im_size)
        if (x,y) == (-1,-1):
            break
        trajectory.append((x, y))
        p = position(x, y, im_size)
        if (x, y) == terminal_state:
            success = True
            break
    # print()

    # print('trajectory:')
    # print(trajectory)
    reward = 0
    if len(trajectory) == 1:
        reward = -1
    elif len(trajectory) == 2:
        reward = 1
    elif len(trajectory) > 2:
        if step > max_step:
            reward -= 0.05 * (len(trajectory)-1)
        else:
            reward -= 0.05 * (len(trajectory)-2)
            reward += 1
    # print('reward:')
    # print('%.3f'%reward)
    return reward, success


class position:
    def __init__(self, x, y, im_size):
        self.pos = (x,y)
        self.left = (x,y-1)
        self.right = (x,y+1)
        self.up = (x-1,y)
        self.down = (x+1,y)
        self.leftup = (x-1,y-1)
        self.rightup = (x-1,y+1)
        self.leftdown = (x+1,y-1)
        self.rightdown = (x+1,y+1)
        self.im_size = im_size