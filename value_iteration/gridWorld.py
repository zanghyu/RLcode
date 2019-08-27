import numpy as np
import scipy.io as sio
import scipy
from scipy.sparse import *
from scipy import *
import random
from utils import *

class Grid: # Environment
  def __init__(self, width, height, start, obstacles):
    self.width = width
    self.height = height
    self.i = start[0]
    self.j = start[1]
    self.obstacles = obstacles

  def set(self, rewards, actions):
    # rewards should be a dict of: (i, j): r (row, col): reward
    # actions should be a dict of: (i, j): A (row, col): list of possible actions
    self.rewards = rewards
    self.actions = actions

  def set_state(self, s):
    self.i = s[0]
    self.j = s[1]

  def current_state(self):
    return (self.i, self.j)

  def is_terminal(self, s):
    return s not in self.actions

  def move(self, action):
    # check if legal move first
    if action in self.actions[(self.i, self.j)]:
      if action == 'Up':
        self.i -= 1
      elif action == 'Down':
        self.i += 1
      elif action == 'Right':
        self.j += 1
      elif action == 'Left':
        self.j -= 1
      elif action == 'LeftUp':
        self.i -= 1
        self.j -= 1
      elif action == 'RightUp':
        self.i -= 1
        self.j += 1
      elif action == 'LeftDown':
        self.i += 1
        self.j -= 1
      elif action == 'RightUp':
        self.i -= 1
        self.j += 1
    # return a reward (if any)
    return self.rewards.get((self.i, self.j), 0)

  def game_over(self):
    # returns true if game is over, else false
    # true if we are in a state where no actions are possible
    return (self.i, self.j) not in self.actions

  def all_states(self):
    # possibly buggy but simple way to get all states
    # either a position that has possible next actions
    # or a position that yields a reward
    return set(self.actions.keys()) | set(self.rewards.keys())

def standard_grid(im_size, init_state, obstacles, terminal_state, terminal_reward):
  # define a grid that describes the reward for arriving at each state
  # and possible actions at each state
  # the grid looks like this
  # x means you can't go there
  # s means start position
  # number means reward at that state
  # .  .  .  1
  # .  x  . -1
  # s  .  .  .

  g = Grid(im_size, im_size, init_state, obstacles)
  rewards = {terminal_state: terminal_reward}
  actions = {}
  for x in range(1,im_size+1):
      for y in range(1,im_size+1):
          point = position(x,y,im_size)
          if point.pos in obstacles or point.pos == terminal_state:
              continue
          cur_actions = check_action(point,obstacles,im_size)
          actions[point.pos] = cur_actions
  g.set(rewards, actions)
  return g

def negative_grid(grid, step_cost=-1):
  # in this game we want to try to minimize the number of moves
  # so we will penalize every move
  g = grid
  for pos in g.actions.keys():
      g.rewards.update({pos:step_cost})
  for pos in g.obstacles:
      g.rewards.update({pos:float('-inf')})
  return g

def get_gridworld_data(input, imsize):
    # run training from input matlab data file, and save test data prediction in output file
    # load data from Matlab file, including
    # im_data: flattened images
    # state_data: concatenated one-hot vectors for each state variable
    # state_xy_data: state variable (x,y position)
    # label_data: one-hot vector for action (state difference)
    im_size = [imsize, imsize]
    matlab_data = sio.loadmat(input)
    im_data = matlab_data["batch_im_data"]
    im_data = (im_data - 1) / 255  # obstacles = 1, free zone = 0
    value_data = matlab_data["batch_value_data"]
    Xim_data = im_data.astype('float32')
    Xim_data = Xim_data.reshape(-1, 1, im_size[0], im_size[1])
    Xval_data = value_data.astype('float32')
    Xval_data = Xval_data.reshape(-1, 1, im_size[0], im_size[1])
    Xim_data = np.squeeze(Xim_data)
    Xval_data = np.squeeze(Xval_data)

    split_len = int(6 / 7.0 * Xim_data.shape[0])

    trainim_data = Xim_data[:split_len]
    trainval_data = Xval_data[:split_len]
    testim_data = Xim_data[split_len:]
    testval_data = Xval_data[split_len:]
    return trainim_data, trainval_data, testim_data, testval_data







