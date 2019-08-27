# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
# Value iteration
import numpy as np
import scipy.io as sio
import scipy
from scipy.sparse import *
from scipy import *
from gridWorld import *

def main():
    SMALL_ENOUGH = 1e-3
    GAMMA = 0.9
    ALL_POSSIBLE_ACTIONS = ('Up', 'Down', 'Left', 'Right', 'LeftUp', 'RightUp', 'LeftDown', 'RightDown')
    terminal_reward = 10
    im_size = 16
    reward_list = []
    total = 0
    success = 0
    print('VI:')

    Xim_data, Xval_data, Xim_test_data, X_val_test_data = get_gridworld_data('data_set/gridworld_16.mat',im_size)
    rand_idx = np.random.permutation(np.arange(len(Xim_data)))
    # rand_idx = range(0,10)
    # print('rand_idx:'+str(rand_idx))
    for i, map_num in enumerate(rand_idx):
        total += 1
        start_state, terminal_state, obstacles = get_info(Xval_data, Xim_data, map_num, terminal_reward, im_size)

        grid = standard_grid(im_size, start_state, obstacles, terminal_state, terminal_reward)
        grid = negative_grid(grid)


        # print("rewards:")
        # print_values(grid.rewards, grid)

        # state -> action
        # choose an action and update randomly
        policy = {}
        for s in grid.actions.keys():
            policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

        V = {}
        states = grid.all_states()
        for s in states:
            V[s] = 0
        for s in grid.obstacles:
            V[s] = float('-inf')

        # this section is different from policy iteration
        # repeat until convergence
        # V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
        iteration = 0
        while True:
            iteration += 1
            biggest_change = 0
            for s in states:
                old_v = V[s]
                # V(s) only has value if it's not a terminal state
                if s in policy:
                    new_v = float('-inf')
                    for a in ALL_POSSIBLE_ACTIONS:
                        grid.set_state(s)
                        r = grid.move(a)
                        v = r + GAMMA * V[grid.current_state()]
                        if v > new_v:
                            new_v = v
                    V[s] = new_v
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))

            if biggest_change < SMALL_ENOUGH:
                break
        reward, suc_flag = get_result(grid, obstacles, start_state, terminal_state, V, im_size, im_size*2)
        reward_list.append(reward)
        if suc_flag:
            success += 1
    reward_list = np.array(reward_list)
    print("success rate: " + str(success / float(total)))
    print('reward: %.3f(+-%.3f)'%(np.mean(reward_list),np.std(reward_list)))
if __name__ == '__main__':
    main()
