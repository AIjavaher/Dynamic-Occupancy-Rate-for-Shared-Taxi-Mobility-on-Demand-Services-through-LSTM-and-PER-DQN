# import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from copy import deepcopy as dc
from collections import deque
import random
from torch.autograd import Variable
import cv2
import os
import xlrd
import pandas as pd


# SumTree is a popular approach to give the priority to each agent's experience in prioritized experience replay
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


######################################
# we stored state , action, reward and the next state as ( s, a, r, s_ ) in SumTree
class Memory:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            check = True
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            while check:
                if data == 0:
                    s = random.uniform(a, b)
                    (idx, p, data) = self.tree.get(s)
                else:
                    check = False

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        # print(is_weight)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


######################################
# the environment is defined with the center and neighbours of each zone
class ENV:
    def __init__(self, row, column, state):
        self.row = row
        self.column = column
        self.state = state

    def env_manhattan(self):
        neigh = xlrd.open_workbook('NYC/manhattan_neighbour.xlsx')
        self.neighsheet = neigh.sheet_by_name('Sheet1')
        center = xlrd.open_workbook('NYC/manhattan_center.xlsx')
        self.centersheet = center.sheet_by_name('Sheet1')

        self.mat = np.zeros((self.row, self.column))
        self.mat_neigh = np.zeros((self.row, self.column))
        # mht = pd.read_excel('D:\\kntu.msSummer1\\reinforcement\\aşk\\NYC\\manhattan.xlsx')
        self.lis_block = []
        self.lis_pass = []
        self.lis_center = []
        self.dic_center = {}
        self.dic_neigh = {}
        for i1 in range(0, self.row):
            for j1 in range(0, self.column):
                self.mat[i1][j1] = self.centersheet.cell(i1, j1).value
                self.mat_neigh[i1][j1] = self.neighsheet.cell(i1, j1).value

        for i in range(0, self.row):
            for j in range(0, self.column):
                v = self.mat[i][j]
                if v == 0:
                    g_b = [i, j]
                    self.lis_block.append(g_b)
                elif v == 1:
                    self.mat[i][j] = int(1)
                    g_p = [i, j]
                    self.lis_pass.append(g_p)
                else:
                    self.dic_center[v] = [i, j]
                    g_c = [i, j]
                    self.lis_center.append(g_c)

        for p in range(0, self.row):
            for q in range(0, self.column):
                value = self.mat_neigh[p][q]
                if value not in self.dic_neigh.keys():
                    self.dic_neigh[value] = []
                    self.dic_neigh[value].append([p, q])
                else:
                    self.dic_neigh[value].append([p, q])

        self.block = []
        for i in self.lis_block:
            if i[1] not in np.arange(1, self.column - 2):
                self.block.append(i)

    def neigh(self):
        goal = []
        for _ in range(1):
            matrix = np.zeros((self.row, self.column))
            gi = np.random.randint(0, self.row)
            gj = np.random.randint(0, self.column)
            g1 = [gi, gj]
            while g1 in self.lis_block:
                gi = np.random.randint(0, self.row)
                gj = np.random.randint(0, self.column)
                g1 = [gi, gj]
            i = dc(gi)
            j = dc(gj)
            g_1 = dc(g1)
            goal.append(g_1)
            while len(goal) < 2:
                impliment = True
                ind1 = np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
                if ind1 == 'N':
                    try:
                        gi -= 1
                        a = matrix[gi][gj]
                    except:
                        impliment = False
                elif ind1 == 'NE':
                    try:
                        gi -= 1
                        gj += 1
                        a = matrix[gi][gj]
                    except:
                        impliment = False
                elif ind1 == 'E':
                    try:
                        gj += 1
                        a = matrix[gi][gj]
                    except:
                        impliment = False
                elif ind1 == 'SE':
                    try:
                        gi += 1
                        gj += 1
                        a = matrix[gi][gj]
                    except:
                        impliment = False
                elif ind1 == 'S':
                    try:
                        gi += 1
                        a = matrix[gi][gj]
                    except:
                        impliment = False
                elif ind1 == 'SW':
                    try:
                        gi += 1
                        gj -= 1
                        a = matrix[gi][gj]
                    except:
                        impliment = False
                elif ind1 == 'W':
                    try:
                        gj -= 1
                        a = matrix[gi][gj]
                    except:
                        impliment = False
                elif ind1 == 'NW':
                    try:
                        gi -= 1
                        gj -= 1
                        a = matrix[gi][gj]
                    except:
                        impliment = False
                if gi < 0 or gj < 0:
                    impliment = False
                if impliment:
                    g2 = [gi, gj]
                    if g2 not in self.lis_block:
                        g_2 = dc(g2)
                        goal.append(g_2)
                    else:
                        gi = i
                        gj = j
                else:
                    gi = i
                    gj = j

        return goal

    # the state is chosen randomly for taxis and the passengers.
    # Also, we consider the satisfying two conditions for the location of the passengers if they assign to a single taxi.
    # First, their origins are in the same or neighboring zones.
    # Second, they must commute to the same or neighboring zones.
    def choose_state(self, i):
        si = np.random.randint(0, self.row)
        sj = np.random.randint(0, self.column)
        s = [si, sj]
        idx = np.random.randint(0, 2)
        if idx == 0:
            goal = self.neigh()
        else:
            goal = []
            while len(goal) < 2:
                i1 = np.random.randint(0, self.row)
                j1 = np.random.randint(0, self.column)
                g1 = [i1, j1]
                i2 = np.random.randint(0, self.row)
                j2 = np.random.randint(0, self.column)
                g2 = [i2, j2]
                while g1 in self.lis_block:
                    i1 = np.random.randint(0, self.row)
                    j1 = np.random.randint(0, self.column)
                    g1 = [i1, j1]
                goal.append(g1)
                while g2 in self.block:
                    i2 = np.random.randint(0, self.row)
                    j2 = np.random.randint(0, self.column)
                    g2 = [i2, j2]
                goal.append(g2)

        return s, goal, idx

    # the main input of the network is the passengers' location and the taxi's location. We used these locations to
    # calculate the Hamilton Distance between the passengers and taxi to feed the network

    def hammingDistance(self, n1, n2):
        x = n1 ^ n2
        setBits = 0

        while (x > 0):
            setBits += x & 1
            x >>= 1
        return setBits

    def enter_n(self, s, goal):
        state_n = np.zeros((8))
        state_n[0] = s[0]
        state_n[1] = s[1]
        state_n[2] = goal[0][0]
        state_n[3] = goal[0][1]
        state_n[4] = self.hammingDistance(s[0], goal[0][0]) + self.hammingDistance(s[1], goal[0][1])
        state_n[5] = goal[1][0]
        state_n[6] = goal[1][1]
        state_n[7] = self.hammingDistance(s[0], goal[1][0]) + self.hammingDistance(s[1], goal[1][1])
        state_n = np.expand_dims(state_n, 0)
        return state_n

    # for every action, the next state, and the reward should be calculated
    # if the next state is the passengers' location the episode will be done.
    def step(self, s, goal, action, l, ll, ind):
        state = copy.deepcopy(s)

        if action == 0:  # Nourth
            if state[0] == 0:
                next_state = state
            else:
                next_state = [state[0] - 1, state[1]]

        elif action == 1:  # Nourth_East
            if state[1] == self.column - 1 or state[0] == 0:
                next_state = state
            else:
                next_state = [state[0] - 1, state[1] + 1]

        elif action == 2:  # East
            if state[1] == self.column - 1:
                next_state = state
            else:
                next_state = [state[0], state[1] + 1]

        elif action == 3:  # South_East
            if state[1] == self.column - 1 or state[0] == self.row - 1:
                next_state = state
            else:
                next_state = [state[0] + 1, state[1] + 1]
        elif action == 4:  # South
            if state[0] == self.row - 1:
                next_state = state
            else:
                next_state = [state[0] + 1, state[1]]

        elif action == 5:  # South_West
            if state[0] == self.row - 1 or state[1] == 0:
                next_state = state
            else:
                next_state = [state[0] + 1, state[1] - 1]

        elif action == 6:  # West
            if state[1] == 0:
                next_state = state
            else:
                next_state = [state[0], state[1] - 1]

        else:  # action=7 Nourth_west
            if state[0] == 0 or state[1] == 0:
                next_state = state
            else:
                next_state = [state[0] - 1, state[1] - 1]

        if next_state in l:
            if next_state not in ll:
                ll.append(next_state)
                reward = 0
            else:
                reward = -1
        else:
            reward = -1

        if next_state in self.lis_block:
            reward = -2

        if ind == 0:
            if l[0] in ll and l[1] in ll:
                done = True
            else:
                done = False
        else:
            if l[0] in ll or l[1] in ll:
                done = True
            else:
                done = False

        return next_state, reward, done


# the Q-network as well as target Q-network
class Network(nn.Module):
    def __init__(self, n_action, n_state):
        super(Network, self).__init__()
        self.n_state = n_state
        self.n_action = n_action

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=self.n_state,
            hidden_size=128,  # rnn hidden unit
            num_layers=2,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(128, self.n_action)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


# the combination of PER approach in DQN framework
class DQN_PER:
    def __init__(self, env, row, column, n_state, n_action, gamma, buffer_size):
        self.env = env
        self.n_action = n_action
        self.row = row
        self.column = column
        self.n_state = n_state
        self.model = Network(self.n_action, self.n_state)
        self.target_model = copy.deepcopy(self.model)
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = 32
        self.buffer_size = buffer_size
        self.step_counter = 0
        self.epsilon = 1
        self.explor = 200000
        self.min_epsilon = 0.0001
        self.update_target_step = 1000
        self.l_score = []
        self.train_counter = 0
        self.memory = Memory(self.buffer_size)

    # the epsilon greedy method to choose action
    def choose_action(self, state):
        state = np.squeeze(state, 0)
        l = [0, 1, 2, 3, 4, 5, 6, 7]
        state = torch.FloatTensor(state)
        state = torch.unsqueeze(state, 0)
        if np.random.random() < self.epsilon:
            action = np.random.choice(l)
        else:
            action = torch.argmax(self.model(state)).item()
        return action

    # we trade-off between exploitation and exploration by diminishing  the epsilon greedy
    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= (1 - self.min_epsilon) / self.explor

    ###################################################################
    # we update the target network to increase the stability
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # stored the experiences and calculate the priority
    def append_sample(self, state, action, reward, next_state, done):
        self.model.eval()
        target = self.model(Variable(torch.FloatTensor(state))).data
        # old_val = target[0][action]
        old_val = target[0][action]
        target_val = self.target_model(Variable(torch.FloatTensor(next_state))).data
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + 0.99 * torch.max(target_val)

        error2 = abs(old_val - target[0][action])
        # error = 0.1 * error1 + error2
        self.memory.add(error2, (state, action, reward, next_state, done))

    # train model function
    def train_model(self):
        self.update_epsilon()
        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        mini_batch = np.array(mini_batch).transpose()
        states = np.vstack(mini_batch[0])
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.vstack(mini_batch[3])
        dones = mini_batch[4]
        # bool to binary
        dones = dones.astype(int)
        # Q function of current state
        states = torch.Tensor(states)
        states = Variable(states).float()
        pred = self.model(states)
        # one-hot encoding
        a = torch.LongTensor(actions).view(-1, 1)
        one_hot_action = torch.FloatTensor(self.batch_size, self.n_action).zero_()
        one_hot_action.scatter_(1, a, 1)
        pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)
        # Q function of next state
        next_states = torch.Tensor(next_states)
        next_states = Variable(next_states).float()
        next_pred = self.target_model(next_states).data
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        # Q Learning: get maximum Q value at s' from target model
        target = rewards + (1 - dones) * 0.99 * next_pred.max(1)[0]
        target = Variable(target)
        errors = torch.abs(pred - target).data.numpy()
        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])
        self.optimizer.zero_grad()
        # MSE Loss function
        loss = (torch.FloatTensor(is_weights) * F.mse_loss(pred, target)).mean()
        loss.backward()
        # and train
        self.optimizer.step()

        return loss.detach().item()


if __name__ == '__main__':
    row = 28
    column = 7
    ###########################
    # create the "record" folder in the PER_DQN.py address to have visualization of how the agent do.
    filelist = [f for f in os.listdir("record")]
    for f in filelist:
        os.remove(os.path.join("record", f))
    #############################
    N = row * column
    n_state = 8
    env = ENV(row, column, n_state)
    gamma = 0.99
    buffer_size = int(1e6)
    n_action = 8
    agent = DQN_HER(env, row, column, n_state, n_action, gamma, buffer_size)
    epochs = 1000

    score_r = []
    score_l = []
    score_d = []
    num = 0

    df = pd.DataFrame(columns=['epoch', 'reward', 'loss', 'done', 'epsilon'])
    epoch_df = []
    reward_df = []
    loss_df = []
    done_df = []
    epsilon_df = []

    for i in range(epochs):

        # agent.her.reset()
        env.env_manhattan()
        s, g, ind = env.choose_state(i)
        state_n = env.enter_n(s, g)

        state = copy.deepcopy(state_n)
        done = False
        rt = 0
        lt = 0
        dt = 0

        list_s = []
        l = g
        ll = []

        ##############
        # record the steps of the agent
        if i % 100 == 0 and i > 50:
            mat1 = np.zeros((row, column))
            mat1.fill(255)
            scale_percent = 2000  # percent of original size
            width = int(mat1.shape[1] * scale_percent / 100)
            height = int(mat1.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            resized = cv2.resize(mat1, dim, interpolation=cv2.INTER_AREA)
            out = 'record/image ${0}.png'.format(num)
            cv2.imwrite(out, resized)
            num += 1
        #########################
        # agent try 50 steps to find the passengers
        for t in range(50):
            action = agent.choose_action(state_n)
            new_state, reward, done = agent.env.step(s, g, action, l, ll, ind)
            rt += reward
            new_state_n = env.enter_n(new_state, g)
            agent.append_sample(
                state_n, action, reward, new_state_n, done)
            if agent.memory.tree.n_entries >= 32:
                loss = agent.train_model()
                lt += loss

            state_n = dc(new_state_n)
            s = new_state

            #############################
            if i % 100 == 0 and i > 50:
                mat = np.zeros((row, column))

                for p, k in enumerate(g):
                    if p == 0 or p == 1:
                        mat[k[0], k[1]] = 200
                    else:
                        mat[k[0], k[1]] = 250
                mat[s[0], s[1]] = 100
                scale_percent = 2000  # percent of original size
                width = int(mat.shape[1] * scale_percent / 100)
                height = int(mat.shape[0] * scale_percent / 100)
                dim = (width, height)
                # resize image
                resized = cv2.resize(mat, dim, interpolation=cv2.INTER_AREA)
                out = 'record/image ${0}.png'.format(num)
                cv2.imwrite(out, resized)
                num += 1
            #######################################

            if done:
                dt += 1
                break
        # learning rate decade
        if i == 600:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = 0.0005
        if i == 800:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = 0.0001
        if i % 100 == 0 and i != 0:
            agent.update_target_model()
            print('update q target')

        # print every 50 epoch
        score_r.append(rt)
        score_l.append(lt)
        score_d.append(dt)
        if i % 50 == 0:
            mscore_r = np.mean(score_r)
            mscore_l = np.mean(score_l)
            mscore_d = np.mean(score_d)
            print(f'episode {i} ,reward {mscore_r}, loss {mscore_l}, done {mscore_d}, epsilon {agent.epsilon}')
            epoch_df.append(i)
            reward_df.append(mscore_r)
            loss_df.append(mscore_l)
            done_df.append(mscore_d)
            epsilon_df.append(agent.epsilon)
            score_r = []
            score_l = []
            score_d = []

    #################################################
    # create the video of the agent steps
    img_array = []
    for i in range(num):
        filename = 'record/image ${0}.png'.format(i)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
        cv2.waitKey(0)

    out = cv2.VideoWriter('record/project.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    cv2.destroyAllWindows()
    out.release()
    ####################################################
    # save the process of the agent improve
    df['epoch'] = epoch_df
    df['reward'] = reward_df
    df['loss'] = loss_df
    df['epsilon'] = epsilon_df
    df.to_csv('PERDQN.csv', index=False)

# save model
torch.save(agent.model.state_dict(),'model_mht_RNN_dict.pth')
