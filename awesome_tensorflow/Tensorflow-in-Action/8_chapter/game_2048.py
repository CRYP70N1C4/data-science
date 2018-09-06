import numpy as np
import random, math
from matplotlib import pyplot  as plt


class Game(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.mat = np.zeros([4, 4], dtype=np.int32)
        self.__add()
        self.__add()
        return self.mat.reshape(16)

    # 0 up 1 right 2 down 3 left
    # env,reward,done,max_num
    def step(self, action):
        origin_mat = np.copy(self.mat)
        self.transpose(4 - action)
        self.__move_up()
        add_score = self.__merge_up()
        self.__move_up()
        self.transpose(action)
        if self.__changed(origin_mat) > 0:
            self.__add()
        return self.mat.reshape(16), add_score, self.__done(), np.max(self.mat)

    def __move_up(self):
        change = False
        for i in range(4):
            for j in range(4):
                for k in reversed(range(1, j + 1)):
                    if self.mat[k - 1][i] == 0 and self.mat[k][i] > 0:
                        self.mat[k - 1][i] = self.mat[k][i]
                        self.mat[k][i] = 0
                        change = True
        return change

    def __merge_up(self):
        add_score = 0
        for i in range(4):
            for j in range(1, 4):
                if self.mat[j][i] > 0 and self.mat[j][i] == self.mat[j - 1][i]:
                    self.mat[j - 1][i] *= 2
                    self.mat[j][i] = 0
                    add_score += self.mat[j - 1][i]
        return add_score;

    def __changed(self, mat):
        for i in range(4):
            for j in range(4):
                if self.mat[i][j] != mat[i][j]:
                    return True
        return False

    def transpose(self, n=1):
        self.mat = np.rot90(self.mat, k=n, axes=(1, 0))

    def __done(self):
        for i in range(4):
            for j in range(1, 4):
                if self.mat[i][j] == 0 or self.mat[i][j - 1] == 0 or self.mat[i][j - 1] == self.mat[i][j]:
                    return False
                if self.mat[j][i] == 0 or self.mat[j - 1][i] == 0 or self.mat[j - 1][i] == self.mat[j][i]:
                    return False
        return True

    # env,space,reward,done,max_num
    def __evaluate_all(self, mat):
        done = self.__done()
        r1, sp1, max_num = Game.__evaluate(self.mat)
        r2, sp2, _ = Game.__evaluate(mat)
        return self.mat.reshape(16), sp2 - sp1, r1 - r2, done, max_num

    @staticmethod
    def __evaluate(mat):
        reward_sum = 0
        space = 0
        max_num = 0
        for i in range(4):
            for j in range(4):
                if mat[i][j] > 0:
                    reward_sum += mat[i][j]
                    if mat[i][j] > max_num:
                        max_num = mat[i][j]

                else:
                    space += 1
        return reward_sum, space, max_num

    def __add(self):
        val = 2 ** random.randint(1, 2)
        while True:
            x = random.randint(0, 3)
            y = random.randint(0, 3)
            if (self.mat[x][y] == 0):
                self.mat[x][y] = val
                break;

    def __str__(self):
        return np.array2string(self.mat)


def random_play():
    g = Game()
    score_map = dict()
    for step in range(500000):
        g.reset()
        done = False
        while not done:
            env, space, reward, done, max__num = g.step(random.randint(0, 3))
            if done:
                if max__num not in score_map:
                    score_map[max__num] = 1
                else:
                    score_map[max__num] = score_map[max__num] + 1

    print(score_map)


def plot():
    score_map = {128: 205641, 64: 214715, 32: 51630, 256: 25337, 16: 2612, 8: 23, 512: 42}
    x = [i for i in score_map.keys()]
    x.sort()
    y = [score_map[i] for i in x]
    x = [math.log(i, 2) for i in x]
    plt.plot(x, y, "r*")

    # plt.bar(x, y)
    plt.show()

# plot()
