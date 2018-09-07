import numpy as np
import QTable
import game_move

np.random.seed(2)
env = game_move.Game()


def random_play():
    env.reset()
    done = False
    step = 0
    while not done:
        action = np.random.randint(0, 2)
        _, _, done = env.step(action)
        step += 1
    print("random final_step =", step)


def play_by_q_leanring():
    q_table = QTable.QTable(actions=[0, 1])
    for i in range(1, 20):
        s = env.reset()
        step = 0
        done = False
        while not done:
            a = q_table.choose_action(s)
            s1, r, done = env.step(a)
            q_table.learn(s, a, r, s1, done)
            s = s1
            step += 1

        print("q learn, total_step:", step)
    print("q_table : ", q_table.q_table)


random_play()
play_by_q_leanring()
