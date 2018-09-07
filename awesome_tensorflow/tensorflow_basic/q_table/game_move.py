class Game():

    def __init__(self):
        self.reset()

    def reset(self):
        self.posi = 0
        self.max = 8
        return self.posi

    def step(self, action):
        if action == 0:
            self.posi = max(0, self.posi - 1)
            return self.posi, -1, self.posi >= self.max
        elif action == 1:
            self.posi += 1
            return self.posi, 1, self.posi >= self.max

    def __str__(self):
        pic = ['_'] * self.max
        pic[self.posi] = '|'
        return ''.join(pic)