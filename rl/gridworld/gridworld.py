class GridWorld:

    def __init__(self, size_x=5, size_y=5, x0=0, y0=0):
        self.size_x, self.size_y = max(size_x, 5), max(size_y, 5)
        self.cur_x, self.cur_y = self.x0, self.y0 = min(x0, self.size_x - 2), min(y0, self.size_y - 2)
        self.Ax, self.Ay = 1, 0 
        self.Bx, self.By = 3, 0
        self.actions = {0 : 'u', 1 : 'd', 2 : 'l', 3 : 'r'}

    def act(self, action):

        if (self.cur_x == self.Ax) and (self.cur_y == self.Ay):
            self.cur_y += 4
            return self.cur_x, self.cur_y, 10
        if (self.cur_x == self.Bx) and (self.cur_y == self.By):
            self.cur_y += 2
            return self.cur_x, self.cur_y, 5

        reward = -1

        # up
        if action == 0:
            if self.cur_y > 0:
                self.cur_y -= 1
            else:
                reward = -1
        # dn
        elif action == 1:
            if self.cur_y < self.size_y - 1:
                self.cur_y += 1
            else:
                reward = -1
        # left
        elif action == 2:
            if self.cur_x > 0:
                self.cur_x -= 1
            else:
                reward = -1
        # right
        elif action == 3:
            if self.cur_x < self.size_x - 1:
                self.cur_x += 1
            else:
                reward = -1

        return self.cur_x, self.cur_y, reward

    def reset_env(self):
        self.cur_x = self.x0
        self.cur_y = self.y0

    def print(self):
        for j in range(self.size_y):
            s = ''
            for i in range(self.size_x):
                if (i,j) == (self.cur_x, self.cur_y):
                    s += '|X'
                elif (i,j) == (self.Ax, self.Ay):
                    s += '|A'
                elif (i,j) == (self.Bx, self.By):
                    s += '|B'
                else:
                    s += '| '
            s += '|'
            print(s)
            print('-'*(2*self.size_x+1))


if __name__ == '__main__':

    grid_world = GridWorld()
    grid_world.print()
    # dn
    grid_world.act(1)
    print('\n')
    grid_world.print()
    # right
    grid_world.act(3)
    print('\n')
    grid_world.print()
    # right
    grid_world.act(3)
    print('\n')
    grid_world.print()
    # up
    grid_world.act(0)
    print('\n')
    grid_world.print()
    # up
    grid_world.act(0)
    print('\n')
    grid_world.print()
    # right
    grid_world.act(3)
    print('\n')
    grid_world.print()
    # right
    grid_world.act(3)
    print('\n')
    grid_world.print()