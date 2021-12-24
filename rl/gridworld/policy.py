from gridworld import GridWorld
import numpy as np

class Policy:
    
    def __init__(self):
        pass

    def get_action(self, x, y):
        pass

class RandomPolicy(Policy):

    def __init__(self):
        super().__init__()

    def get_action(self, x, y):
        return np.random.randint(4)

class EpsGreedyPolicy(Policy):

    def __init__(self, V, epsilon=0.1):
        super().__init__()
        self.V = V
        self.epsilon = epsilon
        self.actions = {0 : 'u', 1 : 'd', 2 : 'l', 3 : 'r'}

    def get_action(self, x, y):
    
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        else:
            V_max = float('-inf')
            if y > 0:
                V_max = max(V_max, self.V[x,y-1])
            if y < self.V.shape[1] - 1:
                V_max = max(V_max, self.V[x,y+1])
            if x > 0:
                V_max = max(V_max, self.V[x-1,y])
            if x < self.V.shape[0] - 1:
                V_max = max(V_max, self.V[x+1,y])

            V_maxes = []
            if (y > 0) and (V_max == self.V[x,y-1]):
                V_maxes.append(0)
            if (y < self.V.shape[1] - 1) and (V_max == self.V[x,y+1]):
                V_maxes.append(1)
            if (x > 0) and (V_max == self.V[x-1,y]):
                V_maxes.append(2)
            if (x < self.V.shape[0] - 1) and (V_max == self.V[x+1,y]):
                V_maxes.append(3)

            idx = np.random.randint(len(V_maxes))

            return V_maxes[idx]