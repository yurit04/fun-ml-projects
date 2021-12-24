from gridworld import GridWorld
from policy import Policy, RandomPolicy, EpsGreedyPolicy
import numpy as np

class PolicyEvaluator:

    def __init__(self, grid_world : GridWorld, policy : Policy):

        self.grid_world = grid_world
        self.policy = policy

        self.V = np.zeros((self.grid_world.size_x, self.grid_world.size_y))

    def one_update(self, alpha, gamma, verbose=False):

        x, y = self.grid_world.cur_x, self.grid_world.cur_y

        action = self.policy.get_action(x, y)

        if verbose:
            print(f'action: {self.grid_world.actions[action]}')

        new_x, new_y, R = self.grid_world.act(action)

        self.V[x,y] += alpha*(R + gamma*self.V[new_x, new_y] - self.V[x,y])

    def print(self):
        print(np.round(self.V,1))

    def evaluate(self, alpha, gamma, verbose=False):

        V_old = self.V.copy()

        i = 0
        max_error = float('inf')

        while (max_error > 0.1) and (i <= 10e6):
            self.one_update(alpha=alpha,gamma=gamma,verbose=verbose)
            if (i>0) and (i % 1000000 == 0):
                max_error = np.max(np.abs(V_old - self.V))
                print(f'max_error: {np.round(max_error,2)}')
                V_old = self.V.copy()
            i += 1
        self.print()
        print('\n')


if __name__ == '__main__':

    grid_world = GridWorld()

    random_policy = RandomPolicy()
    policy_evaluator = PolicyEvaluator(grid_world,random_policy)
    policy_evaluator.evaluate(alpha=0.0001,gamma=0.9,verbose=False)

    V = policy_evaluator.V.copy()

    for i in range(3):
        grid_world.reset_env()
        policy = EpsGreedyPolicy(V, epsilon=0.3/(i+1))
        policy_evaluator = PolicyEvaluator(grid_world,policy)
        policy_evaluator.evaluate(alpha=0.0001,gamma=0.9,verbose=False)
        V = policy_evaluator.V.copy()

    print(np.round(V,1))