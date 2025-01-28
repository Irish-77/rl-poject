
class EnvironmentWrapper:
    def reset(self):
        """Returns the initial observation"""
        raise NotImplementedError

    def step(self, action):
        """
        Performs action, returns next_state, reward, done, info
        """
        raise NotImplementedError