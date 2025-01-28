class Agent:
    """
    Base Agent class. It should be inherited by specific algorithm
    implementations.
    """
    def select_action(self, state):
        raise NotImplementedError

    def update(self, transitions):
        """
        transitions can be a list of (state, action, reward, next_state, done,
        log_prob, etc.)
        """
        raise NotImplementedError

    def save_checkpoint(self, filepath):
        raise NotImplementedError
        
    def load_checkpoint(self, filepath):
        raise NotImplementedError