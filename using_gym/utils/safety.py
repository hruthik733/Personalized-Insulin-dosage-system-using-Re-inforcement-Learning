import numpy as np

class SafetyLayer:
    def __init__(self):
        self.hypo_threshold = 80
        self.predictive_low_threshold = 110

    def apply(self, action, state):
        # Unpack the first two values (glucose, rate) and ignore the rest
        glucose, rate_of_change, *_ = state
        
        if glucose < self.hypo_threshold or (glucose < self.predictive_low_threshold and rate_of_change < -1.0):
            # Return a zero-insulin action of the same shape as the input
            return np.zeros_like(action)
            
        return action