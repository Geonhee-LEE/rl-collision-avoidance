import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (frame,goal,speed, action, reward, n_frame, n_goal, n_speed, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done = map(np.stack, zip(*batch))
        return frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done

    def __len__(self):
        return len(self.buffer)
