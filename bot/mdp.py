from sortedcontainers import SortedDict
import numpy as np
import ubjson

from tqdm import tqdm

class MDPModel:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        # Qtable store negative Q values
        self.state_action_map = {}
        self.qtable = SortedDict() # ((36), (1), (1)) -> (1)
    
    def sample(self, prob = 0.5, num_samples = 1000):
        # Sample a random state from the qtable
        probs = np.zeros(num_samples)
        states = []
        counter = 0
        for key, value in self.qtable.items():
            if np.random.random() < prob:
                probs[counter] = value
                states.append(key)
                counter += 1
            if counter >= num_samples:
                break
        
        probs = np.exp(probs)
        probs /= np.sum(probs)
        idx = np.random.choice(len(states), p=probs)
        return states[idx][0]
    
    def update(self, state, action, reward, next_state):
        key = (state, action)
        if state not in self.state_action_map:
            self.state_action_map[state] = set()
        self.state_action_map[state].add(action)
        if key not in self.qtable:
            self.qtable[key] = 0
        if next_state not in self.state_action_map:
            self.qtable[key] = -reward
        else:
            action_set = self.state_action_map[next_state]
            max_q = max([self.qtable.get((next_state, a), -1e6) for a in action_set], default=0)
            self.qtable[key] = -reward + self.gamma * max_q
    
    def step(self, state, explore=0.0):
        # Sample a random action from the state
        if state not in self.state_action_map:
            return (np.random.randint(0, 36), np.random.randint(0, 36))
        if np.random.random() < explore:
            action_set = self.state_action_map[state]
            action = np.random.choice(list(action_set))
            return (state, action)
        action_set = self.state_action_map[state]
        max_q = -1e6
        best_action = None
        for action in action_set:
            q = self.qtable.get((state, action), -1e6)
            if q > max_q:
                max_q = q
                best_action = action
        if best_action is None:
            return (np.random.randint(0, 36), np.random.randint(0, 36))
        else:
            return best_action
        
    def save(self, filename):
        with open(filename, 'wb') as f:
            ubjson.dump({
                'gamma': self.gamma,
                'state_action_map': self.state_action_map,
                'qtable': self.qtable
            }, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            data = ubjson.load(f)
            self.gamma = data['gamma']
            self.state_action_map = data['state_action_map']
            self.qtable = SortedDict(data['qtable'])
    
class MDPTrainer:
    def __init__(self, env, model, explore=0.5, save_interval=1000, save_path='model.ubjson'):
        self.env = env
        self.model = model
        self.explore = explore
        self.save_interval = save_interval
        self.save_path = save_path
    
    def train(self, num_steps=10000):
        done = True
        for step in tqdm(range(num_steps)):
            if done:
                state, _ = self.env.reset()
                done = False
            action = self.model.step(state, self.explore)
            next_state, reward, done, _ = self.env.step(action)
            self.model.update(state, action, reward, next_state)
            state = next_state
            if step % self.save_interval == 0:
                self.model.save(self.save_path)
            
            if step % 200:
                done = False
                self.env.set_state(self.model.sample()) 