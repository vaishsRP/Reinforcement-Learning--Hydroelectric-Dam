import gymnasium as gym
import numpy as np
import random
from TestEnv import HydroElectric_Test
import argparse


# Set the seed for reproducibility
np.random.seed(7)
random.seed(7)

parser = argparse.ArgumentParser()
parser.add_argument('--train_filepath', type=str, default='DATA/train.xlsx')
parser.add_argument('--validation_filepath', type=str, default='DATA/validate.xlsx')
parser.add_argument('--num_episodes', type=int, default=30)
args = parser.parse_args()

class QAgent():

    def __init__(self, discount_rate=0.99):
        # create environments
        self.env_train = HydroElectric_Test(path_to_test_data=args.train_filepath)
        self.env_val = HydroElectric_Test(path_to_test_data=args.validation_filepath)

        self.env = self.env_train
        self.best_val_reward = -np.inf
        self.best_Qtable = None

        self.prices_1d_train = self.env_train.price_values.flatten()
        self.prices_1d_val = self.env_val.price_values.flatten()
        self.prices_1d = self.prices_1d_train

        self.discount_rate = discount_rate

        self.actions = np.array([-1.0, 0, 1.0], dtype=np.float32)
        self.action_space = len(self.actions)

        self.volume_states = 5
        self.price_states = 2  # above or below avg
        self.hour_states = 24
        self.cold_states = 2  # cold month or not

        # bin edges for digitize
        self.volume_bins = np.linspace(0, self.env.max_volume, self.volume_states + 1)  # 5 bins
        self.binary_bins = np.array([-0.5, 0.5, 1.5], dtype=np.float32)  # 2 bins
        self.hour_bins = np.arange(0.5, 24.5 + 1e-9, 1.0, dtype=np.float32)  # 24 bins

        self.bins = [
            self.volume_bins,
            self.binary_bins,
            self.hour_bins,
            self.binary_bins
        ]

    # Sets active environment to train or validate
    def _set_active_env(self, which: str):
        if which == "train":
            self.env = self.env_train
            self.prices_1d = self.prices_1d_train
        elif which in ("val", "validate", "validation"):
            self.env = self.env_val
            self.prices_1d = self.prices_1d_val
        else:
            raise ValueError(f"Unknown env selector: {which}")

    # returns 1 if price is above avg and 0 if it is below
    def price_above_24h_avg(self, idx, window=24):
        prices = self.prices_1d
        start = max(0, idx - window)
        hist = prices[start:idx + 1]  # include current
        current = prices[idx]
        avg = float(np.mean(hist)) if len(hist) > 0 else float(current)
        return 1.0 if float(current) > avg else 0.0

    # returns if its a cold month
    def is_cold_month(self, month):
        cold_months = {11, 12, 1, 2, 3}
        return 1.0 if int(month) in cold_months else 0.0

    def make_compact_state(self, obs, idx):
        volume = float(obs[0])
        hour = float(obs[2])
        month = float(obs[5])
        pab = float(self.price_above_24h_avg(idx))
        cold = float(self.is_cold_month(month))
        return np.array([volume, pab, hour, cold], dtype=np.float32)

    def discretize_state(self, state):
        digitized_state = []
        for i in range(len(self.bins)):
            raw_idx = np.digitize(state[i], self.bins[i]) - 1
            safe_idx = int(np.clip(raw_idx, 0, len(self.bins[i]) - 2))
            digitized_state.append(safe_idx)
        return digitized_state

    def calculate_energy_in_tank_in_eu(self, volume, price):
        return (volume * self.env.volume_to_MWh * price)

    def calculate_energy_in_tank(self, volume):
        return (volume * self.env.volume_to_MWh)


    def create_Q_table(self):
        self.Qtable = np.zeros((
            self.volume_states,  # Volume bins = 5
            self.price_states,  # Above/below 24h avg = 2
            self.hour_states,  # Hour bins = 24
            self.cold_states,  # Cold month = 2
            self.action_space  # Actions
        ))

    # runs ready algorithem on the given dataset
    def evaluate_greedy(self, which_env: str = "val"):
        self._set_active_env(which_env)

        self.env.counter = 0
        self.env.hour = 1
        self.env.day = 1
        self.env.volume = self.env.max_volume / 2
        obs = self.env.observation()

        done = False

        # reset inventory trackers
        _, real_price, *_ = self.env.observation()
        self.energy_in_tank = self.calculate_energy_in_tank(obs[0])
        self.money_in_tank = self.calculate_energy_in_tank_in_eu(obs[0], real_price)

        total_env_reward = 0.0

        idx = self.env.counter
        compact = self.make_compact_state(obs, idx)
        state = self.discretize_state(compact)

        while not done:
            state_tuple = tuple(state)

            # greedy action
            action_index = int(np.argmax(self.Qtable[state_tuple]))
            action = float(self.actions[action_index])

            # observe before step
            volume_before, price_before, *_ = self.env.observation()

            # step env
            next_obs, reward_env, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_env_reward += float(reward_env)

            # observe after step
            volume_after, price_after, *_ = self.env.observation()

            # inventory update
            delta_volume = volume_after - volume_before
            delta_energy = delta_volume * self.env.volume_to_MWh

            if delta_energy > 0:  # pump (
                self.energy_in_tank += delta_energy
                self.money_in_tank += (1 / 0.8) * price_before * delta_energy

            elif delta_energy < 0:  # sell
                energy_sold = -delta_energy

                avg_price_tank = (
                    self.money_in_tank / self.energy_in_tank
                    if self.energy_in_tank > 1e-9 else 0.0
                )

                cost_removed = energy_sold * avg_price_tank
                self.energy_in_tank -= energy_sold
                self.money_in_tank -= cost_removed

                if self.energy_in_tank <= 1e-9:
                    self.energy_in_tank = 0.0
                    self.money_in_tank = 0.0
            # -----------------------------------------

            # next state
            idx = self.env.counter
            compact = self.make_compact_state(next_obs, idx)
            state = self.discretize_state(compact)

        print(f"Greedy evaluation on {which_env} (env reward): {total_env_reward}")
        return total_env_reward


    def train(self, episode, learning_rate, epsilon=0.05, epsilon_decay=1000, adaptive_epsilon=True,
              adapting_learning_rate=False):

        #always train on train env
        self._set_active_env("train")

        self.rewards = []
        self.rewards_official = []
        self.average_rewards = []
        self.average_rewards_official = []
        self.money_in_tank = 0.0
        self.energy_in_tank = 0.0

        self.train_error = []
        self.val_error = []
        self.train_greedy = []

        self.create_Q_table()

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.epsilon_start = 1
        self.epsilon_end = 0.05

        self.lr0 = learning_rate
        self.learning_rate = learning_rate

        if adapting_learning_rate:
            self.learning_rate = 1

        for i in range(episode):

            print(f'Please wait, the algorithm is learning! The current episode is {i}')

            #TRAIN RESET (train env)
            self._set_active_env("train")
            self.env.counter = 0
            self.env.hour = 1
            self.env.day = 1
            self.env.volume = self.env.max_volume / 2
            obs = self.env.observation()

            _, real_price, *_ = self.env.observation()

            idx = self.env.counter  # should be 0 right after reset

            self.cash = 0.0
            self.money_in_tank = self.calculate_energy_in_tank_in_eu(obs[0], real_price)
            self.energy_in_tank = self.calculate_energy_in_tank(obs[0])

            done = False

            compact = self.make_compact_state(obs, idx)
            state = self.discretize_state(compact)

            total_rewards = 0
            total_reward_official = 0

            if adaptive_epsilon:
                self.epsilon = np.interp(i, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

                if i % 500 == 0 and i <= 1500:
                    print(f"The current epsilon rate is {self.epsilon}")

            if i > self.epsilon_decay:
                self.learning_rate = self.lr0 * 0.05
            else:
                self.learning_rate = self.lr0

            while not done:
                state_tuple = tuple(state)

                # Epsilon-greedy action selection
                if np.random.uniform(0, 1) < self.epsilon:
                    action_index = np.random.randint(self.action_space)
                else:
                    action_index = int(np.argmax(self.Qtable[state_tuple]))

                action = float(self.actions[action_index])

                volume_before, price_before, *_ = self.env.observation()
                inv_before = 0.9 * price_before * self.energy_in_tank - self.money_in_tank

                next_obs, reward_official, terminated, truncated, info = self.env.step(action)

                volume_after, price_after, *_ = self.env.observation()

                delta_volume = volume_after - volume_before
                delta_energy = delta_volume * self.env.volume_to_MWh

                if delta_energy > 0:  #pump (buy)
                    self.energy_in_tank += delta_energy
                    self.money_in_tank += (1 / 0.8) * price_before * delta_energy
                    reward = 0.0

                elif delta_energy < 0:  #sell
                    energy_sold = -delta_energy

                    if self.energy_in_tank > 1e-9:
                        avg_price_tank = self.money_in_tank / self.energy_in_tank
                    else:
                        avg_price_tank = 0.0

                    cost_removed = energy_sold * avg_price_tank
                    revenue = 0.9 * price_before * energy_sold

                    reward = revenue - cost_removed

                    self.energy_in_tank -= energy_sold
                    self.money_in_tank -= cost_removed

                    if self.energy_in_tank <= 1e-9:
                        self.energy_in_tank = 0.0
                        self.money_in_tank = 0.0

                else:
                    reward = 0.0

                inv_after = 0.9 * price_after * self.energy_in_tank - self.money_in_tank

                # potential-based shaping
                reward += self.discount_rate * inv_after - inv_before
                reward = reward / 1000

                done = terminated or truncated

                #discretize next state
                idx = self.env.counter
                next_compact = self.make_compact_state(next_obs, idx)
                next_state = self.discretize_state(next_compact)
                next_state_tuple = tuple(next_state)

                reward = float(reward)

                if done:
                    Q_target = reward
                else:
                    Q_target = reward + self.discount_rate * np.max(self.Qtable[next_state_tuple])

                current_q = self.Qtable[state_tuple][action_index]
                self.Qtable[state_tuple][action_index] = current_q + self.learning_rate * (Q_target - current_q)

                total_rewards += reward
                total_reward_official += reward_official
                state = next_state

            if adapting_learning_rate:
                self.learning_rate = self.learning_rate / np.sqrt(i + 1)

            self.rewards.append(total_rewards)
            self.rewards_official.append(total_reward_official)

#            print(f'Total reward: {np.mean(self.rewards)}')
#            print(f'Total reward_official: {np.mean(self.rewards_official)}')

            # validate greedily on validation env
            val_reward = float(self.evaluate_greedy(which_env="val"))
            train_reward = float(self.evaluate_greedy(which_env="train"))

            if val_reward > self.best_val_reward:
                self.best_val_reward = val_reward
                self.best_Qtable = self.Qtable.copy()


        print('The simulation is done!')


        # np.save("BEST_qtable.npy", self.Qtable)
        print("Done!!, Best evaluation value is:", self.best_val_reward)


agent_standard_greedy = QAgent()
agent_standard_greedy.train(
    episode=args.num_episodes,
    learning_rate=0.03,
    epsilon=1.0,
    epsilon_decay=25,
    adaptive_epsilon=True
)

# Greedy evaluation defaults to validation env now
agent_standard_greedy.evaluate_greedy()

