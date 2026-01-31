from TestEnv import HydroElectric_Test
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='DATA/validate.xlsx')

args = parser.parse_args()

env = HydroElectric_Test(path_to_test_data=args.excel_file)
total_reward = []
cumulative_reward = []
q_table = np.load("DATA/BEST_qtable.npy")

actions = np.array([-1.0, 0, 1.0])
action_space = len(actions)

volume_states = 5
price_states = 2    #above or below average
hour_states = 24
cold_states = 2     #cold or warm month

volume_bins = np.linspace(0, env.max_volume, volume_states + 1)
binary_bins = np.array([-0.5, 0.5, 1.5])
hour_bins = np.arange(0.5, 24.5 + 1e-9, 1.0)

bins = [volume_bins, binary_bins, hour_bins, binary_bins]

prices_1d = env.price_values.flatten()

def is_price_above_24h_avg(day_counter, window=24):

    # Returns if the price is above or below the average price of the last 24 hours

    start = max(0, day_counter - window)
    previous = prices_1d[start:day_counter + 1]
    current = prices_1d[day_counter]
    avg = float(np.mean(previous)) if len(previous) > 0 else float(current)

    if float(current) > avg:
        return 1.0
    else:
        return 0.0

def is_cold_month(month):

    # Returns if it is a cold month

    if int(month) in {11, 12, 1, 2, 3}:
        return 1.0
    else:
        return 0.0

def extract_features(obs, day_counter):

    # Putt all important features in one list

    volume = float(obs[0])
    hour = float(obs[2])
    month = float(obs[5])
    price = float(is_price_above_24h_avg(day_counter))
    cold = float(is_cold_month(month))
    return np.array([volume, price, hour, cold])

def discretize_state(compact_state):

    # Discretenize state

    digitized_state = []
    for i in range(len(bins)):
        raw_idx = np.digitize(compact_state[i], bins[i]) - 1
        safe_idx = int(np.clip(raw_idx, 0, len(bins[i]) - 2))
        digitized_state.append(safe_idx)
    return digitized_state

observation = env.observation()

for i in range(len(prices_1d) - 1):
    # Use counter value for to extract average price per day.
    day_counter = env.counter

    # discretize states
    features_list = extract_features(observation, day_counter)
    state = discretize_state(features_list)
    state_tuple = tuple(state)

    # Select action
    action_index = int(np.argmax(q_table[state_tuple]))
    action = float(actions[action_index])

    # action = RL_agent.act(observation)
    # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    next_observation, reward, terminated, truncated, info = env.step(action)

    # Add total reward
    total_reward.append(reward)
    cumulative_reward.append(sum(total_reward))

    done = terminated or truncated
    observation = next_observation

    # Plot it
    if done:
        print('Total reward: ', sum(total_reward))
        # Plot the cumulative reward over time
        plt.plot(cumulative_reward)
        plt.xlabel('Time (Hours)')
        plt.show()