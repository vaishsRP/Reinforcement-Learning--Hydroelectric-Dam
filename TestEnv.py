import gymnasium as gym
import numpy as np
import pandas as pd

class HydroElectric_Test(gym.Env):


    def __init__(self, path_to_test_data:str):
        # Define a discrete action space, -1 0 or 1
        self.discrete_action_space = gym.spaces.Discrete(3)
        # Define a continuous action space, -1 to 1
        self.continuous_action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Define the test data
        self.test_data = pd.read_excel(path_to_test_data)
        self.price_values = self.test_data.iloc[:, 1:25].to_numpy()
        self.timestamps = self.test_data['PRICES']
        self.counter = 0
        self.hour = 1
        self.day = 1
        self.state = np.empty(7)
        self.max_volume = 100000               # m^3
        self.volume = self.max_volume/2        # m^3
        self.max_flow = 18000                  # m^3/h
        self.pump_efficiency = 0.8             # -
        self.flow_efficiency = 0.9             # -

        self.water_mass = 1000                      # kg/m^3
        self.dam_height = 30                        # m
        self.gravity_constant = 9.81                # m/s^2
        self.volume_to_MWh = (self.water_mass*self.gravity_constant*self.dam_height)*2.77778e-10  # m^3 to MWh

    def step(self, action):
        reward = 0
        action = np.squeeze(action) # Remove the extra dimension
        # Calculate the costs and volume change when pumping water (action >0)
        if (action >0) and (self.volume <= self.max_volume):
            if (self.volume + action*self.max_flow) > self.max_volume:
                action = (self.max_volume - self.volume)/self.max_flow
            pumped_water_volume = action * self.max_flow
            pumped_water_costs = (1 / 0.8) * pumped_water_volume * self.volume_to_MWh * self.price_values[self.day-1][self.hour-1]
            reward = -pumped_water_costs
            self.volume += pumped_water_volume

        # Calculate the profits and volume change when selling water (action <0)
        elif (action < 0) and (self.volume >= 0):
            if (self.volume + action*self.max_flow) < 0:
                action = -self.volume/self.max_flow
            sold_water_volume = action * self.max_flow
            sold_water_profits = 0.9*sold_water_volume*self.volume_to_MWh*self.price_values[self.day-1][self.hour-1]
            reward = abs(sold_water_profits)
            self.volume -= abs(sold_water_volume)
        # No action (action =0)
        elif action ==0:
            reward = 0
        #volume safeguard
        self.volume = np.clip(self.volume, 0, self.max_volume)

        self.counter += 1 # Increase the counter
        self.hour += 1 # Increase the hour
        if self.counter % 24 == 0:  # If the counter is a multiple of 24, increase the day, reset hour to first hour
            self.day += 1
            self.hour = 1
        if self.counter == len(self.price_values.flatten())-1: # If the counter is equal to the number of hours in the test data, terminate the episode
            terminated = True
            truncated = True
        else: # If the counter is not equal to the number of hours in the test data, continue the episode
            terminated = False
            truncated = False
        info = {} # No info
        self.state = self.observation() # Update the state

        return self.state, reward, terminated, truncated, info

    def observation(self):  # Returns the current state
        dam_level = self.volume
        price = self.price_values[self.day -1][self.hour-1]
        hour = self.hour
        day_of_week = self.timestamps[self.day -1].dayofweek # Monday = 0, Sunday = 6
        day_of_year = self.timestamps[self.day -1].dayofyear # January 1st = 1, December 31st = 365
        month = self.timestamps[self.day -1].month # January = 1, December = 12
        year = self.timestamps[self.day -1].year
        self.state = np.array([dam_level, price, int(hour), int(day_of_week), int(day_of_year), int(month), int(year)])

        return self.state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.counter = 0
        self.hour = 1
        self.day = 1
        self.volume = self.max_volume / 2
        self.state = self.observation()
        return self.state, {}