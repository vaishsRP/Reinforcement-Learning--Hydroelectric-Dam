# ProjectRL
A Reinforcement Learning (RL) approach to optimize the operation
of a hydroelectric dam for electricity production. It includes a problem description, two baseline
heuristics, the RL solution, the validation evaluation, and the performance evaluation. Additional rescarch directions and ablation studies are also discussed.
## Requirements
Make sure you are in Python version 3.11 or higher.  
Install the following dependencies:
- matplotlib
- numpy
- argparse
- pandas
- gymnasium

You can install them with:
```bash
pip install matplotlib numpy argparse pandas gymnasium
```

## Navigate to the project
Navigate to the project directory:
```bash
cd Group_24_ProjectRL_2026
```

## Run pre-trained Q-table
If you want to run the pre-trained Q-table, use: 
```bash
python3 main.py --excel_file PATH_TO_DATASET
```
### Notes
- Replace PATH_TO_DATASET with the path to your Excel dataset.
- If --excel_file argument is left empty. The program will automatically run using the validation dataset provided on Canvas.

## Run the Training code
To train the Q-table from scratch, use: 
```bash
python train.py --train_filepath PATH_TO_TRAIN_DATASET --validation_filepath DATA/validate.xlsx --num_episodes 30
```

The newly trained Q-table will automatically be evaluated on the validation set. 

### Notes
- Running 30 episodes takes you about 2 minutes on a Macbook Air M3. 
- Replace PATH_TO_TRAIN_DATASET with the path to your train dataset.
- Replace PATH_TO_VAL_DATASET with the path to your validation dataset.
- If the path arguments are left empty. The program will automatically run using the train and validation dataset provided on Canvas.
- If the --num_episodes is left empty. The program will use 30 episodes by default. 


