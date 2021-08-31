# Reinforcement learning for video games
## Implementing PPO to train an agent to play Street Fighter
### Author: Theodor Wulff
### Course: Machine Learning, Universität Hamburg

## Requirements
The project can be setup by installing the dependencies mentioned in the `requirements.txt`.
This can be done with `pip` via `pip install -r requirements.txt`.

## Environment setup
Loading the environment does not work out of the box by just installing the `retro` library.
Every available game contains a directory under `retro/data/stable/<name-of-game>`.
To add the Street fighter version I used in this project, create the directory `SuperStreetFighterII-Snes` at the mentioned location and copy the contents of the `integration` directory.

Additionally, the ROM file and the corresponding SHA1 hash file (`Super Street Fighter II (USA).sfc/.sha`) have to be renamed to `rom.sfc/.sha`.

After doing so, the file `test.py` should run without errors. 

The integration has been created using the provided integration tool and by [following the corresponding documentation](https://retro.readthedocs.io/en/latest/integration.html#integrating-a-new-rom).

The `scenario.json`/`alternate_scenario.json` define the reward function. The game returns the reward based on the content of the `scenario.json`. To use the other scenario simple rename the according files.

`RyuvsKen.state` is the savestate of the game. The agent starts at this point to play the game.

The variables the environment accesses to calculate the reward, are defined in the `data.json`.

## Project structure
The following paragraphs will describe the use of the individual scripts within this project.

#### baseline.py
Train an agent with one of the baseline implementations.

#### baselineEvaluation.py
Evaluate an agent trained with one of the baseline implementations.

#### discretizer.py
Functions to reduce the action space of the Street Fighter environment.

#### environment.py
Contains the utility function necessary for multiprocessed environments.

#### evaluation.py
Edit and run this script to evaluate an agent trained with my PPO implementation.

#### hparams.json
Hyperparameters used during training. Edit this file for experiments.

#### main.py
Puts the PPO algorithm together. Here the main loop is contained. Run this script to train an agent.

#### model.py
Contains the actor, critic and the pretrained feature extractor networks.

#### ppo.py
Contains the classes PPO, GAE and the function to interact with the environment during the inner loop of PPO.

#### replay.py
Contains functions to create video replays in .mp4 format.

#### test.py
Used to test the loading of the vectorized environment.

#### utils.py
Utility functions, that I didn't know where to put otherwise.