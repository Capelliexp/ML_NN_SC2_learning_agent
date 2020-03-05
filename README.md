# Neural network model learning simple SC2 agent combat
Agent utilizing a neural network to learn a simple "kill all enemies" trail.

![asd](https://github.com/Capelliexp/ML_NN_SC2_learning_agent/blob/master/customized_environments.egg-info/ml_sc2_screenshot.PNG)

# Instructions
* Install the Starcraft 2 game on a Windows machine
  * Create an account and download the Battle.net application from https://www.blizzard.com/en-us/apps/battle.net/desktop
	* Install the free Starcraft 2 game from within the Battle.net application (25.5 gigabytes)
		* The project was completed using Starcraft 2 version 4.11.1.77474, other versions may be supported, but is unconfirmed
* Add Starcraft 2 install folder to the Windows path variables
* Move the supplied simple_1v1.SC2Map file into <StarCraft II game location>\Maps\mini_games\
* Install Python version 3.5.2 from https://www.python.org/downloads/release/python-352/
* Install the required python packages using pip and the supplied package list
	* Run Command: "pip install -r package_list.txt"
* Download this git repository
	* My code (author: 100% me): 
		* Train and save model iterativly: trainer_dqn_mlp_std_simple.py
			* Models saved in gym_ouput/
		* Run trained model: run_model.py
			* Change model_iteration variable to switch which model iteration to run
		* Code that actually interfaces with Starcraft, creates features, and executes returned model action: \customized_environments\envs\my_agent.py
* Run command to train model
	* Run command "python trainer_dqn_mlp_std_simple.py"
  * Let the trainer work for a couple of minutes to generate a few model iterations
	* If the solution does not find the map simple_1v1 map (linkage error), rename the simple_1v1.SC2Map map in <StarCraft II game location>\Maps\mini_games\ to one of the other maps in the folder, and give the same name to the 'map_name' variable in \customized_environments\envs\my_agent.py
* Run command to evaluate model
	* Change the 'realtime' variable in my_agent.py to true 
	* Change model_iteration variable to switch which model iteration to run
	* Run command "python run_model.py"

# versions of dependency used during development
* Starcraft 2 - 4.11.3.77661
* Python - 3.5.6
* Gym - 0.15.4
* protobuf - 3.11.1
* PySC2 - 3.0.0
* stable-baselines - 2.8.0
