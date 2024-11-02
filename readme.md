# Wheeled_quadruped Sim2Sim Framework

## Overview

The Wheeled Quadrupedal Sim2Sim Framework is designed to facilitate the sim2sim validation of wheeled quadrupedal robots using Mujoco as the physics engine. The framework includes modules for 
* environment setup
* state management
* command generation
* policy loading
* visualization

It provides a structured approach to testing and developing control strategies for wheeled robots.

## Setup
1. Clone the repository:
```bash
git clone https://github.com/your-username/wheeled-quadruped-sim2sim.git
cd wheeled-quadruped-sim2sim
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

To run a simulation, you can use the provided `main.py` script. This script initializes the simulation environment and runs the simulation loop. Execute the simulation with:
```bash
python main.py
```

This will run a default simulation scenario as specified in the script. Adjustments can be made by modifying the parameters in main.py or by extending the script to accept command line arguments.
Modifying the Simulation
* Environment: Change the XML file in model_xml to load different environments.
* Policy Model: Update the policy_model_path to use different trained models for the robot's control policy.
* Simulation Parameters: Adjust sim_dt and control_dt to change the simulation and control timesteps.

## License

This project is licensed under MIT License.
