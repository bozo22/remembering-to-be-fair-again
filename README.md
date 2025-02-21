# FACT

Code for the paper _"Remembering to Be Fair Again: Reproducing Non-Markovian Fairness in Sequential Decision Making"_.

## Repository Structure

```
FACT <- Code for the experiments, run scripts
│   
└───core <- Core modules
│   
└───datasets <- Data for plots
│   
└───envs <- Donut, lending and covid Gym environments
│   
└───plots <- Plots
```


## Setup
To set up the required environment, run the following commands:
```sh
conda env create -f environment.yaml
conda activate fact
```

## Reproducing Results
To reproduce the results for different experiments, use the following commands:

### Resource Allocation Results
```sh
chmod +x reproduce_donut.sh
./reproduce_donut.sh
```

### Simulated Lending Results
```sh
chmod +x reproduce_lending.sh
./reproduce_lending.sh
```

## Running Individual Experiments
To run specific experiments, use the following command format:
```sh
./run_{environment}_{experiment_name}.sh
```
Replace `{environment}` and `{experiment_name}` with the appropriate values for your experiment.

Descriptions of the experiments:

- `donut_constant`: Resource allocation with constant stakeholder behavior
- `donut_dynamic`: Resource allocation with dynamic stakeholder behavior
- `donut_gini`: Resource allocation with Gini welfare score
- `covid_fairscm`: COVID-19 simulation, comparing FairSCM with other baselines

## Notes
- Ensure that you have the necessary permissions to execute the scripts (`chmod +x` if required).
- The environment setup should be completed before running any experiments.
- If you encounter any issues, verify that all dependencies in `environment.yaml` are installed correctly.

