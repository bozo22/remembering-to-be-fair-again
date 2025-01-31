# FACT

Code for the paper _"FairQCM and Beyond: Testing Memory-Augmented Fairness in Non-Markovian Settings"_.

## Repository Structure

```
FACT <- Code for the donut and lending experiments, run scripts
│   
└───covid <- Code for the COVID-19 experiments
│   
└───datasets <- Data for plots
│   
└───envs <- Donut and lending Gym environments
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
- `covid_fairscm`: COVID-19 simulation, comparing FairSCM with other baselines

## Notes
- Ensure that you have the necessary permissions to execute the scripts (`chmod +x` if required).
- The environment setup should be completed before running any experiments.
- If you encounter any issues, verify that all dependencies in `environment.yaml` are installed correctly.

