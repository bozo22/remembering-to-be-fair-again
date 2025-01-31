# FACT
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

## Notes
- Ensure that you have the necessary permissions to execute the scripts (`chmod +x` if required).
- The environment setup should be completed before running any experiments.
- If you encounter any issues, verify that all dependencies in `environment.yaml` are installed correctly.

