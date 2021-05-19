# Efficient-Stream-Monitoring
Can we use traces to predict which pairs/groups of metrics to track if the budget for monitoring is limited?

# Requirements
- python3
- conda
- gcc

# Setup 
First setup the conda enviroment and install the required packages:

```sh
conda create --name eff_str_mon -y
conda activate eff_str_mon
conda install pip
pip install -y -r requirements.txt
```

Next generate the required data for experiments:

```sh
make data
```