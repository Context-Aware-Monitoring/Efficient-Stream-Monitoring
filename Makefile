.PHONY: make_directories
make_directories:
	mkdir -p data/raw
	mkdir -p data/interim/experiment_configs
	mkdir -p data/processed/experiment_results
	mkdir -p data/processed/rewards/continous
	mkdir -p data/processed/rewards/top
	mkdir -p data/processed/rewards/threshold
	mkdir -p data/processed/context

data: make_directories
	unzip -q concurrent\ data.zip -d data/raw
	mv data/raw/concurrent\ data data/raw/concurrent_data
	unzip -q sequential_data.zip -d data/raw
