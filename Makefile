.PHONY: make_directories setup_environment
make_directories:
	mkdir -p data/raw
	mkdir -p data/interim
	mkdir -p data/processed
	mkdir -p data/external
data: make_directories
	unzip -q concurrent_data.zip -d data/raw
	mv data/raw/concurrent\ data data/raw/concurrent_data
	unzip -q sequential_data.zip -d data/raw
	cd src/data && python3 make_dataset.py --all
