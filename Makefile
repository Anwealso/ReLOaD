# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

init:
	conda env create -f environment.yml

test:
	($(CONDA_ACTIVATE) igibson ; pytest tests)

.PHONY: init test