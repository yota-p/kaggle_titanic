#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_INTERPRETER = python3
#S3_BUCKET =
GCS_BUCKET = local-abbey-244223/kaggle_titanic

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: all clean test jupyter requirements help

## Delete caches
clean:
	find . -type f -name "*~" -delete
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Run specified version: production
run:
	$(PYTHON_INTERPRETER) src/exp_${ver}*.py

## Run specified version with: nomessage
runs: dir
	$(PYTHON_INTERPRETER) src/main.py ${ver} --nomsg

## Run specified version with: nomessage, debug
runsd: dir
	$(PYTHON_INTERPRETER) src/main.py ${ver} --nomsg --debug

## Start Jupyter-Notebook server
jupyter:
	./startup-jupyter.sh

## Start mlflow server
mlflow:
	./startup-mlflow.sh

## Test
test:
	python setup.py test

## Create base64 encoded script from src/
encode: clean
	python encode.py

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Install dependencies
install:
	pip install -r requirements.txt pip install -r src/requirements.txt pip install -r src/requirements.txt

## Lint using flake8
lint:
	flake8 .

## Upload Data to S3
to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(S3_BUCKET)/data/
else
	aws s3 sync data/ s3://$(S3_BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
from_s3:
ifeq (default,$(S3_PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(S3_BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Upload Data to GCS
to_gcs:
	gsutil -m rsync -d -r data/ gs://$(GCS_BUCKET)/data/

## Download Data from GCS
from_gcs:
	gsutil -m rsync -d -r gs://$(GCS_BUCKET)/data/ data/

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
