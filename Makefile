PYTHON?=python
FLAKE8_EXTRA_FLAGS?=
FLAKE8_STANDARD_FLAGS?=--count --show-source --statistics
# The GitHub editor is 127 chars wide
FLAKE8_MORE_FLAGS?=--count --max-complexity=10 --max-line-length=127 --statistics

# Python syntax errors or undefined names
.PHONY: git-lint
git-lint:
	git ls-files "*.py" | xargs flake8 --select=E9,F63,F7,F82 $(FLAKE8_STANDARD_FLAGS) $(FLAKE8_EXTRA_FLAGS)

.PHONY: git-lint-more
git-lint-more:
	git ls-files "*.py" | xargs flake8 $(FLAKE8_MORE_FLAGS) $(FLAKE8_EXTRA_FLAGS)
