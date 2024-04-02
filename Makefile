SELF := $(lastword $(MAKEFILE_LIST))
PYTHON?=poetry run python
FLAKE8?=poetry run flake8
PRE_COMMIT?=poetry run pre-commit
FLAKE8_EXTRA_FLAGS?=
FLAKE8_STANDARD_FLAGS?=--count --show-source --statistics
# The GitHub editor is 127 chars wide
FLAKE8_MORE_FLAGS?=--count --max-complexity=10 --max-line-length=127 --statistics

TEST_LOAD_EXPERIMENTS = \
	exp_max_of_n \
	exp_modular_fine_tuning \
	# exp_sorted_list \
	#

FORCE_LOAD ?= --force load
RED:=\033[0;31m
# No Color
NC:=\033[0m
GREEN:=\033[0;32m
BOLD:=$(shell tput bold 2>/dev/null || tput -Txterm-256color bold)
NORMAL:=$(shell tput sgr0 2>/dev/null || tput -Txterm-256color sgr0)
comma=,

ifeq ($(strip $(FORCE_LOAD)),--force load)
PIPE_TEE=2>&1 | tee $(1)
else
PIPE_TEE=
endif

define add_target
# $(1) main target
# $(2) intermediate target
# $(3) recipe
$(1): $(1)-$(2)
$(1)-report: $(1)-$(2).retcode
$(1)-print-report-success: $(1)-$(2)-print-report-success
$(1)-print-report-failure: $(1)-$(2)-print-report-failure
$(1)-check: $(1)-$(2)-check

$(1)-$(2).retcode $(1)-$(2):
	{ $(3); RV=$$$$?; echo $$$$RV > $(1)-$(2).retcode; exit $$$$RV; } $(call PIPE_TEE,$(1)-$(2).log)

.PHONY: $(1)-$(2)-print-report $(1)-$(2)-print-report-success $(1)-$(2)-print-report-failure

$(1)-$(2)-print-report: $(1)-$(2)-print-report-success $(1)-$(2)-print-report-failure

PRINT_REPORTS_FAILURE += $(1)-$(2)-print-report-failure
PRINT_REPORTS_SUCCESS += $(1)-$(2)-print-report-success
CHECKS += $(1)-$(2)-check
endef

.PHONY: test-load-experiments test-load-experiments-check test-load-experiments-report test-load-experiments-print-report test-load-experiments-print-report-success test-load-experiments-print-report-failure
$(foreach e,$(TEST_LOAD_EXPERIMENTS),$(eval $(call add_target,test-load-experiments,$(e),$(PYTHON) -m gbmi.$(e).train $(FORCE_LOAD))))
$(eval $(call add_target,test-load-experiments,max-of-2-grok,$(PYTHON) -m gbmi.exp_max_of_n.train --max-of 2 --deterministic --train-for-epochs 1500 --validate-every-epochs 1 --force-adjacent-gap 0$(comma)1 --use-log1p --training-ratio 0.04638671875 --batch-size 190 --log-matrix-interp --checkpoint-every-epochs 1 --use-end-of-sequence --log-every-n-steps 1 --use-end-of-sequence $(FORCE_LOAD)))
$(eval $(call add_target,test-load-experiments,max-of-2-grok-17,$(PYTHON) -m gbmi.exp_max_of_n.train --max-of 2 --deterministic --train-for-epochs 3000 --validate-every-epochs 1 --force-adjacent-gap 0$(comma)1$(comma)2$(comma)17 --use-log1p --lr 0.001 --betas 0.9 0.98 --weight-decay 1.0 --optimizer AdamW --training-ratio 0.099609375 --log-matrix-interp --checkpoint-every-epochs 1 --batch-size 408 --log-every-n-steps 1 --use-end-of-sequence $(FORCE_LOAD)))
$(eval $(call add_target,test-load-experiments,max-of-2-grok-17-kaiming,$(PYTHON) -m gbmi.exp_max_of_n.train --max-of 2 --deterministic --train-for-epochs 3000 --validate-every-epochs 1 --force-adjacent-gap 0$(comma)1$(comma)2$(comma)17 --use-log1p --lr 0.001 --betas 0.9 0.98 --weight-decay 1.0 --optimizer AdamW --training-ratio 0.099609375 --log-matrix-interp --checkpoint-every-epochs 1 --batch-size 408 --log-every-n-steps 1 --use-end-of-sequence --use-kaiming-init $(FORCE_LOAD)))
$(eval $(call add_target,test-load-experiments,max-of-4-123,$(PYTHON) -m gbmi.exp_max_of_n.train --max-of 4 --deterministic --seed 123 --train-for-steps 3000 --lr 0.001 --betas 0.9 0.999 --optimizer AdamW --use-log1p $(FORCE_LOAD)))

test-load-experiments-print-report test-load-experiments-report:
	@$(MAKE) -f $(SELF) --no-print-directory test-load-experiments-print-report-success
	@$(MAKE) -f $(SELF) --no-print-directory test-load-experiments-print-report-failure
	@$(MAKE) -f $(SELF) --no-print-directory test-load-experiments-check

$(PRINT_REPORTS_FAILURE) : test-load-experiments-%-print-report-failure :
	@test "$$(cat test-load-experiments-$*.retcode 2>/dev/null)" != 0 && printf '$(RED)%s\t$(BOLD)FAILED$(NORMAL)$(NC)\n' $* || true

$(PRINT_REPORTS_SUCCESS) : test-load-experiments-%-print-report-success :
	@test "$$(cat test-load-experiments-$*.retcode 2>/dev/null)" = 0 && printf '$(GREEN)%s\t$(BOLD)SUCCEEDED$(NORMAL)$(NC)\n' $* || true

$(CHECKS) : %-check :
	@exit $$(cat $*.retcode || echo 1)

# Python syntax errors or undefined names
.PHONY: git-lint
git-lint:
	git ls-files "*.py" | xargs $(FLAKE8) --select=E9,F63,F7,F82 $(FLAKE8_STANDARD_FLAGS) $(FLAKE8_EXTRA_FLAGS)

.PHONY: git-lint-more
git-lint-more:
	git ls-files "*.py" | xargs $(FLAKE8) $(FLAKE8_MORE_FLAGS) $(FLAKE8_EXTRA_FLAGS)

.PHONY: sort-mailmap
sort-mailmap:
	{ grep '^#' .mailmap; grep '^\s*$$' .mailmap; grep '^[^#]' .mailmap | sort -f; } > .mailmap.tmp
	mv .mailmap.tmp .mailmap

.PHONY: pre-commit-install
pre-commit-install:
	$(PRE_COMMIT) install

.PHONY: pre-commit
pre-commit:
	$(PRE_COMMIT) run --all-files
