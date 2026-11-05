PYTHON ?= python3
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

.PHONY: check
check:
	$(PYTHON) -m compileall -q src
	$(PYTHON) -c "import inference_lab; print('Package import passed')"
