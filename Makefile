PYTHON ?= python3
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

.PHONY: check test smoke mock
check:
	$(PYTHON) -m compileall -q src tests
	$(PYTHON) -c "import inference_lab; print('Package import passed')"
	$(MAKE) test PYTHON="$(PYTHON)"

test:
	$(PYTHON) -m unittest discover -s tests -v

smoke:
	$(PYTHON) scripts/smoke.py

mock:
	$(PYTHON) -m inference_lab.mock_server
