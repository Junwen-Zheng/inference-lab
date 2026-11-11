PYTHON ?= python3
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

.PHONY: check test smoke mock benchmark unit
check:
	$(PYTHON) -m compileall -q src tests scripts
	$(PYTHON) -c "import inference_lab; print('Package import passed')"
	$(MAKE) test PYTHON="$(PYTHON)"

test:
	$(PYTHON) -m unittest discover -s tests -v

smoke:
	$(PYTHON) scripts/smoke.py

mock:
	$(PYTHON) -m inference_lab.mock_server

benchmark:
	$(PYTHON) -m inference_lab.benchmark --config configs/mock.json --output results/mock-benchmark.json

unit:
	$(PYTHON) -m unittest discover -s tests -p 'test_metrics.py' -v
	$(PYTHON) -m unittest discover -s tests -p 'test_client.py' -v
	$(PYTHON) -m unittest discover -s tests -p 'test_benchmark.py' -v
