PYTHON ?= python3
PROMTOOL ?= promtool
PROFILE ?= baseline
SWEEP_DIR ?= results/mock-sweep
QUALITY_DIR ?= results/quality-$(PROFILE)
QUALITY_ENVIRONMENT ?= results/environment.json
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

.PHONY: check test smoke mock benchmark sweep report gpu-plan k8s-render monitoring-render monitoring-check quality unit
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

sweep:
	$(PYTHON) -m inference_lab.sweep --config configs/mock.json --profile mock --concurrency 1 4 8 --repetitions 3 --cooldown-s 0 --output-dir "$(SWEEP_DIR)"

report:
	$(PYTHON) -m inference_lab.report results/mock-benchmark.json --output results/report.html --csv results/summary.csv

gpu-plan:
	bash deploy/docker/run.sh "$(PROFILE)" --dry-run

k8s-render:
	@bash scripts/deploy.sh render "$(PROFILE)"

monitoring-render:
	@kubectl kustomize deploy/observability

monitoring-check:
	"$(PROMTOOL)" check config deploy/observability/prometheus.yml
	"$(PROMTOOL)" test rules tests/test_prometheus_rules.yml

quality:
	$(PYTHON) -m inference_lab.quality --config configs/quality.json --profile "$(PROFILE)" --environment "$(QUALITY_ENVIRONMENT)" --output-dir "$(QUALITY_DIR)"

unit:
	bash -n deploy/docker/run.sh
	bash -n scripts/deploy.sh
	$(PYTHON) -m unittest discover -s tests -p 'test_metrics.py' -v
	$(PYTHON) -m unittest discover -s tests -p 'test_client.py' -v
	$(PYTHON) -m unittest discover -s tests -p 'test_benchmark.py' -v
	$(PYTHON) -m unittest discover -s tests -p 'test_report.py' -v
	$(PYTHON) -m unittest discover -s tests -p 'test_launch.py' -v
	$(PYTHON) -m unittest discover -s tests -p 'test_deploy.py' -v
	$(PYTHON) -m unittest discover -s tests -p 'test_sweep.py' -v
	$(PYTHON) -m unittest discover -s tests -p 'test_quality.py' -v
