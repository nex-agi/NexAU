# Common project workflows wrapped in handy make targets.

UV ?= uv
PACKAGE ?= nexau

.PHONY: install lint format format-check typecheck mypy mypy-coverage pyright test ci gen-llm-logging-data

install:
	$(UV) sync
	$(UV) run pre-commit install

lint:
	$(UV) run ruff check . --fix

format:
	$(UV) run ruff format

format-check:
	$(UV) run ruff format --check

typecheck: mypy pyright

mypy:
	$(UV) run mypy --config-file pyproject.toml .

mypy-coverage:
	$(UV) run mypy --config-file pyproject.toml . --cobertura-xml-report mypy_reports/type_cobertura --html-report mypy_reports/type_html

pyright:
	$(UV) run pyright

test:
	$(UV) run pytest --cov=$(PACKAGE) --cov-report=xml --cov-report=html --cov-report=term

gen-llm-logging-data:
	$(UV) run python -m tests.scripts.generate_llm_aggregator_logging_data

ci: lint format-check typecheck test
