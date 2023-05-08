lint:
	flake8 panqec tests
	mypy panqec tests
coverage:
	pytest --cov=panqec --cov-report=html
	python -m http.server 8080 --directory ./htmlcov/
test:
	pytest --cov=panqec --slow
test-fast:
	pytest --cov=panqec
clean:
	find panqec -type f -name '*.py[co]' -delete
	find panqec -type d -name '__pycache__' -delete
	find tests -type f -name '*.py[co]' -delete
	find tests -type d -name '__pycache__' -delete
doc:
	sphinx-build -b html docs/ docs/_build/html
