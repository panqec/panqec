lint:
	flake8 panqec tests gui
	mypy panqec tests gui
coverage:
	pytest --cov=panqec --cov-report=html
	python -m http.server 8080 --directory ./htmlcov/
test:
	pytest --cov=panqec
clean:
	find panqec -type f -name '*.py[co]' -delete
	find panqec -type d -name '__pycache__' -delete
	find tests -type f -name '*.py[co]' -delete
	find tests -type d -name '__pycache__' -delete
