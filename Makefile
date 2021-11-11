lint:
	flake8 bn3d tests gui
	mypy bn3d tests gui
coverage:
	pytest --cov=bn3d --cov-report=html
	python -m http.server 8080 --directory ./htmlcov/
test:
	pytest --cov=bn3d
clean:
	find bn3d -type f -name '*.py[co]' -delete
	find bn3d -type d -name '__pycache__' -delete
	find tests -type f -name '*.py[co]' -delete
	find tests -type d -name '__pycache__' -delete
