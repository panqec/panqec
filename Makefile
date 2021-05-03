lint:
	flake8 bn3d tests
	mypy bn3d tests
coverage:
	pytest --cov=bn3d --cov-report=html
	python -m http.server 8080 --directory ./htmlcov/
test:
	pytest --cov=bn3d
