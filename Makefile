lint:
	flake8 bn3d tests gui
	mypy bn3d tests gui
coverage:
	pytest --cov=bn3d --cov-report=html
	python -m http.server 8080 --directory ./htmlcov/
test:
	pytest --cov=bn3d
