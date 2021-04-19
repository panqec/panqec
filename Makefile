lint:
	flake8 bn3d tests
	mypy bn3d tests
coverage:
	pytest --cov=bn3d --cov-report=html
test:
	pytest --cov=bn3d
