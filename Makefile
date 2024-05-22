install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C image_processer.py

test:
	python -m pytest -vv --cov=hello test_image_processer.py

all: install lint test
