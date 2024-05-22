install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C image_processor.py

test:
	python -m pytest -vv --cov=hello test_image_processor.py

all: install lint test
