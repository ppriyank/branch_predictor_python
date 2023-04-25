VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

all: run

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	alias sim="bash run.sh"

run: $(VENV)/bin/activate
	$(PYTHON) test.py
	
clean:	
	rm -rf __pycache__
	rm -rf venv 
