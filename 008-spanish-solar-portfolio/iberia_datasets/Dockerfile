# The Iberian dataset pipeline in a container: same Python, same libraries,
# wherever it runs - laptop, CI, or converted to Apptainer on an HPC cluster.
#
#   docker build -t iberia-datasets .
#   docker run --rm iberia-datasets                      # runs the test suite
#   docker run --rm -v $PWD/data:/app/data -v ~/.cdsapirc:/root/.cdsapirc:ro \
#       iberia-datasets python download_era5.py 2023-06
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY iberia ./iberia
COPY recipes ./recipes
COPY tests ./tests
COPY download_era5.py .

CMD ["python", "-m", "pytest", "tests", "-q"]
