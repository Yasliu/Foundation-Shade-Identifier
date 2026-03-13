FROM python:3.10

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app
CMD ["python3", "-m", "uvicorn", "bridge:app", "--host", "0.0.0.0", "--port", "7860"]
