# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

EXPOSE 3000

COPY in_two_dimensions/ ./

# Install production dependencies.
RUN pip install -r requirements.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD exec gunicorn --bind :3000 --workers 1 --threads 8 --timeout 120 app:app
