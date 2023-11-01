# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install any needed packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# First copy just the requirements.txt and install Python dependencies
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt

# Then copy the rest of the files
COPY . /app

# Assign the group of everything in the /app directory to 'root' and make it readable & executable by this group.
RUN chgrp -R 0 /app \
    && chmod -R g+rwX /app

# Allow any user to run the application (not just root or the user that owns the app directory).
USER 1001

# Expose port and define ENTRYPOINT
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
