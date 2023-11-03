FROM python:3.11
WORKDIR /app

# Copy your requirements file into the container
COPY requirements.txt .

# Install the packages from the requirements file
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

CMD python distortion_rate.py
