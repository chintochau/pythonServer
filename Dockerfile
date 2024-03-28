# Start with the CUDA base image
FROM nvidia/cuda:12.2.2-runtime-ubuntu20.04

# Install Python 3.10 and other essentials
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils curl libcudnn8 libcudnn8-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --set python /usr/bin/python3.10 
    

# Install pip using get-pip.py
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Now pip should be properly installed
# You can upgrade pip using the following command if required
# RUN python3 -m pip install --upgrade pip

# Verify pip works and show its version
RUN pip --version

# Copy the requirements file and install the required Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of your application
COPY . .

# Command to run your application
CMD ["python3", "whisper-server.py"]

# run command: docker run --gpus all -it -p 5000:5000 flask-whisper