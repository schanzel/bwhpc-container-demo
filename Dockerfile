FROM tensorflow/tensorflow:1.8.0-gpu-py3

# Install python-tkinter, -hdf5
RUN apt-get update && apt-get install -y \
			python3-tk \
			python3-h5py

# Install Keras
RUN pip --no-cache-dir install \
      keras==2.1.6 \
      tables==3.4.3

RUN mkdir -p /opt/code
COPY . /opt/code/
WORKDIR /opt/code

ENTRYPOINT ["python", "binary_classifier_lstm.py"]

