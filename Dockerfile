FROM tensorflow/tensorflow:1.8.0-gpu-py3

# Install Keras
RUN pip --no-cache-dir install \
      keras==2.1.6 \
      h5py==2.8.0

RUN mkdir -p /opt/code
COPY . /opt/code/
WORKDIR /opt/code

ENTRYPOINT ["python", "binary_classifier_lstm.py"]

