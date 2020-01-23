FROM python:3.6
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# Install Python packages
RUN pip --no-cache-dir install --upgrade \
    docopt \
    dpu-utils \
    ipdb \
    wandb \
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl \
    typed_ast \
    more_itertools \
    scipy \
    toolz \
    tqdm \
    pandas \
    parso \
    pytest \
    mypy

# Other python packages
RUN pip --no-cache-dir install --upgrade \
    altair==3.2.0 \
    annoy==1.16.0 \
    docopt==0.6.2 \
    dpu_utils==0.1.34 \
    ipdb==0.12.2 \
    jsonpath_rw_ext==1.2.2 \
    jupyter==1.0.0 \
    more_itertools==7.2.0 \
    numpy==1.16.5 \
    pandas==0.25.0 \
    parso==0.5.1 \
    pygments==2.4.2 \
    requests==2.22.0 \
    scipy==1.3.1 \
    SetSimilaritySearch==0.1.7 \
    toolz==0.10.0 \
    tqdm==4.34.0 \
    typed_ast==1.4.0 \
    wandb==0.8.12 \
    wget==3.2

RUN pip --no-cache-dir install --upgrade \
    ipdb

# Open Ports for TensorBoard, Jupyter, and SSH
EXPOSE 6006
EXPOSE 7654
EXPOSE 22

WORKDIR /home/dev/src
COPY src/docs/THIRD_PARTY_NOTICE.md .

COPY wandb.login.netrc /root/.netrc

CMD bash