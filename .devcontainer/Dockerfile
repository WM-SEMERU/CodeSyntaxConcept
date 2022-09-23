## Semeru VS-Code mapping data folders and configuring DVC

FROM tensorflow/tensorflow:2.10.0-gpu-jupyter

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
ADD ./requirements.txt .
RUN apt-get update && apt-get install -y git
RUN apt-get install -y openssh-client
RUN pip3 -q install pip --upgrade
RUN pip install --upgrade pip

RUN pip install -r requirements.txt
RUN pip install dvc

#TODO: How to change the project name dynamicaly
RUN git config --global --add safe.directory /workspaces/CausalCodeCapability