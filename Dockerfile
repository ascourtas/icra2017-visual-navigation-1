FROM pytorch/pytorch:nightly-devel-cuda9.2-cudnn7

# Set the working directory.
WORKDIR .

# Copy the file from your host to your current location.
COPY requirements.txt .

# Update python and pip versions.
RUN apt-get -qq update && apt-get -qqy upgrade
RUN apt-get -qqy install xserver-xorg-core
RUN apt-get -y install xserver-xorg-video-dummy
RUN cat /etc/os-release
RUN python3 --version
RUN apt-get -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get --assume-yes install python3.7
RUN python3.7 --version
RUN apt-get update && apt-get -y upgrade && apt-get install curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.7 get-pip.py

RUN apt-get -qq update && apt-get -qqy upgrade
RUN apt-get -qqy install xserver-xorg-core
RUN apt-get -y install xserver-xorg-video-dummy
RUN apt-get -y install x11vnc
RUN apt-get -y install unzip
RUN apt-get -y install pciutils
RUN apt-get -y install software-properties-common
RUN apt-get -y install kmod
RUN apt-get -y install gcc
RUN apt-get -y install make
RUN apt-get -y install linux-headers-generic
RUN apt-get -y install wget
RUN apt-get -y install sudo
RUN apt-get -y install nano

# Install our requirements.
RUN pip3.7 install -r requirements.txt

RUN apt-get -y install libglu1-mesa-dev freeglut3-dev mesa-common-dev

COPY xorg.conf /etc/X11/xorg.conf

COPY . .

# Run the specified command within the container.
CMD [ "sh", "setup.sh" ]