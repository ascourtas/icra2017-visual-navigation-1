FROM ai2thor/ai2thor-base:202002191705

# Set the working directory
WORKDIR .

# Copy the file from your host to your current location
COPY requirements.txt .

# check linux distro
RUN cat /etc/os-release

# set up correct python version
RUN python3 --version # TODO: update the python version, right now its 3.5.2
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get --assume-yes install python3.7
RUN python3.7 --version

# install pip
RUN apt-get update && apt-get -y upgrade && apt-get install curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.7 get-pip.py

RUN pip3.7 install -r requirements.txt

COPY xorg.conf /etc/X11/xorg.conf

RUN apt-get -y install apt-utils
# RUN apt-get -y install fxlrg
RUN apt-get -y install xserver-xorg
RUN apt-get -y install xorg
RUN apt-get -y install openbox
# sudo apt-get install ubuntu-desktop

RUN apt-get -y install xserver-xorg-core && apt-get -y install xserver-xorg-video-dummy && apt-get -y install xauth

# RUN mv .Xauthority old.Xauthority
ENV DISPLAY ":0.0"
# RUN echo $DISPLAY
#RUN export DISPLAY="0.0"
#RUN touch ~/.Xauthority
#RUN HEXKEY="$(xauth list|grep `uname -n`)"
#RUN export HEXKEY
#RUN echo $HEXKEY
#RUN xauth add $DISPLAY . hexkey
#RUN xauth list

RUN echo "shoidnsoidjjjjjjh"
RUN awk 'END{print $1}' /etc/hosts
RUN hostname -i
RUN HOST="$(hostname -i)"
RUN export HOST
RUN echo $HOST
# TBD if this is necessary, still figuring this part out
RUN touch ~/.Xauthority
#RUN xauth generate 172.17.0.2:0.0 . trusted
RUN xauth add 172.17.0.2:0 . $(xxd -l 16 -p /dev/urandom)
RUN xauth list

# RUN apt-get -y install xdpyinfo
# RUN apt-get -y install xwininfo
RUN apt-get -y install xdotool
RUN apt-get -y install wmctrl
RUN apt-get -y install pciutils
RUN apt-get install -y x11-utils
RUN apt-get install -y x11-xserver-utils
RUN apt-get -y install x11vnc


COPY . .

# Inform Docker that the container is listening on the specified port at runtime.
# EXPOSE 8080

# Run the specified command within the container.
CMD [ "python3.7", "bot_agent.py" ]