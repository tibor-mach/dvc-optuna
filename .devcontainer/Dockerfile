FROM --platform=linux/amd64 ubuntu:20.04

# set up local time zone (e.g. to avoid interactive prompt during nodejs installation)
ENV TZ=Europe/Prague
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# USERNAME is taken from docker-compose.yaml or from devcontainer.json (when used as a devcontainer)
ARG USERNAME
ARG USER_UID=1000
# ?: is it ok to have USER_UID and GID the same?
ARG USER_GID=$USER_UID

# Create the user and add them to sudoers
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -d /home/${USERNAME} \
    && apt-get update \
    && apt-get install -y sudo \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && usermod -s /bin/bash ${USERNAME}

# Set user for subsequent commands
USER ${USERNAME}

# install basic packages
RUN sudo apt-get update \
    && sudo apt-get -y upgrade \
    && sudo apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    g++ \
    git \
    gnupg2 \
    nano \
    nodejs \
    npm \
    openssh-client \
    python3.10 \
    python-is-python3 \
    python3-pip \
    wget \
    && sudo rm -rf /var/lib/apt/lists/*

# Set up Zsh (theme https://github.com/romkatv/powerlevel10)
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
    -p ssh-agent \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -a zstyle :omz:plugins:ssh-agent agent-forwarding yes

SHELL ["/bin/zsh", "-c"]

# TODO?: Figure out how to do ssh-agent forwarding from the local machine without mounting the ssh key
# this might be the issue (from terminal output during setup) ssh-agent: SSH_AUTH_SOCK not set on host.
# however, this appears even when ssh-agent is set up locally...

# add local user path to PATH
ENV PATH "${PATH}:/home/${USERNAME}/.local/bin"

# Copy python requirements and install python libraries
COPY requirements.txt requirements-dev.txt /

RUN python -m pip install -r requirements.txt
# (optional) installs jupyter and similar tools
RUN python -m pip install -r requirements-dev.txt

# TODO: Figure out a way to run this as RUN
# set up non-default hooks (these will allow DVC to operate seamlessly with just git commands)
# install the package in editable mode for local development
CMD pre-commit install --hook-type pre-push --hook-type post-checkout --hook-type pre-commit \
    && python -m pip install -e .
