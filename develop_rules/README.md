# rules_dev

개발 관련 linters / formatters 설정

## Setup development workspace

### Install docker

Some lint checkers (e.g. shellcheck) require docker daemon.

Install docker on your env.

For RHEL 8. (https://bytexd.com/how-to-install-docker-on-rhel/)

    $ sudo dnf install -y yum-utils
    $ sudo dnf config-manager --add-repo \
        https://download.docker.com/linux/centos/docker-ce.repo
    $ sudo dnf update
    $ sudo dnf install docker-ce docker-ce-cli containerd.io
    $ sudo systemctl start docker.service
    $ sudo systemctl enable docker.service

### Install git-subtree

The RHEL OS doesn't have git-subtree command on the git package.
Install git-subtree.

    $ sudo dnf install git-subtree

### Install pyenv (Optional)

We recommend to use [pyenv](https://github.com/pyenv/pyenv) to manage
local python version.

Install pyenv. (once)

    $ curl https://pyenv.run | bash

```
# Load pyenv automatically by appending
# the following to
~/.bash_profile if it exists, otherwise ~/.profile (for login shells)
and ~/.bashrc (for interactive shells) :

export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Check local python version.

    $ pyenv versions
    * system (set by /home/hyunseok/.pyenv/version)

Check available python version and install target python. (e.g. 3.8.5)

    $ pyenv install -l
    $ pyenv install -v 3.8.5
    $ pyenv global 3.8.5

### Clone this to your dev repo

Example target directory: dev

Clone it.

    $ git subtree add --prefix dev \
        https://es.naverlabs.com/robot-sw/rules_dev.git <master> --squash

Update it.

    $ git subtree pull --prefix dev \
        https://es.naverlabs.com/robot-sw/rules_dev.git <master> --squash

### Setup dev env

Create python virtual env.

    $ python -m venv <Your favorite venv dir. e.g) ~/.venv/py38>
    $ source ~/.venv/py38/bin/activate
    (py38) $

Run init-repo.sh

    (py38) $ ./dev/init-repo.sh

The linters and formatters will check your changes when you commit.

If you want to change rules for your dev repo, update rules and push changes
to dev repo only or to this.

### Run pre-commit (manually)

Check previously committed files.

    (py38) $ pre-commit run --all-files

Or check specified sub-directory only.

    (py38) $ git ls-files -- <sub-directory> | xargs pre-commit run --files

### Run pre-commit pytype (manually)

Check python types for your changed code
from base, e.g, dev to current HEAD.

    (py38) $ pre-commit run --hook-stage manual \
        --from-ref dev --to-ref HEAD pytype

Or check specified sub-directory only.

    (py38) $ git ls-files -- <sub-directory> | \
        xargs pre-commit run --hook-stage manual pytype --files
