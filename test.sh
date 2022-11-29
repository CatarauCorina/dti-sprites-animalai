sudo add-apt-repository ppa:deadsnakes/ppa;
sudo apt-get update;
sudo apt-get install python3.8;
pip install virtualenv;
virtualenv test --python=python3.8;
source test/bin/activate && pip -V;
pip -V;