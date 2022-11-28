sudo add-apt-repository ppa:deadsnakes/ppa;
sudo apt-get update;
sudo apt-get install python3.8;

pip install virtualenv;
virtualenv sprites --python=python3.8;
source sprites/bin/activate;
python --version;
pip install seaborn;
pip install -r requirements_all.txt;
python --version;
python src/trainer.py --tag default --config clevr6.yml;
