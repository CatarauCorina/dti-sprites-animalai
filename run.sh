sudo add-apt-repository ppa:deadsnakes/ppa;
sudo apt-get update;
sudo apt-get install python3.8;
pip install virtualenv;
chmod +x /content/drive/MyDrive/sprites/sprites/bin/pip
chmod +x /content/drive/MyDrive/sprites/sprites/bin/python
source /content/drive/MyDrive/sprites/sprites/bin/activate || virtualenv sprites --python=python3.8 && pip install -r requirements_all.txt;
pip -V;
python --version;
pip install seaborn;
export IS_SERVER="True";
pip freeze && python src/trainer.py --tag default --config clevr6.yml;
