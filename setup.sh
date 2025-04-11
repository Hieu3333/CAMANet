#git clone https://github.com/Markin-Wang/CAMANet.git
git clone https://github.com/Hieu3333/CAMANet.git
cd CAMANet
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
cd CAMANet
conda env create --name .env --file env.yml
conda activate .env
# pip install pycocotools pyyaml==6.0 timm==0.5.4 torchcam==0.3.1 yacs==0.1.8 opencv-python==4.5.5.64 pycocoevalcap==1.2 ml-collections==0.1.1 easydict==1.9 contextlib2==21.6.0
pip install pyyaml==6.0 timm==0.5.4 torchcam==0.3.1 yacs==0.1.8 opencv-python==4.5.5.64 ml-collections==0.1.1 easydict==1.9 contextlib2==21.6.0
sudo apt-get install default-jre
sudo apt-get install default-jdk

# Run on local
scp -P 24730 "/mnt/c/Users/hieu3/Downloads/iu_xray.zip" root@185.150.27.254:/workspace/CAMANet #Replace ip address
scp -P 24730 "/mnt/c/Users/hieu3/Downloads/labels.json" root@185.150.27.254:/workspace/CAMANet/data/iu_xray/labels

#Continue on cloud
pip install unzip
unzip iu_xray.zip -d data
cd data/iu_xray
mkdir labels

mkdir -p /workspace/CAMANet/pycocoevalcap/meteor/data/
cd /workspace/CAMANet/pycocoevalcap/meteor/data/
wget https://raw.githubusercontent.com/cmu-mtlab/meteor/master/data/paraphrase-en.gz


