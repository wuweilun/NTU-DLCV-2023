conda create -n flipped-vqa python=3.8
conda activate flipped-vqa

#setup.sh
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install fairscale
pip install fire
pip install sentencepiece
pip install transformers
pip install timm
pip install pandas
pip install setuptools==59.5.0
pip install pysrt

pip install gdown
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./VALOR/apex

#preinstall.sh
pip install tensorboardx
pip install toolz
pip install ipdb
pip install matplotlib
pip install pytorch_pretrained_bert
pip install easydict
pip install pyyaml

gdown 1LeMob-x5z2zKMadL4BRJWjBt7dJ-Bh7d -O data.zip
unzip ./data.zip
gdown 1ORokvQINUC-aVIPsomJeHW2rotSgiCSv -O pretrained.zip
jar xvf pretrained.zip
gdown 1M0CaR4rtOt3iEGAz7ykRtEJdkTmO5nv6 -O checkpoint.zip
unzip checkpoint.zip