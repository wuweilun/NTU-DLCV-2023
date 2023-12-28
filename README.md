# Setup
For convenience, we have provided a bash script that can execute all the commands below at once. If there are any issues, please confirm each step individually. 
```bash
sh all_setup.sh
```

### Environment (Conda)
Please use following codes to set up your environment.
```bash
conda create -n flipped-vqa python=3.8
conda activate flipped-vqa
sh ./DLCV-Fall-2023-Final-1-catchingstar/setup.sh
pip install gdown
```
If you get error on `setup.sh`, please find newer version in torch because 30xx or 40xx GPU seems need higher version and change the first line in `setup.sh`.
Following is an example:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Dataset
Please use following codes to set up datasets.
```bash
gdown 1LeMob-x5z2zKMadL4BRJWjBt7dJ-Bh7d -O data.zip
unzip ./data.zip
```

### Pretrained Weight (LLAMA)
Please make share your device already install java to use `jar` command and use following codes to download pretrained weights.
```bash
gdown 1ORokvQINUC-aVIPsomJeHW2rotSgiCSv -O pretrained.zip
jar xvf pretrained.zip
```
If you can’t download pretrained.zip, please download this link directly:
https://drive.google.com/file/d/1ORokvQINUC-aVIPsomJeHW2rotSgiCSv/view

### Our Pretrained Weight
Please use following codes to download our pretrained weights.
```bash
gdown 1M0CaR4rtOt3iEGAz7ykRtEJdkTmO5nv6 -O checkpoint.zip
unzip checkpoint.zip
```