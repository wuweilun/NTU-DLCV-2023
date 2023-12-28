# Setup
## Option 1: Quickly setup
For convenience, we provide a bash script that can execute all the commands below at once. If there are any issues, please confirm each step individually.  

Please make share your device already install java to use `jar` command and use following codes to download pretrained weights.
```bash
conda create -n flipped-vqa python=3.8
conda activate flipped-vqa
sh all_setup.sh
```

## Option 2: Manual installation
### Environment (Conda)
```bash
conda create -n flipped-vqa python=3.8
conda activate flipped-vqa
sh setup.sh
pip install gdown
```
If you get error on `setup.sh`, please find newer version in torch because 30xx or 40xx GPU seems need higher version and change the first line in `setup.sh`.
Following is an example:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Dataset
```bash
gdown 1LeMob-x5z2zKMadL4BRJWjBt7dJ-Bh7d -O data.zip
unzip ./data.zip
```

### Pretrained Weight (LLAMA)
```bash
gdown 1ORokvQINUC-aVIPsomJeHW2rotSgiCSv -O pretrained.zip
jar xvf pretrained.zip
```
If you can’t download pretrained.zip, please download this link directly:
https://drive.google.com/file/d/1ORokvQINUC-aVIPsomJeHW2rotSgiCSv/view

### Our Pretrained Weight
```bash
gdown 1M0CaR4rtOt3iEGAz7ykRtEJdkTmO5nv6 -O checkpoint.zip
unzip checkpoint.zip
```

# Inference
## Method: LLAMA + Hint +Voting
```bash
bash inference.sh
```
We use a combination of different model settings to achieve better results.

The output json file named "voting_result.json" will be in the root directory

### Result
|Int_Acc| Seq_Acc | Pre_Acc | Fea_Acc | Mean |
|-----|-----|-----|-----|-----|
|65.51|68.07|58.66|50.09|60.58|

# Train