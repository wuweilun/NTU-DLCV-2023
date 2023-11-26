wget -O adapter_none_4.pth 'https://www.dropbox.com/scl/fi/icg4ezn6gev7bzvoxxkka/adapter_none_4.pth?rlkey=lnicjw7l5opl0lylxb32ryajo&dl=1'

python3 -c "import clip; clip.load('ViT-L/14')"
python3 -c "import timm; timm.create_model('vit_large_patch14_clip_224.openai', pretrained=True, num_classes=0)"