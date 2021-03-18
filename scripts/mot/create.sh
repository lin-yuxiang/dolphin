SHELL_FOLDER=$(dirname $(readlink -f "$0"))

mkdir -p SHELL_FOLDER/../../data/mot/

wget https://motchallenge.net/data/2DMOT2015.zip -O SHELL_FOLDER/../../data/mot/2DMOT2015.zip
wget https://motchallenge.net/data/MOT20.zip -O SHELL_FOLDER/../../data/mot/MOT20.zip

unzip SHELL_FOLDER/../../data/mot/2DMOT2015.zip -d SHELL_FOLDER/../../data/mot
unzip SHELL_FOLDER/../../data/mot/MOT20.zip -d SHELL_FOLDER/../../data/mot

mkdir SHELL_FOLDER/../../data/mot/2DMOT2015/images
mkdir SHELL_FOLDER/../../data/mot/MOT20/images

mv SHELL_FOLDER/../../data/mot/2DMOT2015/train SHELL_FOLDER/../../data/mot/2DMOT2015/images
mv SHELL_FOLDER/../../data/mot/2DMOT2015/test SHELL_FOLDER/../../data/mot/2DMOT2015/images
mkdir -p SHELL_FOLDER/../../data/mot/2DMOT2015/labels_with_ids/train

mv SHELL_FOLDER/../../data/mot/MOT20/train SHELL_FOLDER/../../data/mot/MOT20/images
mv SHELL_FOLDER/../../data/mot/MOT20/test SHELL_FOLDER/../../data/mot/MOT20/images
mkdir -p SHELL_FOLDER/../../data/mot/MOT20/labels_with_ids/train

python3 SHELL_FOLDER/gen_labels_15.py
python3 SHELL_FOLDER/gen_labels_20.py