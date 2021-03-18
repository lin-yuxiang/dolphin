SHELL_FOLDER=$(dirname $(readlink -f "$0"))

mkdir -p SHELL_FOLDER/../../data/segmentation/

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O SHELL_FOLDER/../../data/segmentation/VOCtrainval_11-May-2012.tar

unzip SHELL_FOLDER/../../data/segmentation/VOCtrainval_11-May-2012.tar -d SHELL_FOLDER/../../data/segmentation/

rm SHELL_FOLDER/../../data/segmentation/VOCtrainval_11-May-2012.tar