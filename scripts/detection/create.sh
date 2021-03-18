SHELL_FOLDER=$(dirname $(readlink -f "$0"))

mkdir -p SHELL_FOLDER/../../data/detection/coco2017

cd SHELL_FOLDER/../../data/detection/

wget http://images.cocodataset.org/zips/train2017.zip -O ./train2017.zip
unzip ./train2017.zip -d ./coco2017/

wget http://images.cocodataset.org/zips/test2017.zip -O ./test2017.zip
unzip ./test2017.zip -d ./coco2017/

wget http://images.cocodataset.org/zips/val2017.zip -O ./val2017.zip
unzip ./val2017.zip -d ./coco2017/

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O ./annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d ./coco2017/

rm train2017.zip test2017.zip val2017.zip annotations_trainval2017.zip