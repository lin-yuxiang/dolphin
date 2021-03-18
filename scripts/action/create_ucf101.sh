SHELL_FOLDER=$(dirname $(readlink -f "$0"))

mkdir -p SHELL_FOLDER/../../data/action/

cd SHELL_FOLDER/../../data/action/

wget -c https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate -O ./UCF101.rar
unrar x ./UCF101.rar
rm UCF101.rar

wget https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate -O UCF101TrainTestSplits-RecognitionTask.zip
unzip ./UCF101TrainTestSplits-RecognitionTask.zip -d ./
rm UCF101TrainTestSplits-RecognitionTask.zip

python SHELL_FOLDER/build_rawframes.py ./UCF-101 ./ucf101_rawframes/ --task rgb --level 2 --ext avi
echo "Generate raw frames (RGB)"