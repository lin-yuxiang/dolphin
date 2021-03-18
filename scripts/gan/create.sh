FILE=$1
SHELL_FOLDER=$(dirname $(readlink -f "$0"))

if [[ $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" && $FILE != "mini" && $FILE != "mini_pix2pix" && $FILE != "mini_colorization" ]]; then
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1

mkdir -p SHELL_FOLDER/../../data/gan/

wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip -O SHELL_FOLDER/../../data/gan/$FILE.zip
mkdir -p SHELL_FOLDER/../../data/gan/$FILE
unzip SHELL_FOLDER/../../data/gan/$FILE.zip -d SHELL_FOLDER/../../data/gan/
rm SHELL_FOLDER/../../data/gan/$FILE.zip