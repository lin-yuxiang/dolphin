SHELL_FOLDER=$(dirname $(readlink -f "$0"))

mkdir -p SHELL_FOLDER/../../data/depth/

wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat -O SHELL_FOLDER/../../data/depth/nyu_depth_v2_labeled.mat
wget http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy -O SHELL_FOLDER/../../data/depth/NYU_ResNet-UpProj.npy