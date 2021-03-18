SHELL_FOLDER=$(dirname $(readlink -f "$0"))

mkdir SHELL_FOLDER/../../data/action/bmn

cd SHELL_FOLDER/../../data/action/bmn

wget https://doc-0c-4k-docs.googleusercontent.com/docs/securesc/sj735phhl8bvjegb9olpk0k6pi52tett/j2e549iqq7626dt03n377l204srrbuu5/1608563475000/11751760401767102393/14871839807242329865/1ISemndlSDS2FtqQOKL0t3Cjj9yk2yznF?e=download&authuser=0&nonce=fo1de4blc74k8&user=14871839807242329865&hash=342t0n8tau8e9ijkkb10vet4t9ajf3dj -O csv_mean_100.zip

unzip csv_mean_100.zip