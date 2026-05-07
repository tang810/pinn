wget --content-disposition "https://dataserv.ub.tum.de/s/m1470791/download?path=%2F&files=data_6k.tar.gz"
tar -xvzf data_6k.tar.gz

mkdir weights
cd weights

python -m gdown https://drive.google.com/uc\?id\=1OvZJ0LkXiSAlxCeAnR82S4GclupYCvC2
python -m gdown https://drive.google.com/uc\?id\=1Ai9cvuqpZz4omvj7IaEGR7KimZjyzE9v
python -m gdown https://drive.google.com/uc\?id\=1t53s3ZEb2Yvzk-bLvVxLja6q9ZwU9q01
