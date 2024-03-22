#!/usr/bin/env bash

source config.sh

rm -rf $DIR $SERVICE_FILE_PATH
systemctl disable --now $SERVICE_NAME

cat $SERVICE_FILE_TEMPLATE_PATH > $SERVICE_FILE_PATH
sed -i "s/<name>/$NAME/g" $SERVICE_FILE_PATH
sed -i "s/<user>/$USER/g" $SERVICE_FILE_PATH

mkdir $DIR $ENV_PATH $DATA_PATH
apt install -y python3-pip python3-venv
pip3 install wget gdown virtualenv
python3 -m venv $ENV_PATH

source $ENV_PATH/bin/activate
pip3 install cython wheel
pip3 install -r $REQ_BOT_FILE_PATH
deactivate

cp -r . $DIR
gdown --id $MODEL_FILE_ID -O $MODEL_FILE_PATH
gdown --id $SRC_TOK_FILE_ID -O $SRC_TOK_FILE_PATH
gdown --id $TRG_TOK_FILE_ID -O $TRG_TOK_FILE_PATH

chmod 755 $DIR
chown -R $USER:$USER $DIR

systemctl daemon-reload
systemctl enable --now $SERVICE_NAME