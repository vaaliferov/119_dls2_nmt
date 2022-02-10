#!/usr/bin/env bash
# https://github.com/tdlib/telegram-bot-api
# https://tdlib.github.io/telegram-bot-api/build.html?os=Linux
# https://core.telegram.org/api/obtaining_api_id
# https://my.telegram.org/apps

source secret.sh
source config.sh

if [ "$#" != 1 ]; then
    echo "usage: $0 [build|fetch|run|kill]" >&2
    exit 1
fi

if [ "$1" = "build" ]; then
    sudo apt-get update && sudo apt-get upgrade
    sudo apt-get install make git zlib1g-dev libssl-dev gperf cmake g++
    git clone --recursive https://github.com/tdlib/telegram-bot-api.git
    cd telegram-bot-api && mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=.. ..
    cmake --build . --target install
    cd ../.. && mkdir telegram 
    mv telegram-bot-api/bin/telegram-bot-api telegram
    rm -rf telegram-bot-api
fi

if [ "$1" = "fetch" ]; then
    gdown --id $TG_BOT_API_FILE_ID -O telegram/
    chmod +x telegram/telegram-bot-api
fi

if [ "$1" = "run" ]; then
    nohup ./telegram/telegram-bot-api \
        --api-id $TG_API_ID \
        --api-hash $TG_API_HASH \
        --dir ./telegram --local >/dev/null 2>&1 &
    ps aux | grep -i telegram-bot-api*
fi

if [ "$1" = "kill" ]; then
    pkill -f telegram-bot-api
    ps aux | grep -i telegram-bot-api*
fi