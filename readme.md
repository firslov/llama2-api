# Llama2 API

A Llama2 streaming output API with OpenAI style, support for multi-gpu inference with model of 13B or larger.

## Setup

1. Install `llama` from [official repository](https://github.com/facebookresearch/llama).

2. Download `llama2 weights` from [this repository](https://github.com/FlagAlpha/Llama2-Chinese), it's recommended to use `pth` format.

3. Clone this repo:

```shell
git clone --depth=1 https://github.com/firslov/llama2-api.git
```

4. Install requirements:

```shell
pip install -r requirements.txt
```

## Run

Set arguments in `run_api.sh`, then

```shell
./run_api.sh
```