from typing import Optional
from streamllama import sLlama
import fire
import torch.distributed as dist

from fastapi import FastAPI
from fastapi import Request
import uvicorn
from sse_starlette.sse import EventSourceResponse


app = FastAPI()


def my_generator(g):
    res = None
    while res != "[DONE]":
        dist.barrier()
        res = next(g)
        dist.barrier()
        yield res


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    # initialize Llama 2
    generator = sLlama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=2,
    )

    if dist.get_rank() == 0:

        @app.post("/v1/chat/completions")
        async def message_route(request: Request):
            # load messages from JSON request
            request_data = await request.json()
            dialogs = [request_data.get("messages")]

            # !!
            dist.broadcast_object_list([dialogs, max_gen_len, temperature, top_p])

            g = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            return EventSourceResponse(my_generator(g))

        uvicorn.run(app, host="0.0.0.0", port=666)

    else:
        while True:
            config = [None] * 4
            try:
                dist.broadcast_object_list(config)
                g = generator.chat_completion(
                    config[0],
                    max_gen_len=config[1],
                    temperature=config[2],
                    top_p=config[3],
                )
                res = None
                while res != "[DONE]":
                    dist.barrier()
                    res = next(g)
                    dist.barrier()
            except:
                pass


if __name__ == "__main__":
    fire.Fire(main)
