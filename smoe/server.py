"""OpenAI 互換の API サーバー。mlx_lm.server に、ストリーミング版のモデルを差し込む。

uv run python -m smoe.server --model models/qwen3.6-35b-a3b-4bit-smoe --cache-gb 3.0 --prefetch-m 8 --port 8080

- /v1/chat/completions(ストリーミング応答あり)などは mlx_lm.server のものをそのまま使う。
  会話の共通部分の KV キャッシュ再利用(prompt cache)も効く。
- リクエストの "model" が何であっても、--model で指定したモデルを使う(1 モデルだけ常駐させる)。
- mlx_lm.server の他の引数(--host、--port、--prompt-cache-size など)はそのまま渡せる。
"""

import argparse
import gc
import logging
import sys

import mlx.core as mx
import mlx_lm.server as srv

from smoe import load_streaming

# prefill を何トークンずつ処理するか。大きいほど SSD の読み直しが減って速いが、一時メモリが増える。
# Qwen3.6-35B・キャッシュ 2.45GB で、2048 はピーク 5.5GB(GPU 作業領域を超える)、1024 は 4.9GB。
DEFAULT_PREFILL_STEP = "1024"


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--cache-gb", type=float, default=3.0)
    ap.add_argument("--prefetch-m", type=int, default=8, help="先読みする個数(モデルの top-k が目安)")
    ap.add_argument("--pf-threads", type=int, default=1)
    ours, rest = ap.parse_known_args()

    class StreamingModelProvider(srv.ModelProvider):
        def load(self, model_path, adapter_path=None, draft_model_path=None):
            # リクエストのモデル名は無視して、起動時に指定したモデルを使う
            return super().load("default_model", None, None)

        def _load(self, model_path, adapter_path=None, draft_model_path=None):
            self.reset()
            gc.collect()
            mx.clear_cache()
            path = self._model_map.get(model_path, model_path)
            model, tokenizer, cache = load_streaming(
                path, cache_gb=ours.cache_gb, prefetch_m=ours.prefetch_m, pf_threads=ours.pf_threads
            )
            if self.cli_args.chat_template:
                tokenizer.chat_template = self.cli_args.chat_template
            logging.info("smoe: loaded %s (cache %.2fGB, prefetch %d)", path, ours.cache_gb, ours.prefetch_m)
            self.model_key = (model_path, adapter_path, draft_model_path)
            self.model = model
            self.tokenizer = tokenizer
            self.draft_model = None
            # ストリーミング層は 1 系列ずつの処理を前提にしている
            self.is_batchable = False
            self.smoe_cache = cache

    srv.ModelProvider = StreamingModelProvider
    # 常駐プロセスなので、起動時にワイヤード上限を推奨値まで上げない(生成中は mlx-lm が一時的に設定する)
    srv.maybe_set_recommended_wired_limit = lambda: None
    if "--prefill-step-size" not in rest:
        rest += ["--prefill-step-size", DEFAULT_PREFILL_STEP]
    sys.argv = [sys.argv[0]] + rest
    srv.main()


if __name__ == "__main__":
    main()
