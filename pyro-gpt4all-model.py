#!/usr/bin/env python
# PyRo remote object for a Langchain Gpt4All
# This is example code: do not use in production!

import os
import argparse
import Pyro5.server

from langchain.llms import GPT4All
#from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ----------------------------------------------------------------------
# Proxy object for Gpt4All
"""
param allow_download: bool = False
    If model does not exist in ~/.cache/gpt4all/, download it.
param backend: Optional[str] = None
param cache: Optional[bool] = None
param callback_manager: Optional[BaseCallbackManager] = None
param callbacks: Callbacks = None
param echo: Optional[bool] = False
    Whether to echo the prompt.
param embedding: bool = False
    Use embedding mode only.
param f16_kv: bool = False¶
    Use half-precision for key/value cache.
param logits_all: bool = False
    Return logits for all tokens, not just the last token.
param max_tokens: int = 200
    Token context window.
param metadata: Optional[Dict[str, Any]] = None
    Metadata to add to the run trace.
param model: str [Required]
    Path to the pre-trained GPT4All model file.
param n_batch: int = 8
    Batch size for prompt processing.
param n_parts: int = -1
    Number of parts to split the model into. If -1, the number of parts is automatically determined.
param n_predict: Optional[int] = 256
    The maximum number of tokens to generate.
param n_threads: Optional[int] = 4
    Number of threads to use.
param repeat_last_n: Optional[int] = 64
    Last n tokens to penalize
param repeat_penalty: Optional[float] = 1.18
    The penalty to apply to repeated tokens.
param seed: int = 0
    Seed. If -1, a random seed is used.
param stop: Optional[List[str]] = []
    A list of strings to stop generation when encountered.
param streaming: bool = False
    Whether to stream the results or not.
param tags: Optional[List[str]] = None
    Tags to add to the run trace.
param temp: Optional[float] = 0.7
    The temperature to use for sampling.
param top_k: Optional[int] = 40
    The top-k value to use for sampling.
param top_p: Optional[float] = 0.1
    The top-p value to use for sampling.
param use_mlock: bool = False
    Force system to keep model in RAM.
param verbose: bool [Optional]
    Whether to print out response text.
param vocab_only: bool = False
    Only load the vocabulary, no weights.
"""
class RemoteGpt4All:

    def __init__(self, model_path, max_tokens=1500, debug=False):
        #if debug: print("[DEBUG] Instantiating Tokenizer")
        #self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if debug: print("[DEBUG] Instantiating Model")
        # n_ctx=512, n_threads=8
        self._model = GPT4All(model=model_path, max_tokens=max_tokens)
        #xform_pipe = pipeline( "text-generation", model=self._model, tokenizer=self._tokenizer, max_new_tokens=max_tokens )

    # NOTE: Pyro will not expose private methods like __call__
    def __call__(self, prompt, **kwargs):
        return self._model.__call__(prompt, **kwargs)
    
    @Pyro5.server.expose
    def call(self, prompt, **kwargs):
        return self._model.__call__(prompt, **kwargs)

    @Pyro5.server.expose
    def abatch(self, **kwargs):
        return self._model.abatch(**kwargs)

    @Pyro5.server.expose
    def agenerate(self, **kwargs):
        return self._model.agenerate(**kwargs)

    @Pyro5.server.expose
    def agenerate_prompt(self, **kwargs):
        return self._model.agenerate_prompt(**kwargs)

    @Pyro5.server.expose
    def ainvoke(self, **kwargs):
        return self._model.ainvoke(**kwargs)

    @Pyro5.server.expose
    def apredict(self, **kwargs):
        return self._model.apredict(**kwargs)

    @Pyro5.server.expose
    def apredict_messages(self, **kwargs):
        return self._model.apredict_messages(**kwargs)

    @Pyro5.server.expose
    def astream(self, **kwargs):
        return self._model.astream(**kwargs)

    @Pyro5.server.expose
    def astream_log(self, **kwargs):
        return self._model.astream_log(**kwargs)

    @Pyro5.server.expose
    def atransform(self, **kwargs):
        return self._model.atransform(**kwargs)

    @Pyro5.server.expose
    def batch(self, **kwargs):
        return self._model.batch(**kwargs)

    @Pyro5.server.expose
    def bind(self, **kwargs):
        return self._model.bind(**kwargs)

    # Not Supported:
    # classmethod construct(_fields_set: Optional[SetStr] = None, **values: Any) → Model
    # def copy(*, include: Optional[Union[AbstractSetIntStr, MappingIntStrAny]] = None, exclude: Optional[Union[AbstractSetIntStr, MappingIntStrAny]] = None, update: Optional[DictStrAny] = None, deep: bool = False) → Model

    @Pyro5.server.expose
    def dict(self, **kwargs):
        return self._model.dict(**kwargs)

    # Not Supported:
    # classmethod from_model_id(model_id: str, task: str, device: int = - 1, model_kwargs: Optional[dict] = None, pipeline_kwargs: Optional[dict] = None, **kwargs: Any) → LLM[source]
    # classmethod from_orm(obj: Any) → Model

    @Pyro5.server.expose
    def generate(self, prompts, stop=None, **kwargs):
        return self._model.generate(prompts, stop, **kwargs)

    @Pyro5.server.expose
    def generate_prompt(self, prompts, **kwargs):
        return self._model.generate_prompt(**kwargs)

    @Pyro5.server.expose
    def get_num_tokens(self, text):
        return self._model.get_num_tokens(text)

    @Pyro5.server.expose
    def get_num_tokens_from_messages(self, messages):
        return self._model.get_num_tokens_from_messages(messages)

    @Pyro5.server.expose
    def get_token_ids(self, text):
        return self._model.get_token_ids(text)

    @Pyro5.server.expose
    def invoke(self, **kwargs):
        return self._model.invoke(**kwargs)

    @Pyro5.server.expose
    def json(self, **kwargs):
        return self._model.json(**kwargs)

    @Pyro5.server.expose
    def map():
        return self._model.map()


    # Not Supported:
    #classmethod parse_file(path: Union[str, Path], *, content_type: unicode = None, encoding: unicode = 'utf8', proto: Protocol = None, allow_pickle: bool = False) → Model
    #classmethod parse_obj(obj: Any) → Model
    #classmethod parse_raw(b: Union[str, bytes], *, content_type: unicode = None, encoding: unicode = 'utf8', proto: Protocol = None, allow_pickle: bool = False) → Model

    @Pyro5.server.expose
    def predict(self, **kwargs):
        return self._model.predict(**kwargs)
    #def predict(self, text, stop=None, **kwargs):
    #    return self._model.predict(text, stop, **kwargs)

    @Pyro5.server.expose
    def predict_messages(self, messages, stop=None, **kwargs):
        return self._model.predict_messages(messages, stop, **kwargs)

    # Not Supported:
    #@Pyro5.server.expose
    #def save(file_path):
    #    return self._model.save(file_path)

    @classmethod
    @Pyro5.server.expose
    def schema(cls, **kwargs):
        return Gpt4All.schema(**kwargs)

    @classmethod
    @Pyro5.server.expose
    def schema_json(cls, **kwargs):
        return Gpt4All.schema_json(**kwargs)

    @Pyro5.server.expose
    def stream(self, **kwargs):
        return self._model.stream(**kwargs)

    @Pyro5.server.expose
    def to_json(self):
        return self._model.to_json(**kwargs)

    # Not Supported:
    #@Pyro5.server.expose
    #def to_json_not_implemented() → SerializedNotImplemented

    @Pyro5.server.expose
    def transform(self, **kwargs):
        return self._model.transform(**kwargs)

    # Not Supported:
    # classmethod update_forward_refs(**localns: Any) → None
    # classmethod validate(value: Any) → Model

    @Pyro5.server.expose
    def with_config(self, **kwargs):
        return self._model.with_config(**kwargs)

    @Pyro5.server.expose
    def with_fallbacks(self, **kwargs):
        return self._model.with_fallbacks(**kwargs)

    @Pyro5.server.expose
    def with_retry(self, **kwargs):
        return self._model.with_retry(**kwargs)

    @Pyro5.server.expose
    @property 
    def lc_attributes(self): # Dict
        return self._model.lc_attributes()

    @Pyro5.server.expose
    @property 
    def lc_namespace(self): # List[str]
        return self._model.lc_namespace()

    @Pyro5.server.expose
    @property 
    def lc_secrets(self): # Dict[str, str]
        return self._model.lc_secrets()

    @Pyro5.server.expose
    @property
    def lc_serializable(self): # bool
        return self._model.lc_serializable()

    
# ----------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
            description='Utility to create a Langchain Gpt4All Model for a locally-saved model and make it available on the local network via Pyro',
            epilog="Example: python pyro-gpt4all-mall.py --host localhost --port 9999 t5-base")

    parser.add_argument('--host', dest='host', type=str, required=True,
            help='Hostname or IP address to listen on (e.g. 192.168.1.20)')

    parser.add_argument('--port', dest='port', type=int, required=True,
            help='Port to listen on (e.g. 39321)')
    
    parser.add_argument('--max-new-tokens', dest='max_new_tokens', default=1250,
            help='Max length generated content [1250]')

    parser.add_argument('--debug', dest='debug', action='store_true',
            help='Enable debug output')

    parser.add_argument(dest='model_path', type=str,
            help='Path to the locally-saved model (e.g. "Downloads/orca-mini-13b.ggmlv3.q4_0.bin")')

    args = parser.parse_args()
    return args

# ----------------------------------------------------------------------
if __name__ == '__main__':
    args = get_args()
    pipe = RemoteGpt4All(args.model_path, args.max_new_tokens, debug=args.debug)

    if args.debug: print("[DEBUG] Configuring Pyro daemon")
    daemon = Pyro5.server.Daemon(host=args.host, port=args.port)

    if args.debug: print("[DEBUG] Registering Gpt4AllModel with Pyro")
    uri_pipe = daemon.register(pipe, objectId='Gpt4AllModel')

    print("Server URI (pipe):", uri_pipe)
    print("Starting Pyro daemon loop. Press Ctrl-C to exit")
    daemon.requestLoop()
