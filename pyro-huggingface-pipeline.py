#!/usr/bin/env python
# PyRo remote object for a Langchain HuggingFacePipeline
# This is example code: do not use in production!

import os
import argparse
import Pyro5.server

from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ----------------------------------------------------------------------
# Proxy object for HuggingFacePipeline
# This means the Pyro object is a proxy class for a proxy.
# NOTE: This is a boilerplate wrapper for the underlying class. Needs work.
class RemoteHuggingFacePipeline:

    def __init__(self, model_name, task="text-generation", max_tokens=1500, debug=False):
        if debug: print("[DEBUG] Instantiating Tokenizer")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if debug: print("[DEBUG] Instantiating Model")
        self._model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        if debug: print("[DEBUG] Transformer Pipeline")
        xform_pipe = pipeline( "text-generation", model=self._model, tokenizer=self._tokenizer, max_new_tokens=max_tokens )
        if debug: print("[DEBUG] HuggingFace Pipeline")
        self._pipe = HuggingFacePipeline(pipeline=xform_pipe)

    # NOTE: Pyro will not expose private methods like __call__
    def __call__(self, **kwargs):
        return self._pipe.__call__(**kwargs)
    
    @Pyro5.server.expose
    def call(self, **kwargs):
        return self._pipe.__call__(**kwargs)

    @Pyro5.server.expose
    def abatch(self, **kwargs):
        return self._pipe.abatch(**kwargs)

    @Pyro5.server.expose
    def agenerate(self, **kwargs):
        return self._pipe.agenerate(**kwargs)

    @Pyro5.server.expose
    def agenerate_prompt(self, **kwargs):
        return self._pipe.agenerate_prompt(**kwargs)

    @Pyro5.server.expose
    def ainvoke(self, **kwargs):
        return self._pipe.ainvoke(**kwargs)

    @Pyro5.server.expose
    def apredict(self, **kwargs):
        return self._pipe.apredict(**kwargs)

    @Pyro5.server.expose
    def apredict_messages(self, **kwargs):
        return self._pipe.apredict_messages(**kwargs)

    @Pyro5.server.expose
    def astream(self, **kwargs):
        return self._pipe.astream(**kwargs)

    @Pyro5.server.expose
    def astream_log(self, **kwargs):
        return self._pipe.astream_log(**kwargs)

    @Pyro5.server.expose
    def atransform(self, **kwargs):
        return self._pipe.atransform(**kwargs)

    @Pyro5.server.expose
    def batch(self, **kwargs):
        return self._pipe.batch(**kwargs)

    @Pyro5.server.expose
    def bind(self, **kwargs):
        return self._pipe.bind(**kwargs)

    # Not Supported:
    # classmethod construct(_fields_set: Optional[SetStr] = None, **values: Any) → Model
    # def copy(*, include: Optional[Union[AbstractSetIntStr, MappingIntStrAny]] = None, exclude: Optional[Union[AbstractSetIntStr, MappingIntStrAny]] = None, update: Optional[DictStrAny] = None, deep: bool = False) → Model

    @Pyro5.server.expose
    def dict(self, **kwargs):
        return self._pipe.dict(**kwargs)

    # Not Supported:
    # classmethod from_model_id(model_id: str, task: str, device: int = - 1, model_kwargs: Optional[dict] = None, pipeline_kwargs: Optional[dict] = None, **kwargs: Any) → LLM[source]
    # classmethod from_orm(obj: Any) → Model

    @Pyro5.server.expose
    def generate(self, prompts, stop=None, **kwargs):
        return self._pipe.generate(prompts, stop, **kwargs)

    @Pyro5.server.expose
    def generate_prompt(self, prompts, **kwargs):
        return self._pipe.generate_prompt(**kwargs)

    @Pyro5.server.expose
    def get_num_tokens(self, text):
        return self._pipe.get_num_tokens(text)

    @Pyro5.server.expose
    def get_num_tokens_from_messages(self, messages):
        return self._pipe.get_num_tokens_from_messages(messages)

    @Pyro5.server.expose
    def get_token_ids(self, text):
        return self._pipe.get_token_ids(text)

    @Pyro5.server.expose
    def invoke(self, **kwargs):
        return self._pipe.invoke(**kwargs)

    @Pyro5.server.expose
    def map():
        return self._pipe.map()


    # Not Supported:
    #classmethod parse_file(path: Union[str, Path], *, content_type: unicode = None, encoding: unicode = 'utf8', proto: Protocol = None, allow_pickle: bool = False) → Model
    #classmethod parse_obj(obj: Any) → Model
    #classmethod parse_raw(b: Union[str, bytes], *, content_type: unicode = None, encoding: unicode = 'utf8', proto: Protocol = None, allow_pickle: bool = False) → Model

    @Pyro5.server.expose
    def predict(self, **kwargs):
        return self._pipe.predict(**kwargs)
    #def predict(self, text, stop=None, **kwargs):
    #    return self._pipe.predict(text, stop, **kwargs)

    @Pyro5.server.expose
    def predict_messages(self, messages, stop=None, **kwargs):
        return self._pipe.predict_messages(messages, stop, **kwargs)

    # Not Supported:
    #@Pyro5.server.expose
    #def save(file_path):
    #    return self._pipe.save(file_path)

    @classmethod
    @Pyro5.server.expose
    def schema(cls, **kwargs):
        return HuggingFacePipeline.schema(**kwargs)

    @classmethod
    @Pyro5.server.expose
    def schema_json(cls, **kwargs):
        return HuggingFacePipeline.schema_json(**kwargs)

    @Pyro5.server.expose
    def stream(self, **kwargs):
        return self._pipe.stream(**kwargs)

    @Pyro5.server.expose
    def to_json(self):
        return self._pipe.to_json(**kwargs)

    # Not Supported:
    #@Pyro5.server.expose
    #def to_json_not_implemented() → SerializedNotImplemented

    @Pyro5.server.expose
    def transform(self, **kwargs):
        return self._pipe.transform(**kwargs)

    # Not Supported:
    # classmethod update_forward_refs(**localns: Any) → None
    # classmethod validate(value: Any) → Model

    @Pyro5.server.expose
    def with_config(self, **kwargs):
        return self._pipe.with_config(**kwargs)

    @Pyro5.server.expose
    def with_fallbacks(self, **kwargs):
        return self._pipe.with_fallbacks(**kwargs)

    @Pyro5.server.expose
    def with_retry(self, **kwargs):
        return self._pipe.with_retry(**kwargs)

    @Pyro5.server.expose
    @property 
    def lc_attributes(self): # Dict
        return self._pipe.lc_attributes()

    @Pyro5.server.expose
    @property 
    def lc_namespace(self): # List[str]
        return self._pipe.lc_namespace()

    @Pyro5.server.expose
    @property 
    def lc_secrets(self): # Dict[str, str]
        return self._pipe.lc_secrets()

    @Pyro5.server.expose
    @property
    def lc_serializable(self): # bool
        return self._pipe.lc_serializable()

    
# ----------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
            description='Utility to create a Langchain HuggingFacePipeline for a model and make it available on the local network via Pyro',
            epilog="Example: python pyro-huggingface-pipeline.py --host localhost --port 9999 t5-base")

    parser.add_argument('--host', dest='host', type=str, required=True,
            help='Hostname or IP address to listen on (e.g. 192.168.1.20)')

    parser.add_argument('--port', dest='port', type=int, required=True,
            help='Port to listen on (e.g. 39321)')
    
    parser.add_argument('--max-new-tokens', dest='max_new_tokens', default=1250,
            help='Max length generated content [1250]')

    parser.add_argument('--task', dest='task', default='text-generation',
            help='Task for pipeline [text-generation]')

    parser.add_argument('--debug', dest='debug', action='store_true',
            help='Enable debug output')

    parser.add_argument(dest='model_name', type=str,
            help='Name of the model to load (e.g. "tiiuae/falcon-7b-instruct")')

    args = parser.parse_args()
    return args

# ----------------------------------------------------------------------
if __name__ == '__main__':
    args = get_args()
    pipe = RemoteHuggingFacePipeline(args.model_name, args.task, args.max_new_tokens, debug=args.debug)

    if args.debug: print("[DEBUG] Configuring Pyro daemon")
    daemon = Pyro5.server.Daemon(host=args.host, port=args.port)

    if args.debug: print("[DEBUG] Registering HuggingFacePipeline with Pyro")
    uri_pipe = daemon.register(pipe, objectId='HuggingFacePipeline')

    print("Server URI (pipe):", uri_pipe)
    print("Starting Pyro daemon loop. Press Ctrl-C to exit")
    daemon.requestLoop()
