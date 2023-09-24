               Python Remoote Objects (PyRO) for AI Models

These are quick-and-dirty utilities to make locally-downloaded AI models
(e.g. LLMs from HuggingFace or GPT4All) available n a local network. This
is useful if you have a server or workstation with lots of
RAM/HDD/VRAM, and want it to do your actual development on a laptop or
other lightweight machine. NOT FOR USE ON PRODUCTION NETWORKS.

clients/  - sample clients for models

# GPT4All
pyro-gpt4all-pipeline.py

# HuggingFace
pyro-huggingface-pipeline.py

Example:
```
bash$ python pyro-huggingface-pipeline.py --host 192.168.6.66 --port 39321 --debug tiiuae/falcon-7b-instruct
[DEBUG] Instantiating Tokenizer
[DEBUG] Instantiating Model
[DEBUG] Transformer Pipeline
[DEBUG] HuggingFace Pipeline
[DEBUG] Configuring Pyro daemon
[DEBUG] Registering HuggingFacePipeline with Pyro
Server URI (pipe): PYRO:HuggingFacePipeline@192.168.6.66:39321
Starting Pyro daemon loop. Press Ctrl-C to exit

```
Client code:
```python
import Pyro5.client
pipe_uri = 'PYRO:HuggingFacePipeline@192.168.6.66:39321'
pipe = Pyro5.client.Proxy(pipe_uri)
pipe.predict(text='Which team did Babe Ruth play for?')
# '\nBabe Ruth played for the New York Yankees.'
```
