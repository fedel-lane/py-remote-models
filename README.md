               Python Remoote Objects (PyRO) for AI Models

These are quick-and-dirty utilities to make locally-downloaded AI models
(e.g. LLMs from HuggingFace or GPT4All) available n a local network. This
is useful if you have a server or workstation with lots of
RAM/HDD/VRAM, and want it to do your actual development on a laptop or
other lightweight machine. NOT FOR USE ON PRODUCTION NETWORKS.

clients/  - sample clients for models

-- GPT4All
pyro-gpt4all-pipeline.py

-- HuggingFace
pyro-huggingface-pipeline.py
Client code:
```python
import Pyro5.client
pipe_uri = 'PYRO:HuggingFacePipeline@192.168.1.100:39321'
pipe = Pyro5.client.Proxy(pipe_uri)
pipe.predict(text='Which team did Babe Ruth play for?')
```
