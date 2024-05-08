# HPML Final Project: TinyMedusa
Medusa Inference Acceleration for Tiny Llama Models
Akhil Golla (ag4812), Naren Loganathan (nl2878)

---
Description:

We implemented Medusa Inference Accelaration on small models and profiled inference time to see if this inference acceletarioin method could work in such cases. We experimented with the Maykeye/TinyLLama-v0 that has 4.62M params from HuggingFace. This model wsa trained on the roneneldan/TinyStories dataset, which has a small vocabulary with the specific intent to train small Lms.

---

Repository Outline:

- data_prepartion
   -  generate_tiny_stories.py : generates json file of data to train on
- axolotl_training_config
   - tiny_stories_llama_1000.yml : config file to train using axolotl
- medusa_model
   - cli.py : runs client that asks for user input and outputs text generation
   - kv_cache.py : implementation of key value cache optimmization
   - medusa_model_profiled.py : base medusa model with profiling
   - modeling_llama_kv.py : base llama model with medusa attention mask and key value cache
   - utils.py : utility functions
- README.md :)

---
Generate dataset used for medusa head training from TinyStories dataset on HuggingFace:

cd data_preparation

python generate_tinystories.py

Running axolotl training script:

git clone ....

Running Inference client for medusa:

...

Running Inference client for base model:

...

---

Results:
