# HPML Final Project: TinyMedusa
Medusa Inference Acceleration for Tiny Llama Models
Akhil Golla (ag4812), Naren Loganathan (nl2878)

---
Description:

We implemented Medusa Inference Accelaration on small models and profiled inference time to see if this inference acceletarioin method could work in such cases. We experimented with the Maykeye/TinyLLama-v0 that has 4.62M params from HuggingFace. This model wsa trained on the roneneldan/TinyStories dataset, which has a small vocabulary with the specific intent to train small Lms.

---

Repository Outline:

- data_prepartion
   -  generate_tiny_stories : generates json file of data to train on
- 

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
