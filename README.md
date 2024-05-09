# COMS E6998 HPML (Spring 2024) Final Project: TinyMedusa

## Medusa Inference Acceleration Profiling / Training Tiny Llama Models

 - Akhil Golla (ag4812)
 - Naren Loganathan (nl2878)

---

**Description:**

Medusa is a technique for accelerating inference on LLMs. It essentially involves attaching multiple residual blocks (heads) onto the last hidden layer of the language model, followed by fine-tuning.

<p align="center">
<img width="450" alt="Blah" src="https://github.com/akhilgolla24/TinyMedusa/assets/67233931/a300fc06-3f33-43d0-8a6a-3b821334c897">
</p>

We implemented [Medusa Inference Acceleration](https://arxiv.org/abs/2401.10774) (specifically, Medusa-1) on small models and profiled various stages of the Medusa inference process to see this inference acceleration method in action. We also experimented with [Jiayi-Pan/Tiny-Vicuna-1B](https://huggingface.co/Maykeye/TinyLLama-v0](https://huggingface.co/Jiayi-Pan/Tiny-Vicuna-1B)) (1B params) from HuggingFace. This model was trained on the [WizardVicuna](https://huggingface.co/datasets/roneneldan/TinyStories](https://github.com/melodysdreamj/WizardVicunaLM)).

The code in this repository (Medusa profiling / training / implementation) is based on that of the [original repository](https://github.com/FasterDecoding/Medusa).

---

**Repository Outline:**

```
- axolotl_training_config
   - tiny_vicuna.yml : Contains a configuration (.yml) file for training Medusa heads on a HuggingFace model using axolotl

- medusa_model
   - cli.py : Runs a FastChat client that asks for user input and outputs text generation w/ medusa models (+ profiling)
   - kv_cache.py : Implementation of the key value cache optimization
   - medusa_model_profiled.py : Medusa model definition + profiling code for the text generation phase
   - modeling_llama_kv.py : Base Llama model with support for Medusa's attention mask and key value caching
   - utils.py : Utility functions (used during Medusa inference)

- requirements.txt

- README.md
```

---

### Environment
Set up a python environment and install the required dependencies using `pip`. (Note: make the `transformers` version 4.34.0)
```bash
$ pip install -r requirements.txt
```

### Dataset
We use the ShareGPT dataset, which is a subset of the Vicuna training data. It has conversations between a human and client.
```bash
git clone https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered
```
Specifically, we use the ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json

### Training
We use a fork of the [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) library modified for Medusa support to manage the training process. Use [this](https://github.com/ctlllll/axolotl) fork for the training code.  Add a training configuration (`.yml`) to [`examples/medusa`](https://github.com/ctlllll/axolotl/tree/main/examples/medusa), and run the below command (after following the normal axolotl installation process).

```bash
$ accelerate launch -m axolotl.cli.train examples/medusa/your_config.yml
```

### Running Inference Client for Medusa (with profiling of inference stages):
```bash
$ python -m cli --model [MODEL_NAME] [--max_steps (NUMBER_OF_MEDUSA_STEPS)] [--use-medusa (inference uses medusa acceleration, without it runs base model)]
# This should open up a prompt where you can interact with the model and subsequently observe inference speeds
# e.g. python -m cli --model FasterDecoding/medusa-1.0-vicuna-7b-v1.5 --use-medusa
```
---

### Results:

**Environment Used:** GCP VM with an NVIDIA L4 32GB

We used Weights & Biases to monitor training. Here are some wandb [statistics](https://wandb.ai/narenl/medusa_test_stories_1000?nw=nwusernarenl) from our training process (Medusa heads trained over TinyLLama). We've also attached some graphs depicting the top-x accuracy or loss of certain Medusa heads (the i-th head predicting the (i + 1)-th token in a possible continuation), and how they've improved throughout the training process (cosine lr scheduler).

<img width="450" alt="Blah" src="https://github.com/akhilgolla24/TinyMedusa/assets/67233931/bf4ed161-c1d7-462e-bc35-1ef063e49d89">
<img width="450" alt="Blah" src="https://github.com/akhilgolla24/TinyMedusa/assets/67233931/775416be-3b8e-422b-8890-06f382221d43">
<img width="450" alt="Blah" src="https://github.com/akhilgolla24/TinyMedusa/assets/67233931/ecf80fd5-29c1-4f0d-a59d-fe775674f6df">
<img width="450" alt="Blah" src="https://github.com/akhilgolla24/TinyMedusa/assets/67233931/01712fed-090f-4a59-b1ed-f55e56378755">

 - Example output from profiling Medusa on vicuna-7b:
<img width="974" alt="vicuna_results" src="https://github.com/akhilgolla24/TinyMedusa/assets/60799338/d076df06-6539-4823-aec6-5acee5cadce7">


