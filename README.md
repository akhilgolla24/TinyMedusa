# COMS E6998 HPML (Spring 2024) Final Project: TinyMedusa

## Medusa Inference Acceleration Profiling / Training Tiny Llama Models

 - Akhil Golla (ag4812)
 - Naren Loganathan (nl2878)

---

**Description:**

Medusa is technique for accelerating inference on LLMs. It essentially involves attaching multiple residual blocks (heads) onto the last hidden layer of the language model, followed by fine-tuning.

We implemented [Medusa Inference Acceleration](https://arxiv.org/abs/2401.10774) on small models and profiled various stages in the Medusa inference process to see if this inference acceleration method could work in such cases. We experimented with the [Maykeye/TinyLLama-v0](https://huggingface.co/Maykeye/TinyLLama-v0) (4.62M params) from HuggingFace. This model was trained on the [roneneldan/TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), which has a small vocabulary specifically designed for training smaller language models for story generation.

The code in this repository for Medusa profiling / training / implementation is based of that of the [original authors](https://github.com/FasterDecoding/Medusa).

---

**Repository Outline:**

```
- data_preparation
   -  generate_tiny_stories.py : generates json file of data to train on
- axolotl_training_config
   - tiny_stories_llama_1000.yml : contains a configuration (.yml) file to train using axolotl
- medusa_model
   - cli.py : runs client that asks for user input and outputs text generation w/ medusa models (+ profiling)
   - kv_cache.py : implementation of key value cache optimmization
   - medusa_model_profiled.py : base medusa model with profiling
   - modeling_llama_kv.py : base Llama model with Medusa attention mask addition and key value cache optimization
   - utils.py : utility functions (for Medusa inference)
- README.md
```

---

### Generate dataset used for medusa head training from TinyStories dataset on HuggingFace:
```
cd data_preparation
python generate_tinystories.py
```


### Training
We use a the modified [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) library to manage the training process. Use this [fork](https://github.com/ctlllll/axolotl) for the training code.  Add the training config to [`examples/medusa`](https://github.com/ctlllll/axolotl/tree/main/examples/medusa).
```bash
accelerate launch -m axolotl.cli.train examples/medusa/your_config.yml
```

### Running Inference Client for Medusa (with profiling of inference stages):
```
python -m cli --model [MODEL_NAME] [--max_steps (NUMBER_OF_MEDUSA_STEPS)]
# This should open up a prompt where you can interact with the model and subsequently observe inference speeds
# e.g. python -m cli --model FasterDecoding/medusa-1.0-vicuna-7b-v1.5
```

 - Running Inference Client for base model (directly using FastChat):
```

```

---

Results:
<img width="974" alt="vicuna_results" src="https://github.com/akhilgolla24/TinyMedusa/assets/60799338/d076df06-6539-4823-aec6-5acee5cadce7">


