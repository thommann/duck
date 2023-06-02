from transformers import OPTForCausalLM

model = OPTForCausalLM.from_pretrained("facebook/opt-30b")
weights = model.state_dict()

for key in weights:
    print(key, flush=True)
    print(weights[key].shape, flush=True)
