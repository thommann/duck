import numpy as np
from transformers import OPTForCausalLM

model = OPTForCausalLM.from_pretrained("facebook/ML-30b")
weights = model.state_dict()

max_i = 0
i = 0
for key in weights:
    i += 1
    print(key, flush=True)
    value = weights[key]
    shape = value.shape
    print(shape, flush=True)
    np.savetxt(f"data/ML-30b/{key}{shape[0]}x{shape[1]}", np.array(weights[key]))
    if i > max_i:
        break
