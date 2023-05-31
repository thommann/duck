from transformers import OPTForCausalLM

model = OPTForCausalLM.from_pretrained("facebook/opt-66b")
weights = model.state_dict()

# write the token embedding weights to a csv file using numpy
key = 'model.decoder.embed_tokens.weight'
# np.savetxt(f"{key}.csv", weights[key].numpy(), delimiter=",")
