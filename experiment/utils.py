import torch

class AbstractEmb():
    def __init__(self, x):
      self.x = x
    
    @property
    def input_ids(self): 
      return self.x

class AbstractTokenizer():
    def __call__(self, x, *args, **kwds):
        return AbstractEmb(x)
    
    def pad(self, x, *args, **kwds):
        return AbstractEmb(x["input_ids"])
    
def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params
