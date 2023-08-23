import torch.nn as nn
from kogpt2_transformers import get_kogpt2_model


class DialogKoGPT2(nn.Module):
  def __init__(self):
    super(DialogKoGPT2, self).__init__()
    self.kogpt2 = get_kogpt2_model()

  def generate(self,
               input_ids,
               do_sample=True,
               max_length= 60,
               top_p=0.92, #0.92
               top_k=80, #50
               temperature= 0.8, #0.6
               no_repeat_ngram_size =None,
               #repetition_penalty=1.2,
               num_return_sequences=2, #3
               early_stopping=True, #False
               pad_token_id=-1,
               ):
    return self.kogpt2.generate(input_ids,
               do_sample=do_sample,
               max_length=max_length,
               top_p = top_p,
               top_k=top_k,
               temperature=temperature,
               no_repeat_ngram_size= no_repeat_ngram_size,
               #repetition_penalty=repetition_penalty,
               num_return_sequences=num_return_sequences,
               early_stopping = early_stopping,
               pad_token_id=pad_token_id,
              )

  def forward(self, input, labels = None):
    if labels is not None:
      outputs = self.kogpt2(input, labels=labels)
    else:
      outputs = self.kogpt2(input)

    return outputs

