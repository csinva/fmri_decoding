import json
import os
from typing import List
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import softmax
import imodelsx.util

from decoding import config

class LMWrapper():    
    """wrapper for https://huggingface.co/openai-gpt
    """
    def __init__(self, model_checkpoint, gpt_perceived_or_imagined='perceived', device = 'cuda'): 
        self.device = device
        self.model_checkpoint = model_checkpoint
        if model_checkpoint == 'gpt':
            path = os.path.join(config.DATA_LM_DIR, gpt_perceived_or_imagined, "model")
            self.model = AutoModelForCausalLM.from_pretrained(path).eval().to(self.device)
            with open(os.path.join(config.DATA_LM_DIR, gpt_perceived_or_imagined, "vocab.json"), "r") as f:
                self.vocab = json.load(f)
            self.word2id = {w: i for i, w in enumerate(self.vocab)}
            self.UNK_ID = self.word2id['<unk>']
        elif model_checkpoint == 'meta-llama/Meta-Llama-3-8B':
            # self.model = AutoModelForCausalLM.from_pretrained(
                # "meta-llama/Meta-Llama-3-8B", dtype=torch.float16, device_map="auto").eval()
            # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
            self.llm_embs = imodelsx.llm.LLMEmbs(checkpoint=model_checkpoint)
            # self.word2id = self.tokenizer.get_vocab()
            # self.vocab = self.tokenizer.get_vocab()
            # id_to_token = sorted(self.word2id.items(), key=lambda x: x[1])
            # self.vocab = [token for token, idx in id_to_token]
            # self.UNK_ID = 0
            

    def _encode_words_to_ids(self, words: List[str]):
        """map from words to ids
        """
        if self.model_checkpoint == 'gpt':
            return [self.word2id[x] if x in self.word2id else self.UNK_ID for x in words]
        else:
            raise NotImplementedError
            # return self.tokenizer(' '.join(words))['input_ids']

    def _encode_texts_to_tensor(self, contexts_list: List[List[str]]):
        """get word ids for each context
        """
        context_array = np.array([self._encode_words_to_ids(words) for words in contexts_list])
        ids = torch.tensor(context_array).long()
        mask = torch.ones(ids.shape).int()
        return ids, mask

    @torch.no_grad()
    def get_hidden(self, contexts_list: List[List[str]], layer):
        """get hidden layer representations

        Returns
        -------
        embs : np.ndarray
            array of shape (num_contexts, hidden_size)
        """
        if self.model_checkpoint == 'gpt':
            ids, mask = self._encode_texts_to_tensor(contexts_list)
            outputs = self.model(
                input_ids = ids.to(self.device), 
                attention_mask = mask.to(self.device),
                output_hidden_states = True
            )
            hidden_states = outputs.hidden_states[layer].detach().cpu().numpy()
            # hidden_states is now (num_contexts, seq_len, hidden_size)
            return hidden_states[:, -1, :]  # return the last token's embedding for each context
        else:
            embs = self.llm_embs([' '.join(x) for x in contexts_list], layer_idx=layer, batch_size=256)
            return embs
            

    @torch.no_grad()
    def get_probs(self, contexts_list: List[str]): #, ids, mask):
        """get next word probability distributions
        """
        if self.model_checkpoint == 'gpt':
            ids, mask = self._encode_texts_to_tensor(contexts_list)
            outputs = self.model(input_ids = ids.to(self.device), attention_mask = mask.to(self.device))
            probs = softmax(outputs.logits, dim = 2).detach().cpu().numpy()
            return probs.squeeze()


def test_lm_wrapper(lm_wrapper):
    test_words = ['the', 'cat', 'sat', 'on', 'the', 'mat']

    print('get_hidden')
    hidden = lm_wrapper.get_hidden([test_words], layer=9)
    print('hidden shape', hidden.shape)

    print('get_probs')
    probs = lm_wrapper.get_probs([test_words])
    print('probs shape', probs.shape, probs, np.sum(probs), np.sum(probs[0, -1]))

if __name__ == '__main__':
    print('lets test the lm wrapper...')
    lm_wrapper = LMWrapper(model_checkpoint='gpt', gpt_perceived_or_imagined='perceived', device = 'cuda')
    test_lm_wrapper(lm_wrapper)

    print('now lets test llama...')
    lm_wrapper = LMWrapper(model_checkpoint='meta-llama/Meta-Llama-3-8B', gpt_perceived_or_imagined='perceived', device = 'cuda')
    test_lm_wrapper(lm_wrapper)