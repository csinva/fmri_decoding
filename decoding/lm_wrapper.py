import json
import os
from typing import List
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import softmax

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
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3-8B", dtype=torch.float16, device_map="auto").eval()
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
            self.word2id = self.tokenizer.get_vocab()
            # self.vocab = self.tokenizer.get_vocab()
            id_to_token = sorted(self.word2id.items(), key=lambda x: x[1])
            self.vocab = [token for token, idx in id_to_token]
            self.UNK_ID = 0
            

    def encode(self, words):
        """map from words to ids
        """
        if self.model_checkpoint == 'gpt':
            return [self.word2id[x] if x in self.word2id else self.UNK_ID for x in words]
        else:
            # return self.tokenizer.convert_tokens_to_ids(words)
            return self.tokenizer(' '.join(words))['input_ids']
        
    def encode_and_stack_running_segments_into_matrix(self, words, context_words):
        """get word ids for each phrase in a stimulus story
        """
        num_context_words = context_words + 1
        story_ids = self.encode(words)
        story_array = np.full(shape=(len(story_ids), num_context_words), fill_value=self.UNK_ID)
        for i in range(len(story_array)):
            segment = story_ids[i: i+num_context_words]
            story_array[i, :len(segment)] = segment
        return torch.tensor(story_array).long()

    def encode_texts_to_tensor(self, contexts_list: List[str]):
        """get word ids for each context
        """
        context_array = np.array([self.encode(words) for words in contexts_list])
        return torch.tensor(context_array).long()

    def get_hidden(self, ids, layer):
        """get hidden layer representations
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(
                input_ids = ids.to(self.device), 
                attention_mask = mask.to(self.device),
                output_hidden_states = True
            )
        return outputs.hidden_states[layer].detach().cpu().numpy()

    def get_probs(self, ids):
        """get next word probability distributions
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), attention_mask = mask.to(self.device))
        probs = softmax(outputs.logits, dim = 2).detach().cpu().numpy()
        return probs


def test_lm_wrapper(lm_wrapper):
    test_words = ['the', 'cat', 'sat', 'on', 'the', 'mat']
    print('encode', lm_wrapper.encode(test_words))

    print('get_voc', lm_wrapper.vocab[:10], len(lm_wrapper.vocab))

    print('encode_and_stack_running_segments_into_matrix')
    mat = lm_wrapper.encode_and_stack_running_segments_into_matrix(test_words, context_words=3)
    print('mat shape', mat.shape, mat)

    print('get_hidden')
    hidden = lm_wrapper.get_hidden(lm_wrapper.encode_texts_to_tensor([test_words]), layer = 9)
    print('hidden shape', hidden.shape)

    print('get_probs')
    probs = lm_wrapper.get_probs(lm_wrapper.encode_texts_to_tensor([test_words[:-1]]))
    print('probs shape', probs.shape, probs, np.sum(probs), np.sum(probs[0, -1]))

if __name__ == '__main__':
    print('lets test the lm wrapper...')
    lm_wrapper = LMWrapper(model_checkpoint='gpt', gpt_perceived_or_imagined='perceived', device = 'cuda')
    test_lm_wrapper(lm_wrapper)

    print('now lets test llama...')
    lm_wrapper = LMWrapper(model_checkpoint='meta-llama/Meta-Llama-3-8B', gpt_perceived_or_imagined='perceived', device = 'cuda')
    test_lm_wrapper(lm_wrapper)