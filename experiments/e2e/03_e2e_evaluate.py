from decoding.e2e.config import REPO_DIR
from decoding.e2e.utils_eval import WER, BLEU, METEOR 
import nltk
import json
import copy
import re
import numpy as np
import argparse
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel, LlamaForCausalLM, LlamaTokenizer
import ast

# Ensure NLTK punkt tokenizer is available (quiet download if missing)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        # If download fails (no internet), proceed and let the calling code surface the error
        pass

def segment(result, chunk_size=10,checkpoint_path=''):
    if 'huth' in checkpoint_path:
        result['content_pred'] = [' '.join(result['content_pred'][i:i+chunk_size]).replace('  ',' ') for i in range(0, len(result['content_pred']), chunk_size)]
    else:
        result['content_pred'] = [' '.join(result['content_pred'][i:i+chunk_size]).replace('  ',' ') for i in range(0, len(result['content_pred']), chunk_size)]
    result['content_true'] = [' '.join(result['content_true'][i:i+chunk_size]) for i in range(0, len(result['content_true']), chunk_size)]

def split_content_pred_by_results(re):
    re['content_pred'] = []
    result = re['result'][-1]
    l = 0
    bad_i = []
    for i in range(len(re['content_pred_old'])):
        if result[l:l+len(re['content_pred_old'][i][0])] not in re['content_pred_old'][i]:
            bad_i.append(i)
            re['content_pred'].append('')
            continue
        re['content_pred'].append(' '.join(result[l:l+len(re['content_pred_old'][i][0])]))
        l += len(re['content_pred_old'][i][0])
    if len(bad_i) > 0:
        print('bad_i', bad_i)

tokenizer = None

def split_content_pred_by_results2(re, checkpoint_path):
    global tokenizer
    if tokenizer is None:
        if 'gpt2' in checkpoint_path:
            # tokenizer = AutoTokenizer.from_pretrained('/home/bingxing2/home/scx7140/.cache/huggingface/hub/models--gpt2-large/snapshots/97935fc1a406f447320c3db70fe9e9875dca2595')
            tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        elif 'llama-7b' in checkpoint_path or 'release' in checkpoint_path:
            tokenizer = AutoTokenizer.from_pretrained('/home/bingxing2/home/scx7140/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852')  
    re['content_pred'] = []
    result = re['result_ids'][-1]

    # Normalize result to a list of integer token ids so tokenizer.decode always receives integers.
    def _normalize_results_chandan(result):
        if isinstance(result, str):
            # Try JSON first, then AST literal eval as a fallback for string-encoded lists
            try:
                parsed = json.loads(result)
            except Exception:
                parsed = ast.literal_eval(result)
            result = parsed
        # If result is a list/tuple of string ids, convert to ints where possible
        if isinstance(result, (list, tuple)):
            try:
                result = [int(x) if isinstance(x, str) and x.isdigit() else x for x in result]
            except Exception:
                # Fallback: leave as-is; tokenizer.decode will raise a helpful error if unsupported
                pass
        else:
            # Single scalar -> wrap into list
            try:
                result = [int(result)]
            except Exception:
                result = [result]
        return result
    result = _normalize_results_chandan(result)


    l = 0
    # detect whether result contains string tokens (words) rather than integer ids
    is_str_tokens = len(result) > 0 and isinstance(result[0], str)
    for i in range(len(re['word_rate'])):
        if re['word_rate'][i] < 0:
            re['content_pred'].append('')
            if i + 1 < len(re['word_rate']):
                re['word_rate'][i+1] += re['word_rate'][i]
        elif re['word_rate'][i] == 0:
            re['content_pred'].append('')
        else:
            tmp_result = result[l:l+re['word_rate'][i]]
            k = 0
            if is_str_tokens:
                # work with string tokens directly
                while l+re['word_rate'][i]+k < len(result) and len(' '.join(result[l:l+re['word_rate'][i]+k]).split()) == len(' '.join(result[l:l+re['word_rate'][i]+k+1]).split()):
                    k += 1
                re['content_pred'].append(' '.join(result[l:l+re['word_rate'][i]+k]))
            else:
                # work with integer token ids and tokenizer.decode
                while l+re['word_rate'][i]+k < len(result) and len(tokenizer.decode(result[l:l+re['word_rate'][i]+k]).split()) == len(tokenizer.decode(result[l:l+re['word_rate'][i]+k+1]).split()):
                    k += 1
                re['content_pred'].append(tokenizer.decode(result[l:l+re['word_rate'][i]+k]))
            l += re['word_rate'][i]+k
            if i + 1 < len(re['word_rate']):
                re['word_rate'][i+1] -= k
        
def normalize_text(text_from_tokens):
    text_from_tokens = re.sub(r'(\w+)\.(\w+)', r'\1. \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\?(\w+)', r'\1? \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\!(\w+)', r'\1! \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\:(\w+)', r'\1: \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\;(\w+)', r'\1; \2', text_from_tokens)
    return text_from_tokens

def language_evaluate_mask_with_sig(re, metrics, dataset_name='Huth',token_based=False, checkpoint_path=None):
    re['content_pred_old'] = copy.deepcopy(re['content_pred'])
    if token_based == False:
        split_content_pred_by_results2(re, checkpoint_path)
    else:
        split_content_pred_by_results(re, )
    # preprocess
    for i in range(len(re['content_true'])):
        re['content_true'][i] = re['content_true'][i].replace('<|endoftext|>','').replace('??','').replace('⁇','').replace('</s>','').replace('<unk>','').replace('  ', ' ').replace('\n', ' ')
        re['content_pred'][i] = re['content_pred'][i].replace('<|endoftext|>','').replace('??','').replace('⁇','').replace('</s>','').replace('<unk>','').replace('  ', ' ').replace('\n', ' ').replace('<s>', '')
    
    segment(re, checkpoint_path=checkpoint_path)
    
    re['content_pred_tokens'] = []
    re['content_true_tokens'] = []
    for i in range(len(re['content_true'])):
        # 可以考虑对比直接 split和nltk.word_tokenize
        re['content_pred_tokens'].append(normalize_text(re['content_pred'][i]).split())
        if dataset_name in ['Huth']:
            re['content_pred_tokens'][-1] = [word.lower() for word in re['content_pred_tokens'][-1]]
        re['content_true_tokens'].append(normalize_text(re['content_true'][i]).split())
    
    for mname, metric in metrics.items():
        re[mname] = np.array([metric.score(ref = [re['content_true_tokens'][i]], pred = [re['content_pred_tokens'][i]]) for i in range(len(re['content_pred']))])
    return re

def load_metric(remove_stopwords=False, use_meteor=True):
    metrics = {}
    metrics["WER"] = WER(use_score = True, remove_stopwords=remove_stopwords)
    metrics["BLEU"] = BLEU(n = 1,remove_stopwords=remove_stopwords)
    if use_meteor:
        try:
            metrics["METEOR"] = METEOR(remove_stopwords=remove_stopwords)
        except Exception:
            # If METEOR cannot be initialized (missing wordnet), skip it
            print('Warning: METEOR metric unavailable (wordnet missing). Skipping METEOR.')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', default=os.path.join(REPO_DIR, 'results'), type=str, required=False)
    parser.add_argument('--dir', default='', type=str, required=True)
    parser.add_argument('--token_based', default='False', type=str, required=False)
    parser.add_argument('--no_meteor', action='store_true', help='Disable METEOR metric (avoids NLTK wordnet dependency)')
    args = parser.parse_args()
    args.token_based = args.token_based == 'True'
    if 'huth' in args.dir:
        args.token_based = True

    all_file_names = os.listdir(os.path.join(args.results_path, args.dir))
    all_file_names = [x for x in all_file_names if x.endswith('.json') and 'e2e' in x]
    
    for file_name in all_file_names:
        file_path = os.path.join(args.results_path, args.dir, file_name)
        if os.path.exists(file_path):
            result = json.load(open(file_path))
            metrics = load_metric(use_meteor=not args.no_meteor)
            language_evaluate_mask_with_sig(result, metrics, token_based = args.token_based,checkpoint_path = args.dir)
            out_meteor = ('%.3f' % np.mean(result['METEOR'])) if 'METEOR' in result else 'n/a'
            output_str = file_path + f" bleu_1: {'%.3f' % np.mean(result['BLEU'])} wer: {'%.3f' % np.mean(result['WER'])} meteor: {out_meteor}"
            print(output_str)
