import os
import numpy as np
import json
import argparse

from decoding import config
from decoding.lm_wrapper import LMWrapper
from decoding.stimulus_model import LMEmbeddingExtractor
from decoding.utils_stim import get_stim
from decoding.utils_resp import get_resp
from decoding.utils_ridge.ridge import ridge, bootstrap_ridge
import logging
np.random.seed(42)

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, default = 'S3', choices=['S1', 'S2', 'S3'])
    parser.add_argument("--gpt_perceived_or_imagined", type = str, default = "perceived", choices=['imagined', 'perceived'])
    parser.add_argument("--sessions", nargs = "+", type = int, 
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--model_checkpoint", type = str, default = 'gpt', help = "which model checkpoint to use",
                        choices=['gpt', 'meta-llama/Meta-Llama-3-8B'])
    parser.add_argument("--model_layer", type = int, default = 9, help = "which GPT layer to use")
    parser.add_argument("--num_words_context", type = int, default = 5, help = "how many context words to use")
    parser.add_argument("--save_dir", type = str, default = config.MODEL_DIR, help = "directory to save the model")
    args = parser.parse_args()

    # training stories
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f) 
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])

    # load lm
    lm_wrapper = LMWrapper(
        model_checkpoint=args.model_checkpoint,
        gpt_perceived_or_imagined=args.gpt_perceived_or_imagined,
    )
    lm_embedding_extractor = LMEmbeddingExtractor(model = lm_wrapper, layer = args.model_layer, context_words = args.num_words_context)

    # estimate encoding model
    print('load and process stimulus...')
    rstim, tr_stats, word_stats = get_stim(stories, lm_embedding_extractor)
    print('load response...')
    rresp = get_resp(args.subject, stories, stack = True)
    nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))
    print('estimate encoding model...')
    weights, alphas, bscorrs = bootstrap_ridge(rstim, rresp, use_corr = False, alphas = config.ALPHAS,
        nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks)        
    bscorrs = bscorrs.mean(2).max(0)
    vox = np.sort(np.argsort(bscorrs)[-config.VOXELS:])
    del rstim, rresp
    print('mean bscorrs', bscorrs.mean(), 'mean bscorrs of selected voxels', bscorrs[vox].mean())
    
    # estimate noise model
    print('estimate noise model...')
    stim_dict = {story : get_stim([story], lm_embedding_extractor, tr_stats = tr_stats) for story in stories}
    resp_dict = get_resp(args.subject, stories, stack = False, vox = vox)
    noise_model = np.zeros([len(vox), len(vox)])
    for hstory in stories:
        tstim, hstim = np.vstack([stim_dict[tstory] for tstory in stories if tstory != hstory]), stim_dict[hstory]
        tresp, hresp = np.vstack([resp_dict[tstory] for tstory in stories if tstory != hstory]), resp_dict[hstory]
        bs_weights = ridge(tstim, tresp, alphas[vox])
        resids = hresp - hstim.dot(bs_weights)
        bs_noise_model = resids.T.dot(resids)
        noise_model += bs_noise_model / np.diag(bs_noise_model).mean() / len(stories)
    del stim_dict, resp_dict
    
    # save
    print('save...')
    os.makedirs(args.save_dir, exist_ok = True)
    np.savez(os.path.join(args.save_dir,
                          f"{args.subject}___{args.model_checkpoint.replace('/', '_')}___encoding_model_{args.gpt_perceived_or_imagined}"), 
        weights = weights, noise_model = noise_model, alphas = alphas, voxels = vox, stories = stories,
        tr_stats = np.array(tr_stats), word_stats = np.array(word_stats), bscorrs=bscorrs)
    print(f'done! saved to {args.save_dir}')