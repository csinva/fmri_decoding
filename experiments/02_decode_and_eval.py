from collections import defaultdict
from copy import deepcopy
import logging
import os
import random
import joblib
import numpy as np
import json
import argparse

from tqdm import tqdm
import h5py
from pathlib import Path
import imodelsx.cache_save_utils

from decoding import config
from decoding.lm_wrapper import LMWrapper
from decoding.decoder import Decoder, Hypothesis
from decoding.lm_sampler import LMSampler
from decoding.encoding_model import EncodingModel
from decoding.stimulus_model import StimulusModel, get_lanczos_mat, affected_trs, LMEmbeddingExtractor
from decoding.utils_stim import predict_word_rate, predict_word_times
from decoding.utils_eval import generate_null, load_transcript, get_window_tuples_of_fixed_duration, segment_into_word_lists_based_on_timing, WER, BLEU, METEOR, BERTSCORE


def decode(args, r):
    # determine word rate model voxels based on experiment
    if args.experiment in ["imagined_speech", "perceived_movies"]:
        word_rate_voxels = "speech"
    else:
        word_rate_voxels = "auditory"

    # load responses
    # hf = h5py.File(os.path.join(config.DATA_TEST_DIR, "test_response", args.subject, args.experiment, args.task + ".hf5"), "r")
    hf = h5py.File(os.path.join(
        config.DATA_PATH_TO_DERIVATIVE_DS004510, 'preprocessed_data',
        config.map_to_uts_subject(args.subject), args.experiment, args.task + ".hf5"), "r")
    resp = np.nan_to_num(hf["data"][:])
    hf.close()
    
    # load gpt
    with open(os.path.join(config.DATA_LM_DIR, args.gpt_perceived_or_imagined, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
        decoder_vocab = json.load(f)
    # lm_wrapper = LMWrapper(path = os.path.join(config.DATA_LM_DIR, args.gpt_perceived_or_imagined, "model"), vocab = gpt_vocab) #, device = config.GPT_DEVICE)
    lm_wrapper = LMWrapper(
        model_checkpoint=args.model_checkpoint,
        gpt_perceived_or_imagined=args.gpt_perceived_or_imagined,
    )
    features = LMEmbeddingExtractor(model = lm_wrapper, layer = args.model_layer, context_words = args.num_words_context)
    lm_sampler = LMSampler(lm_wrapper, decoder_vocab, nuc_mass = args.lm_nuc_mass, nuc_ratio = config.LM_RATIO)

    # load models
    # load_location = os.path.join(config.MODEL_DIR, args.subject)
    load_model_dir = config.MODEL_DIR
    word_rate_model = np.load(os.path.join(load_model_dir, f"{args.subject}___wordrate_model_{word_rate_voxels}.npz"), allow_pickle = True)
    # encoding_model = np.load(os.path.join(load_model_dir, "encoding_model_%s.npz" % args.gpt_perceived_or_imagined))
    encoding_model = np.load(os.path.join(load_model_dir, 
        f"{args.subject}___{args.model_checkpoint.replace('/', '_')}___encoding_model_{args.gpt_perceived_or_imagined}.npz"))
    weights = encoding_model["weights"]
    noise_model = encoding_model["noise_model"]
    tr_stats = encoding_model["tr_stats"]
    word_stats = encoding_model["word_stats"]
    em = EncodingModel(resp, weights, encoding_model["voxels"], noise_model, device = config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)
    assert args.task not in encoding_model["stories"]
    
    # predict word times
    word_rate = predict_word_rate(resp, word_rate_model["weights"], word_rate_model["voxels"], word_rate_model["mean_rate"])
    if args.experiment == "perceived_speech":
        word_times, tr_times = predict_word_times(word_rate, resp, starttime = -10)
    else:
        word_times, tr_times = predict_word_times(word_rate, resp, starttime = 0)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)

    if args.frac_to_decode < 1.0:
        nwords = int(len(word_times) * args.frac_to_decode)
        word_times = word_times[:nwords]
        lanczos_mat = lanczos_mat[:nwords]

    # decode responses
    print('decoding...')
    decoder = Decoder(word_times, config.WIDTH)
    sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device = config.SM_DEVICE)

    for sample_index in tqdm(range(len(word_times))):
        # identify TRs influenced by words in the range [start_index, end_index]
        trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)

        # number of prior words within [seconds] of the currently sampled time point
        ncontext = decoder.time_window(sample_index, config.LM_TIME, floor = 5)

        # get possible extension words for each hypothesis in the decoder beam
        beam_word_logprob_pairs = lm_sampler.beam_propose(decoder.beam, ncontext)
        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_word_logprob_pairs[c]
            if len(nuc) < 1: continue
            extend_words = [hyp.words + [x] for x in nuc]
            extend_embs = list(features.extend(extend_words))
            stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
            likelihoods = em.prs(stim, trs)
            local_extensions = [Hypothesis(parent = hyp, extension = x) for x in zip(nuc, logprobs, extend_embs)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        decoder.extend(verbose = False)
        
    print('saving...')
    if args.experiment in ["perceived_movie", "perceived_multispeaker"]:
        decoder.word_times += 10
    # save_location = os.path.join(args.save_dir, args.subject, args.experiment)
    # os.makedirs(save_location, exist_ok = True)
    r['decoded_words'] = np.array(decoder.beam[0].words)
    r['decoded_word_times'] = np.array(decoder.word_times)
    # np.savez(path, words = np.array(self.beam[0].words), times = np.array(self.word_times))
    # r['']
    # decoder.save(os.path.join(save_location, args.task))
    # print('done!')
    return r


def evaluate(args, r):
    # if len(args.references) == 0:
        # args.references.append(args.task)
        
    with open(os.path.join(config.DATA_TEST_DIR, "eval_segments.json"), "r") as f:
        eval_segments = json.load(f)
                
    # load language similarity metrics
    metrics = {
        'WER': WER(use_score = True),
        'BLEU': BLEU(n = 1),
        'METEOR': METEOR(),
        'BERT': BERTSCORE(
            idf_sents = np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy")),
            rescale = False,
            score = "recall")
    }
    # if "WER" in args.metrics: metrics["WER"] = WER(use_score = True)
    # if "BLEU" in args.metrics: metrics["BLEU"] = BLEU(n = 1)
    # if "METEOR" in args.metrics: metrics["METEOR"] = METEOR()
    # if "BERT" in args.metrics: metrics["BERT"] = BERTSCORE(
    #     idf_sents = np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy")), 
    #     rescale = False, 
    #     score = "recall")

    # load prediction transcript
    # pred_path = os.path.join(config.RESULT_DIR, args.subject, args.experiment, args.task + ".npz")
    # pred_data = np.load(pred_path)
    # pred_words, pred_times = pred_data["words"], pred_data["times"]
    pred_words, pred_times = r['decoded_words'], r['decoded_word_times']

    # generate null sequences
    print('generating null sequences...')
    null_word_list = generate_null(pred_times, args.gpt_perceived_or_imagined, args.num_null, args)

    print('scoring...')
    # for reference in args.references:
    # reference = args.task # args.references[0]

    # load reference transcript
    ref_data = load_transcript(args.experiment, args.task)
    ref_words, ref_times = ref_data["words"], ref_data["times"]

    # segment prediction and reference words into windows
    start_time, end_time = eval_segments[args.task]
    window_start_stop_tuples = get_window_tuples_of_fixed_duration(start_time, end_time, duration=config.WINDOW)
    ref_windowed_word_lists = segment_into_word_lists_based_on_timing(ref_words, ref_times, window_start_stop_tuples)
    pred_windowed_word_lists = segment_into_word_lists_based_on_timing(pred_words, pred_times, window_start_stop_tuples)
    null_windowed_word_lists_list = [segment_into_word_lists_based_on_timing(null_words, pred_times, window_start_stop_tuples)
                                        for null_words in null_word_list]

    # if decoding only a fraction of the data, cut off wherever there are no predicted words in the list
    if args.frac_to_decode < 1.0:
        idxs_nonempty_pred = [i for i, x in enumerate(pred_windowed_word_lists) if len(x) > 0]
        ref_windowed_word_lists = [ref_windowed_word_lists[i] for i in idxs_nonempty_pred]
        pred_windowed_word_lists = [pred_windowed_word_lists[i] for i in idxs_nonempty_pred]
        null_windowed_word_lists_list = [[x[i] for i in idxs_nonempty_pred] for x in null_windowed_word_lists_list]
    r['num_decoded_windows'] = len(pred_windowed_word_lists)
    
    for metric_name, metric in metrics.items():

        # get null score for each window and the entire story
        print('get null score...')
        window_null_scores = np.array([metric.score(ref = ref_windowed_word_lists, pred = null_windows) 
                                        for null_windows in null_windowed_word_lists_list])
        story_null_scores = window_null_scores.mean(1)
        r[f'{metric_name}_null_window_mean_score'] = window_null_scores.mean()

        # get raw score and normalized score for each window
        print('get raw windowed score...')
        r[f'{metric_name}_window_scores'] = metric.score(ref = ref_windowed_word_lists, pred = pred_windowed_word_lists)
        r[f'{metric_name}_window_zscores'] = (r[f'{metric_name}_window_scores'] - window_null_scores.mean(0)) / window_null_scores.std(0)
        r[f'{metric_name}_window_mean_score'] = r[f'{metric_name}_window_scores'].mean()

        # get raw score and normalized score for the entire story
        print('get raw story score...')
        r[f'{metric_name}_story_scores'] = metric.score(ref = ref_windowed_word_lists, pred = pred_windowed_word_lists)
        r[f'{metric_name}_story_zscores'] = (r[f'{metric_name}_story_scores'] - story_null_scores.mean()) / story_null_scores.std()
        r[f'{metric_name}_story_mean_score'] = r[f'{metric_name}_story_scores'].mean()
    return r

# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """
    # shared args
    parser.add_argument("--subject", type = str, default = 'S3', choices=['S1', 'S2', 'S3'])
    parser.add_argument("--experiment", type = str, default='perceived_speech')
    parser.add_argument("--task", type = str, default='wheretheressmoke')
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--save_dir", type = str, default = os.path.join(config.RESULT_DIR, 'decoding', 'test'))
    parser.add_argument("--frac_to_decode", type = float, default = 1.0, help = "fraction of words to decode")
    parser.add_argument("--use_test_setup", type = int, default = 1, help = "whether to use test setup (speeds things up)", choices = [0, 1])
    parser.add_argument("--gpt_perceived_or_imagined", type = str, default = "perceived", choices=['imagined', 'perceived'])
    

    # decoder args
    parser.add_argument("--model_checkpoint", type = str, default = 'gpt', help = "which model checkpoint to use")
    parser.add_argument("--model_layer", type = int, default = 9, help = "which GPT layer to use")
    parser.add_argument("--num_words_context", type = int, default = 5, help = "how many context words to use")
    parser.add_argument("--lm_nuc_mass", type = float, default = 0.9,
                        help = "nucleus sampling mass for LM decoder during beam search")

    # evaluation args
    # parser.add_argument("--metrics", nargs = "+", type = str, default = ["WER", "BLEU", "METEOR", "BERT"])
    # parser.add_argument("--references", nargs = "+", type = str, default = [])
    parser.add_argument("--num_null", type = int, default = 10)
    return parser

def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)"""
    parser.add_argument(
        "--use_cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to check for cache",
    )
    return parser

if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    if args.use_test_setup:
        args.frac_to_decode = 0.01
        args.use_cache = 0

    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = imodelsx.cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )

    if args.use_cache and already_cached:
        logging.info(f"cached version exists! Successfully skipping :)\n\n\n")
        exit(0)
    for k in sorted(vars(args)):
        logger.info("\t" + k + " " + str(vars(args)[k]))
    logging.info(f"\n\n\tsaving to " + save_dir_unique + "\n")

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r["git_commit_id"] = imodelsx.cache_save_utils.get_git_commit_id()
    r["save_dir_unique"] = save_dir_unique
    
    r = decode(args, r)
    r = evaluate(args, r)

    # save results
    os.makedirs(save_dir_unique, exist_ok=True)
    joblib.dump(
        r, os.path.join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    logging.info(f"Succesfully completed with mean window bert-score of {r['BERT_window_mean_score']}\n\n")
    