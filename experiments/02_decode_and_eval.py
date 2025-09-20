import os
import numpy as np
import json
import argparse

from tqdm import tqdm
import h5py
from pathlib import Path

from decoding import config
from decoding.GPT import GPT
from decoding.Decoder import Decoder, Hypothesis
from decoding.LanguageModel import LanguageModel
from decoding.EncodingModel import EncodingModel
from decoding.StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures
from decoding.utils_stim import predict_word_rate, predict_word_times
from decoding.utils_eval import generate_null, load_transcript, windows, segment_data, WER, BLEU, METEOR, BERTSCORE


def decode(args):
    # determine GPT checkpoint based on experiment
    if args.experiment in ["imagined_speech"]:
        gpt_checkpoint = "imagined"
    else:
        gpt_checkpoint = "perceived"

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
    with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
        decoder_vocab = json.load(f)
    gpt = GPT(path = os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "model"), vocab = gpt_vocab, device = config.GPT_DEVICE)
    features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass = config.LM_MASS, nuc_ratio = config.LM_RATIO)

    # load models
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels), allow_pickle = True)
    encoding_model = np.load(os.path.join(load_location, "encoding_model_%s.npz" % gpt_checkpoint))
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
        beam_word_logprob_pairs = lm.beam_propose(decoder.beam, ncontext)
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
    save_location = os.path.join(args.save_dir, args.subject, args.experiment)
    os.makedirs(save_location, exist_ok = True)
    decoder.save(os.path.join(save_location, args.task))
    print('done!')


def evaluate(args):
    if len(args.references) == 0:
        args.references.append(args.task)
        
    with open(os.path.join(config.DATA_TEST_DIR, "eval_segments.json"), "r") as f:
        eval_segments = json.load(f)
                
    # load language similarity metrics
    metrics = {}
    if "WER" in args.metrics: metrics["WER"] = WER(use_score = True)
    if "BLEU" in args.metrics: metrics["BLEU"] = BLEU(n = 1)
    if "METEOR" in args.metrics: metrics["METEOR"] = METEOR()
    if "BERT" in args.metrics: metrics["BERT"] = BERTSCORE(
        idf_sents = np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy")), 
        rescale = False, 
        score = "recall")

    # load prediction transcript
    pred_path = os.path.join(config.RESULT_DIR, args.subject, args.experiment, args.task + ".npz")
    pred_data = np.load(pred_path)
    pred_words, pred_times = pred_data["words"], pred_data["times"]

    # generate null sequences
    print('generating null sequences...')
    if args.experiment in ["imagined_speech"]:
        gpt_checkpoint = "imagined"
    else:
        gpt_checkpoint = "perceived"
    null_word_list = generate_null(pred_times, gpt_checkpoint, args.null)
        
    print('scoring...')
    window_scores, window_zscores = {}, {}
    story_scores, story_zscores = {}, {}
    for reference in args.references:

        # load reference transcript
        ref_data = load_transcript(args.experiment, reference)
        ref_words, ref_times = ref_data["words"], ref_data["times"]

        # segment prediction and reference words into windows
        window_cutoffs = windows(*eval_segments[args.task], config.WINDOW)
        ref_windows = segment_data(ref_words, ref_times, window_cutoffs)
        pred_windows = segment_data(pred_words, pred_times, window_cutoffs)
        null_window_list = [segment_data(null_words, pred_times, window_cutoffs) for null_words in null_word_list]
        
        for mname, metric in metrics.items():

            # get null score for each window and the entire story
            print('get null score...')
            window_null_scores = np.array([metric.score(ref = ref_windows, pred = null_windows) 
                                           for null_windows in null_window_list])
            story_null_scores = window_null_scores.mean(1)

            # get raw score and normalized score for each window
            print('get raw windowed score...')
            window_scores[(reference, mname)] = metric.score(ref = ref_windows, pred = pred_windows)
            window_zscores[(reference, mname)] = (window_scores[(reference, mname)] 
                                                  - window_null_scores.mean(0)) / window_null_scores.std(0)

            # get raw score and normalized score for the entire story
            print('get raw story score...')
            story_scores[(reference, mname)] = metric.score(ref = ref_windows, pred = pred_windows)
            story_zscores[(reference, mname)] = (story_scores[(reference, mname)].mean()
                                                 - story_null_scores.mean()) / story_null_scores.std()
    
    print('saving...')
    save_dir = os.path.join(args.save_dir, args.subject, args.experiment, args.task, 'scores')
    os.makedirs(save_dir, exist_ok = True)
    np.savez(save_dir, 
             window_scores = window_scores, window_zscores = window_zscores, 
             story_scores = story_scores, story_zscores = story_zscores)
    print('done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # shared args
    parser.add_argument("--subject", type = str, default = 'S3', choices=['S1', 'S2', 'S3'])
    parser.add_argument("--experiment", type = str, default='perceived_speech')
    parser.add_argument("--task", type = str, default='wheretheressmoke')
    parser.add_argument("--save_dir", type = str, default = config.RESULT_DIR)

    # evaluation args
    parser.add_argument("--metrics", nargs = "+", type = str, default = ["WER", "BLEU", "METEOR", "BERT"])
    parser.add_argument("--references", nargs = "+", type = str, default = [])
    parser.add_argument("--null", type = int, default = 10)
    args = parser.parse_args()
    
    decode(args)
    evaluate(args)
    