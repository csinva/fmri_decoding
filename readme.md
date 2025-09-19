# Tang semantic decoding

This repository contains code used in the paper "Semantic reconstruction of continuous language from non-invasive brain recordings" by Jerry Tang, Amanda LeBel, Shailee Jain, and Alexander G. Huth.  

## Setup
- install dependencies with `uv sync`
- set up data
  - set paths in `decoding.tang.config.py` for where you want to download things and save results
  - then run `uv run experiments/tang/_download_data.py` to download all the data necessary in the right places. If this fails, download the data manually as follows:
    - Download [language model data](https://utexas.box.com/shared/static/7ab8qm5e3i0vfsku0ee4dc6hzgeg7nyh.zip) and extract contents into `decoding.tang.config.DATA_LM_DIR`
    - Download [training data](https://utexas.box.com/shared/static/3go1g4gcdar2cntjit2knz5jwr3mvxwe.zip) and extract contents into `decoding.tang.config.DATA_TRAIN_DIR`
      - download stimulus data for `train_stimulus/` and response data for `train_response/[SUBJECT_ID]` from [OpenNeuro ds003020](https://openneuro.org/datasets/ds003020/). This should be specified in `decoding.tang.config.DATA_PATH_TO_DERIVATIVE_DS003020`
    - Download [test data](https://utexas.box.com/shared/static/ae5u0t3sh4f46nvmrd3skniq0kk2t5uh.zip) and extract contents into `decoding.tang.config.DATA_TEST_DIR`
      - download stimulus data for `test_stimulus/[EXPERIMENT]` and response data for `test_response/[SUBJECT_ID]` from [OpenNeuro ds004510](https://openneuro.org/datasets/ds004510/). This should be specified in `decoding.tang.config.DATA_PATH_TO_DERIVATIVE_DS004510`

## Running
1. Estimate the encoding model. The encoding model predicts brain responses from contextual features of the stimulus extracted using GPT. The `--gpt` parameter determines the GPT checkpoint used. Use `--gpt imagined` when estimating models for imagined speech data, as this will extract features using a GPT checkpoint that was not trained on the imagined speech stories. Use `--gpt perceived` when estimating models for other data. The encoding model will be saved in `MODEL_DIR/[SUBJECT_ID]`. Alternatively, download [pre-fit encoding models](https://utexas.box.com/s/ri13t06iwpkyk17h8tfk0dtyva7qtqlz).

```bash
uv run python experiments/tang/00_train_EM.py --subject S3 --gpt perceived
```

2. Estimate the word rate model. The word rate model predicts word times from brain responses. Two word rate models will be saved in `MODEL_DIR/[SUBJECT_ID]`. The `word_rate_model_speech` model uses brain responses in speech regions, and should be used when decoding imagined speech and perceived movie data. The `word_rate_model_auditory` model uses brain responses in auditory cortex, and should be used when decoding perceived speech data. Alternatively, download [pre-fit word rate models](https://utexas.box.com/s/ri13t06iwpkyk17h8tfk0dtyva7qtqlz).

```bash
uv run python experiments/tang/01_train_WR.py --subject S3
```

3. Test the decoder on brain responses not used in model estimation. The decoder predictions will be saved in `RESULTS_DIR/[SUBJECT_ID]/[EXPERIMENT_NAME]`.

```bash
uv run python experiments/tang/02_run_decoder.py --subject S3
```

4. Evaluate the decoder predictions against reference transcripts. The evaluation results will be saved in `SCORE_DIR/[SUBJECT_ID]/[EXPERIMENT_NAME]`.

```bash
uv run python experiments/tang/03_evaluate_predictions.py --subject S3
```

# E2E baseline

This is the official repo for our paper [Language Generation from Brain Recordings](https://arxiv.org/abs/2311.09889). Language generation from brain recordings is a novel approach that supports direct language generation with BCIs (brain-computer interfaces) without pre-defineng or pre-generating language candidates to select from.
Code is taken from [Zenodo](https://zenodo.org/records/14838723).


### Quick Start
We have provided an example dataset to facilitate the replication of experiments. To run the example dataset, you can go into the sub-directory *language_generation/src* and use the following command:

Install / setup with ```uv sync```

```bash
# model training and evaluation (runing BrainLLM)
uv run python experiments/e2e_baseline/00_language_generation.py --task_name Pereira_example --cuda 0 --load_check_point False --model_name gpt2 --checkpoint_path example --batch_size 8 --lr 1e-4 --pos False --pretrain_lr 1e-3 --pretrain_epochs 10 --wandb none --mode all --dataset_path /home/chansingh/fmri_decoding/dataset/
# control evaluation (runing PerBrainLLM)
uv run python experiments/e2e_baseline/00_language_generation.py --task_name Pereira_example --cuda 0 --load_check_point False --model_name gpt2 --checkpoint_path example --batch_size 8 --lr 1e-4 --pos False --pretrain_lr 1e-3 --pretrain_epochs 10 --wandb none --input_method permutated --mode evaluate --output test_permutated --dataset_path /home/chansingh/fmri_decoding/dataset/
# control evaluation (runing LLM)
uv run python experiments/e2e_baseline/00_language_generation.py --task_name Pereira_example --cuda 0 --load_check_point False --model_name gpt2 --checkpoint_path example --batch_size 8 --lr 1e-4 --pos False --pretrain_lr 1e-3 --pretrain_epochs 10 --wandb none --input_method mask_input --mode evaluate --output test_nobrain --dataset_path /home/chansingh/fmri_decoding/dataset/
```

To run with the datasets utilized in our paper, please download the dataset from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/04e8cfe6c9c743c69f08/) and unzip it.
- unzip it to `/path_to_repo/released/` (or wherever you specify in the parameter *-dataset_path*).
- unpack the files via `gunzip *.gz` in each sub-directory.

```bash
# same command as above except changed --task_name and --dataset_path
uv run python experiments/e2e_baseline/00_language_generation.py --task_name Huth_1 --cuda 0 --load_check_point False --model_name gpt2 --checkpoint_path Huth_1 --batch_size 8 --lr 1e-4 --pos False --pretrain_lr 1e-3 --pretrain_epochs 10 --wandb none --mode all --pos True
``` 

To evaluate the model performance, you can refer to the code in *language_generation/src/post_hoc_evaluate.py*

## Full-text construction (e2e)

In addition to the language completion task, our method also supports generating a complete piece of text based on brain signals spanning a few minutes. The relevant code can be found in the directory of *end2end_generation/*. 
The implementation of full story construction is based on [Tang et al.](https://github.com/HuthLab/semantic-decoding) (thanks for their code).
To run this code, you also need to download some helpful files from their code, i.e., the *data_lm* directory.
- set the appropriate paths in config to point to this directory.
- data splitting is specificied via --data_spliting end2end

Here is a example that generate the human semantics while they are perceiving story of "where there's smoke":

```bash
# train BrainLLM with the spliting strategy that left out the story of "where there's smoke"
uv run python experiments/e2e_baseline/00_language_generation.py --task_name Huth_1 --cuda 0 --load_check_point False --model_name gpt2 --checkpoint_path Huth_1_gpt2_e2e --batch_size 8 --lr 1e-4 --pos False --pretrain_lr 1e-3 --pretrain_epochs 0 --wandb none --mode all --pos True --data_spliting end2end

# fit model to predict word timings
uv run python experiments/e2e_baseline/01_fit_encoding.py --task_name Huth_1 --checkpoint_path Huth_1_huth_encoding --wandb none --mode all --data_spliting end2end --model huth

# run inference for full story construction
uv run python experiments/e2e_baseline/02_e2e.py --task_name Huth_1 --cuda 0 --load_check_point False --model_name gpt2 --checkpoint_path Huth_1_gpt2_e2e --wandb none --mode evaluate --pos True --data_spliting end2end --mode end2end --use_bad_words_ids False --ncontext 10 --gcontext 10 --length_penalty 0.3 --beam_width 3 --extensions 3

# run evaluation with Huth's metrics
uv run python experiments/e2e_baseline/03_e2e_evaluate.py --dir Huth_1_gpt2_e2e
``` 

### Model Training
To train the model, you need to special the parameter *-mode* as *training* (only training) or *all* (training and evaluation).
You can specify several hyper parameters according to your requirement, the default parameters for Pereira's dataset, Huth's dataset, and Narratives dataset are provided in *language_generation/scripts/example.sh*, *language_generation/scripts/huth.sh*, and *language_generation/scripts/narratives.sh*, respectively.
The meaning of hyper parameters are listed below:

|  **Parameter**  | **Meaning**  |
|   :----   |   :----   |
| model_name | the selected LLM, choose from {gpt2,gpt2-medium,gpt2-large,gpt2-xl,llama-2} |
| method | only supported *decoding* in the released verison |
| task_name | *{dataset_name}_{participant_name}*, dataset_name selected from *{Pereira,Huth,Narratives}* |
| test_trail_ids | specify the range of test dataset, view the dict *dataset2agrs* in *language_generation/src/config.py* for default setting |
| valid_trail_ids | specify the range of validation dataset, view the dict *dataset2agrs* in *language_generation/src/config.py* for default setting |
| random_number | for cross-validation evaluation, cooperate with parameter *test_trail_ids* and *valid_trail_ids*|
| batch_size | set as 8 in our experiment |
| fmri_pca | how to do data dimensionality reduction, default is *True* |
| cuda | specify the device number |
| layer | not used in the released verison |
| num_epochs | specify the maximum number of training epochs |
| lr | learning rate, set as 1e-4 in our experiment |
| dropout | dropout rate for brain decoder |
| checkpoint_path | path of training checkpoint for saving and downloading |
| load_check_point | whether to load existing checkpoint |
| enable_grad | whether to allow the parameter in LLM updated or not |
| mode | *train*: only training and evaluate in the validation set; *evaluate*: evaluate in the test set; *all*: train and evaluate|
| additional_loss | training with additional loss, not used in the released verison |
| fake_input | training with fake input, not used in the released verison |
| add_end | not used in the released verison |
| context | whether to discard data sample without any text prompt or not |
| roi_selected | roi-based experiment, not used in the released verison |
| project_name | specify the project name for [wandb](https://wandb.ai/site) |
| noise_ratio | not used in the released verison |
| wandb | specify how to sync the experimental in [wandb](https://wandb.ai/site), selected from *{online, offline, none}* |
| generation_method | generation method for the LLM, selected from *{greeddy, beam}* |
| pos | specify whether to use position embedding in the brain decoder |
| output | specify whether to use position embedding in the brain decoder |
| data_spliting | specify how to split the dataset, selected from *{random, cross_story}*, default is *random* |
| brain_model | the based model for the brain decoder, selected from *{mlp,rnn,linear,big_mlp,multi_mlp}* |
| weight_decay | weight decay |
| l2 | weight for l2 regularized loss |
| num_layers | number of layers in the brain decoder |
| evaluate_log | whether to evaluate in the test set for model in each training epoch |
| normalized | whether to normalize the input |
| activation | activation function, selected from *{relu,sigmoid,tanh,relu6}* |
| pretrain_epochs | number of epochs in warm up step |
| pretrain_lr | learning rate in warm up step|
| data_size | maximum training data samples |
| results_path | path to save model results |
| dataset_path | path to the downloaded dataset |
| shuffle_times | permutation times for PerBrainLLM |

### Model Evaluation
To evaluate the model with different prompt input, i.e., BrainLLM, PerBrainLLM, and LLM, you can specify the parameter *-input_method* as *normal*, *permutated*, *without_brain*, respectively. To test the model performance without any text prompt, you should train and evaluate the model while setting *-input_method* as *without_text*.

After that, you can get output files for different prompt inputs. Then, you can evaluate their performance by runing the python script *language_generation/src/post_hoc_evaluatoion.py* with the path of output files specified.
Refer to *language_generation/src/post_hoc_evaluatoion.py* for example usage:
```bash
uv run python decoding/language_generation/post_hoc_evaluate.py
```

### Dataset
We test our approach on three public fMRI datasets: [Pereira's dataset](https://www.nature.com/articles/s41467-018-03068-4), [Huth's dataset](https://www.nature.com/articles/s41597-023-02437-z), and [Narratives dataset](https://www.nature.com/articles/s41597-021-01033-3). The brief introduction, ethical information, statistics, and useage details of these datasets are provied in our paper.
A preprocessed verison dataset is released in [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/04e8cfe6c9c743c69f08/), where the sub-directory of *Pereira*, *Huth*, and *Narratives* contain the preprocessed data for each participant and story in Pereira's dataset, Huth's dataset, and Narratives dataset, respectively. 

### Experimental results
This is the overall experimental results in terms of language similarity metrics. Refer to our paper for the explaination of metrics and more analyses.
| Dataset    | Model        | Bleu-1(↑) | ROUGE-1(↑) | ROUGE-L(↑) | WER(↓) |
|------------|--------------|-----------|------------|------------|--------|
| Pereira’s  | BrainLLM     | 0.3432    | 0.2987     | 0.2878     | 0.7576 |
|            | PerBrainLLM  | 0.3269    | 0.2815     | 0.2751     | 0.7783 |
|            | StdLLM          | 0.2415    | 0.2133     | 0.2096     | 0.8349 |
| Huth’s     | BrainLLM     | 0.1899    | 0.1780     | 0.1709     | 0.8946 |
|            | PerBrainLLM  | 0.1668    | 0.1536     | 0.1474     | 0.9109 |
|            | StdLLM          | 0.1500    | 0.1360     | 0.1310     | 0.9200 |
| Narratives | BrainLLM     | 0.1375    | 0.1301     | 0.1209     | 0.9239 |
|            | PerBrainLLM  | 0.1269    | 0.1211     | 0.1105     | 0.9311 |
|            | StdLLM          | 0.0953    | 0.0858     | 0.0829     | 0.9485 |


## Citation
If you find our work helpful, please consider citing us:
```
@article{ye2023language,
  title={Language Generation from Brain Recordings},
  author={Ye, Ziyi and Ai, Qingyao and Liu, Yiqun and Zhang, Min and Lioma, Christina and Ruotsalo, Tuukka},
  journal={arXiv preprint arXiv:2311.09889},
  year={2023}
}
```
