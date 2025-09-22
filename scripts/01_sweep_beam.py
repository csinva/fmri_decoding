from imodelsx import submit_utils
from os.path import dirname, join, expanduser
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'save_dir': [join(repo_dir, 'results', 'sep22_sweep_beam')],
    'use_test_setup': [0],
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys
params_coupled_dict = {
}

# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
# specify amlt resources
amlt_kwargs = {
    'amlt_file': join(repo_dir, 'scripts', 'launch.yaml'),
    #'sku': '10C3', # 4 cpus
    
    'sku': '40G1-A100',
    # 'sku': 'G2-A100',
    'target___name': 'palisades26',
    # 'target___name': 'msrresrchvc',
    # 'target___name': 'msroctovc',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),

    'env': {
        'HUGGINGFACE_TOKEN': f'{open(expanduser("~/.HF_TOKEN"), "r").read().strip()}',
    },
}
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '02_decode_and_eval.py'),
    # actually_run=False,

    # by default loops over jobs in serial
    # n_cpus=8,  # Uncomment to parallelize over cpus
    gpu_ids=[0, 1, 2, 3],  # Uncomment to run individual jobs over each gpu
    # gpu_ids=[0],  # Uncomment to run all jobs on a single gpu
    # gpu_ids=[[0, 1], [2, 3]], # Uncomment to run jobs on [0, 1] and [2, 3] gpus respectively
    # gpu_ids=[[0, 1, 2, 3]],  # Run job on all gpus together

    # uncomment this to run jobs on cluster (need to run this script from the scripts directory)
    # amlt_kwargs=amlt_kwargs,
    cmd_python='.venv/bin/python',
)
    