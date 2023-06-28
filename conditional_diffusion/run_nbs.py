# contents of `papermill_runner.py`
import papermill as pm

base_exp_dir = "/media/master/wext/msc_studies/second_semester/research_project/project/rsl_depth_completion/rsl_depth_completion/conditional_diffusion/models/cond_scale"
eval_batch_to_exp_dir = {
    "eval_batch_rand_sdm.pt": "008_cfg.exp_targets=['cond_scale']",
    "eval_batch_rand_rgb.pt": "007_cfg.exp_targets=['cond_scale']",
    "eval_batch_rand_rgb_and_sdm.pt": "006_cfg.exp_targets=['cond_scale']",
    "eval_batch.pt": "005_cfg.exp_targets=['cond_scale']",
}

parameters = [
    {"eval_batch": eval_batch, "exp_dir": f"{base_exp_dir}/{exp_dir}/model-last.pt"}
    for eval_batch, exp_dir in eval_batch_to_exp_dir.items()
]

from multiprocessing import Pool

with Pool(len(parameters)) as p:
    kwargs = {"kernel_name": "ssdc", "putput_path": f"fine-tuning-{dataset}.ipynb"}
    p.starmap(pm.execute_notebook, parameters, **kwargs)
    # p.map(pm.execute_notebook, parameters)


for params in parameters:
    pm.execute_notebook(
        "fine-tuning.ipynb",
        f"fine-tuning-{params['dataset']}.ipynb",
        kernel_name="ssdc",
        parameters=params,
    )
