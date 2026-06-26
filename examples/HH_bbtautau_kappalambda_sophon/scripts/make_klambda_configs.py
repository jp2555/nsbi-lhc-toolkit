"""Generate the EveNet kappa_lambda data-efficiency sweep configs + shifter run scripts.

Adapted from Exotic-Higgs-Study/Make_script.py, stripped to a classification-only kappa_lambda
task and extended with a `seeds` loop (no mass / assignment / segmentation / spanet / noise).

For each (pretrain in {finetune, frozen, scratch}) x dataset_size x seed it:
  * deep-copies configs/train_klambda.yaml,
  * sets the freeze (options.default), pretrained weights (pretrain_model_load_path),
    data fraction (Dataset.dataset_limit), seed, paths, and run/checkpoint tags,
  * (optionally) emits a matching predict config if workflow.predict_yaml is set,
  * writes <farm>/<tag>.yaml and a train-evenet.sh of `shifter ... python3 scripts/train.py` lines.

Run on Perlmutter (outside the image is fine — this only writes YAML/sh):
  python scripts/make_klambda_configs.py configs/workflow_klambda.yaml \
         --farm config_farm --store_dir <STORE> --ray_dir <RAY_TMP>
then:  bash config_farm/train-evenet.sh
"""
import argparse
import os
from copy import deepcopy

import yaml


def _abs(p):
    return os.path.abspath(p) if p else p


def prepare(args):
    with open(args.config_workflow) as f:
        wf = yaml.safe_load(f)

    farm = _abs(args.farm)
    os.makedirs(farm, exist_ok=True)
    config_dir = os.path.dirname(_abs(args.config_workflow))

    with open(os.path.join(config_dir, wf["train_yaml"])) as f:
        train_tmpl = yaml.safe_load(f)
    predict_tmpl = None
    if wf.get("predict_yaml"):
        with open(_abs(wf["predict_yaml"])) as f:
            predict_tmpl = yaml.safe_load(f)

    working_dir = _abs(wf["working_dir"])
    image = wf["image"]
    store = args.store_dir or wf["store_dir"]
    pretrain_choice = wf["pretrain_choice"]
    sizes = wf["dataset_size_choice"]
    seeds = wf.get("seeds", [0])

    train_lines, predict_lines = [], []

    for pretrain, spec in pretrain_choice.items():
        for size in sizes:
            for seed in seeds:
                tag = f"evenet-klambda-{pretrain}-size{size}-seed{seed}"
                cfg = deepcopy(train_tmpl)

                cfg["event_info"]["default"] = _abs(os.path.join(config_dir, wf["event_info"]))
                cfg["network"]["default"] = _abs(wf["network"])
                cfg["resonance"]["default"] = _abs(wf["resonance"])
                cfg["options"]["default"] = _abs(spec["option"])          # freeze lives here

                cfg["options"]["Dataset"]["dataset_limit"] = size
                cfg["options"]["Dataset"]["normalization_file"] = os.path.join(
                    store, "evenet-train", "normalization.pt")
                cfg["platform"]["data_parquet_dir"] = wf.get("data_parquet_dir",
                                                             os.path.join(store, "evenet-train"))

                cfg["options"]["Training"]["pretrain_model_load_path"] = spec["path"]
                cfg["options"]["Training"]["seed"] = seed
                cfg["options"]["Training"]["model_checkpoint_save_path"] = os.path.join(
                    store, "checkpoints", tag)
                if size < 0.1:                                            # more epochs for tiny data
                    cfg["options"]["Training"]["epochs"] = 100
                cfg["logger"]["wandb"]["run_name"] = tag

                cfg_path = os.path.join(farm, f"{tag}.yaml")
                with open(cfg_path, "w") as fout:
                    yaml.safe_dump(cfg, fout, sort_keys=False)

                load_all = "--load_all" if size < 0.2 else ""
                train_lines.append(
                    f"cd {working_dir}; shifter --image={image} "
                    f"python3 scripts/train.py {cfg_path} --ray_dir {args.ray_dir} {load_all}\n")

                if predict_tmpl is not None:
                    pc = deepcopy(predict_tmpl)
                    pc["network"]["default"] = _abs(wf["network"])
                    pc["event_info"]["default"] = _abs(os.path.join(config_dir, wf["event_info"]))
                    pc["resonance"]["default"] = _abs(wf["resonance"])
                    pc["platform"]["data_parquet_dir"] = os.path.join(store, "evenet-test")
                    pc["options"]["prediction"]["output_dir"] = os.path.join(store, "predictions", tag)
                    pc["options"]["Training"]["model_checkpoint_load_path"] = os.path.join(
                        store, "checkpoints", tag)
                    pc["options"]["Dataset"]["normalization_file"] = os.path.join(
                        store, "evenet-train", "normalization.pt")
                    pc_path = os.path.join(farm, f"{tag}_predict.yaml")
                    with open(pc_path, "w") as fout:
                        yaml.safe_dump(pc, fout, sort_keys=False)
                    predict_lines.append(
                        f"cd {working_dir}; shifter --image={image} "
                        f"python3 scripts/predict.py {pc_path}\n")

    with open(os.path.join(farm, "train-evenet.sh"), "w") as f:
        f.writelines(train_lines)
    if predict_lines:
        with open(os.path.join(farm, "predict-evenet.sh"), "w") as f:
            f.writelines(predict_lines)

    n = len(pretrain_choice) * len(sizes) * len(seeds)
    print(f"wrote {n} train configs to {farm} "
          f"({len(pretrain_choice)} configs x {len(sizes)} sizes x {len(seeds)} seeds)"
          + (f" + {len(predict_lines)} predict configs" if predict_lines else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config_workflow")
    ap.add_argument("--farm", default="config_farm")
    ap.add_argument("--store_dir", default=None, help="override workflow.store_dir")
    ap.add_argument("--ray_dir", default="ray_tmp")
    prepare(ap.parse_args())


if __name__ == "__main__":
    main()
