import os
from pathlib import Path
import einops
import torch
import tqdm
import webdataset as wds
from utilities import utils as utils
import ray


@ray.remote
def wds_write_ith_shard(configs, dataset, mode, i, n):
    shard_path = Path(os.path.join(configs["webdataset_root_path"], "webdataset", configs["dataset"], mode))
    shard_path.mkdir(parents=True, exist_ok=True)

    pattern = os.path.join(configs["dataset_root_path"], "webdataset", configs["dataset"], mode, f"sample-{mode}-{i}-%06d.tar")

    with wds.ShardWriter(pattern, maxcount=configs["max_samples_per_shard"]) as sink:
        for index in tqdm.tqdm(range(i, len(dataset), n)):
            batch = dataset[index]
            if isinstance(batch, dict):
                image = batch["image"]
            else:
                (image, labels) = batch
            
            if isinstance(batch, dict):
                labels_dict = {}
                for key in batch:
                    if key != "image":
                        labels_dict[key] = batch[key]
                sink.write({"__key__": "sample%06d" % index, "image.pth": image, "labels.pth": labels_dict})
            else:
                sink.write({"__key__": "sample%06d" % index, "image.pth": image, "labels.pth": labels})


def wds_write_parallel(configs):
    ray.init()
    n = configs["webdataset_write_processes"]
    for mode in ["train","val","test"]:
        dataset = utils.get_dataset(configs, configs['dataset'],mode=mode,download=False,dataset_root=configs['dataset_root_path'],imagenet_norm=False)
        print("=" * 40)
        print("Creating shards for dataset: ", configs["dataset"])
        print("Mode: ", mode, " Size: ", len(dataset))
        print("=" * 40)

        ray.get([wds_write_ith_shard.remote(configs, dataset, mode, i, n) for i in range(n)])