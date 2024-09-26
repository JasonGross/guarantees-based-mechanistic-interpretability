# %%
import shutil
from pathlib import Path

import torch
from tqdm.auto import tqdm

base = Path(".").resolve()
artifact_base = base / "artifacts"
wandbs = list(artifact_base.glob("*/*.pth"))
model_base = base / "trained-models"
model_base.mkdir(exist_ok=True, parents=True)

with tqdm(wandbs) as pbar:
    for path in pbar:
        cache = torch.load(path, map_location="cpu")
        # if cache["run_config"]["experiment"]["p"] not in (7, 12):
        #     continue
        pbar.set_postfix(
            {
                "seed": cache["run_config"]["seed"],
                "orig_name": path.name,
                "suffix_drop": "-".join(path.name.split("-")[-6:]),
            }
        )
        seed = cache["run_config"]["seed"]
        target = (
            model_base / f"{'-'.join(path.name.split('-')[:-6])}-{seed}{path.suffix}"
        )
        if not target.exists():
            shutil.copy(path, target)
    # break
# %%
# gtar --transform='s|.*/||' --owner=0 --group=0 --numeric-owner -czf modular-add-7,12-pizza-no-eos-partial.tar.gz models/ModularAdd-{7,12}*attention-rate-1*no-eos*.pth
# gtar --transform='s|.*/||' --owner=0 --group=0 --numeric-owner -czf modular-add-7,12-pizza-partial.tar.gz $(ls models/ModularAdd-{7,12}*attention-rate-1*.pth | grep -v no-eos)
