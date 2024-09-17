# %%
import wandb
from tqdm.auto import tqdm

# %%
# Initialize your wandb API client
entity = "gbmi"
projects = [
    "ModularAdd-59-10000-epochs-attention-rate-1-no-eos-nondeterministic",
    "ModularAdd-59-10000-epochs-attention-rate-1-nondeterministic",
    "ModularAdd-12-3000-epochs-nondeterministic",
    "ModularAdd-12-3000-epochs-no-eos-nondeterministic",
    "ModularAdd-7-3000-epochs-nondeterministic",
    "ModularAdd-7-3000-epochs-no-eos-nondeterministic",
    "ModularAdd-12-3000-epochs-attention-rate-1-no-eos-nondeterministic",
    "ModularAdd-12-3000-epochs-attention-rate-1-nondeterministic",
    "ModularAdd-7-3000-epochs-attention-rate-1-no-eos-nondeterministic",
    "ModularAdd-7-3000-epochs-attention-rate-1-nondeterministic",
]
api = wandb.Api(overrides=dict(entity=entity))
# %%
artifact_groups = {}
with (
    tqdm(desc="Space cleaned up", position=0) as pbar_space,
    tqdm(projects, desc="Projects", position=1) as tq,
):
    for project in tq:
        tq.set_postfix(dict(project=project))
        runs = list(api.runs(path=f"{entity}/{project}"))
        with tqdm(runs, desc="Runs", position=2) as pbar_run:
            for r in pbar_run:
                pbar_run.set_postfix(dict(id=r.id))
                wandb_id = f"{entity}/{project}/{r.id}"
                artifact_groups[wandb_id] = {}
                for artifact in tqdm(
                    r.logged_artifacts(), desc="Artifacts", position=3, leave=False
                ):
                    name = artifact.name.split(":")[0]
                    if name not in artifact_groups[wandb_id]:
                        artifact_groups[wandb_id][name] = []
                    artifact_groups[wandb_id][name].append(artifact)
                with tqdm(
                    total=sum(
                        len(group) - 1 for group in artifact_groups[wandb_id].values()
                    ),
                    desc="Artifacts",
                    position=3,
                    leave=False,
                ) as pbar:
                    for name, group in artifact_groups[wandb_id].items():
                        # Sort versions by creation time, newest first
                        sorted_versions = sorted(
                            group, key=lambda x: x.created_at, reverse=True
                        )
                        # Keep the first (latest) version, delete the rest
                        for artifact in sorted_versions[1:]:
                            pbar.set_postfix(
                                dict(
                                    keeping=sorted_versions[0].name,
                                    deleting=artifact.name,
                                    size=artifact.size,
                                )
                            )
                            pbar_space.update(artifact.size / 1024 / 1024)
                            artifact.delete()
                            pbar.update(1)


# artifacts = {r.id: list(r.logged_artifacts()) for r in tqdm(list(api.runs(path=f"{entity}/{project}")))}
# # %%
# artifact_groups = {}
# for rid, run_artifacts in artifacts.items():
#     wandb_id = f"{entity}/{project}/{rid}"
#     artifact_groups[wandb_id] = {}
#     for artifact in run_artifacts:
#         name = artifact.name.split(":")[0]
#         if name not in artifact_groups[wandb_id]:
#             artifact_groups[wandb_id][name] = []
#         artifact_groups[wandb_id][name].append(artifact)
# %%
# For each group, keep only the latest version
# for name, group in artifact_groups[rid].items():
#     # Sort versions by creation time, newest first
#     sorted_versions = sorted(group, key=lambda x: x.created_at, reverse=True)
#     # Keep the first (latest) version, delete the rest
#     print(f"Keeping {sorted_versions[0].name}")
#     for artifact in sorted_versions[1:]:
#         print(f"Deleting {artifact.name}")
#         artifact.delete()
# %%
# Delete all but the latest version
# for version in artifact_versions:
#     if version.version != latest_version.version:
#         print(f"Deleting version {version.version} of artifact {artifact_name}")
#         version.delete()

# print(f"Kept only the latest version: {latest_version.version}")
