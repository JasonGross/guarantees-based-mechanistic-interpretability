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
def bytes_to_str(size: float) -> str:
    if size < 1024:
        return f"{size:.2f} B"
    size /= 1024
    if size < 1024:
        return f"{size:.4f} KB"
    size /= 1024
    if size < 1024:
        return f"{size:.8f} MB"
    size /= 1024
    return f"{size:.16f} GB"


# %%
artifact_groups = {}
with (
    # tqdm(desc="Space cleaned up", position=0) as pbar_space,
    tqdm(projects, desc="Projects", position=0) as tq,
):
    tq_position = 0
    runs_position = tq_position + 1
    for project in tq:
        total_cleaned = 0
        tq.set_postfix(dict(project=project, total_cleaned=bytes_to_str(total_cleaned)))
        runs = list(api.runs(path=f"{entity}/{project}"))
        with tqdm(runs, desc="Runs", position=runs_position) as pbar_run:
            position = runs_position + 1
            for r in pbar_run:
                pbar_run.set_postfix(dict(id=r.id))
                wandb_id = f"{entity}/{project}/{r.id}"
                artifact_groups[wandb_id] = {}
                for artifact in tqdm(
                    r.logged_artifacts(),
                    desc="Artifacts",
                    position=position,
                    leave=False,
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
                    position=position,
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
                            # pbar_space.update(artifact.size / 1024 / 1024)
                            total_cleaned += artifact.size
                            tq.set_postfix(
                                dict(
                                    project=project,
                                    total_cleaned=bytes_to_str(total_cleaned),
                                )
                            )
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
