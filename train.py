"""
train.py
--------
Main training entry. Uses Hydra + PyTorch Lightning.

    python train.py                                              # default v1
    python train.py --config-name=train/v1                       # explicit
    python train.py trainer.max_epochs=10                        # override
    python train.py --config-name=ablations/no_pathway_gating    # ablation
    python train.py --config-name=ablations/no_pathway_loss
    python train.py --config-name=ablations/no_contrast
    python train.py --config-name=ablations/no_molformer
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from cytobridge.data import CytoBridgeDataModule
from cytobridge.gradient_audit import (
    append_gradient_audit,
    component_weights,
    measure_component_gradient_norms,
)
from cytobridge.losses import CytoBridgeLoss, CytoBridgeLossConfig, InfoNCEConfig
from cytobridge.model import CytoBridge, CytoBridgeConfig
from eval.model_selection import (
    PRIMARY,
    TIE_BREAKER,
    EpochSelectionTracker,
)
from eval.validation import compute_validation_metrics


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        injected = os.environ.get("CYTOBRIDGE_SOURCE_COMMIT", "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", injected):
            return injected
        return "not-a-git-repo"


def gpu_snapshot() -> list[dict[str, str]]:
    """Collect GPU metadata without initializing CUDA before worker spawn."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            rows.append(
                dict(zip(("index", "name", "memory_total_mib", "driver"), fields, strict=True))
            )
    return rows


def write_run_metadata(cfg: DictConfig, out_path: str | Path = "run_metadata.json") -> None:
    meta = {
        "seed": int(cfg.seed),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_snapshot": gpu_snapshot(),
        "git_commit": git_commit(),
        "source_tree_sha256": os.environ.get("CYTOBRIDGE_SOURCE_TREE_SHA256"),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(meta, indent=2, default=str))
    tmp_path.replace(path)


def resolve_resume_ckpt(cfg: DictConfig) -> str | None:
    resume_from = OmegaConf.select(cfg, "ckpt.resume_from", default=None)
    if resume_from in (None, "", False):
        return None
    if str(resume_from).lower() == "last":
        ckpt_dir = (
            OmegaConf.select(cfg, "ckpt.dirpath", default=None)
            or f"ckpts/{cfg.run_name}"
        )
        ckpt_path = Path(ckpt_dir) / "last.ckpt"
    else:
        ckpt_path = Path(str(resume_from))
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Requested checkpoint resume path does not exist: {ckpt_path}. "
            "Set ckpt.resume_from=null to start a fresh run, or use ckpt.resume_from=last "
            "after a previous run has written ckpts/<run_name>/last.ckpt."
        )
    return str(ckpt_path)


def _config_mapping(
    cfg: DictConfig, path: str, *, default: dict[str, object]
) -> dict:
    """Return an optional Hydra mapping as a plain dictionary."""
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        return dict(default)
    if not OmegaConf.is_config(value):
        value = OmegaConf.create(value)
    resolved = OmegaConf.to_container(value, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Expected a mapping at config path {path!r}.")
    return resolved


class LitCytoBridge(pl.LightningModule):
    def __init__(
        self,
        model_cfg: dict,
        loss_cfg: dict,
        opt_cfg: dict,
        metric_cfg: dict | None = None,
        gradient_audit_cfg: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = CytoBridge(CytoBridgeConfig(**model_cfg))
        self.loss_fn = CytoBridgeLoss(
            CytoBridgeLossConfig(
                lam_recon=loss_cfg.get("lam_recon", 1.0),
                lam_contrast=loss_cfg.get("lam_contrast", 0.5),
                lam_pathway=loss_cfg.get("lam_pathway", 0.3),
                lam_kl=loss_cfg.get("lam_kl", 0.05),
                lam_delta=loss_cfg.get("lam_delta", 0.0),
                lam_direction=loss_cfg.get("lam_direction", 0.0),
                lam_drugspec=loss_cfg.get("lam_drugspec", 0.0),
                lam_logfc=loss_cfg.get("lam_logfc", 0.0),
                lam_norm_recon=loss_cfg.get("lam_norm_recon", 0.0),
                huber_beta=loss_cfg.get("huber_beta", 1.0),
                infonce=InfoNCEConfig(
                    temperature=loss_cfg.get("temperature", 0.07),
                    weight_hard=loss_cfg.get("weight_hard", 2.0),
                ),
            )
        )
        self.opt_cfg = opt_cfg
        self.metric_cfg = metric_cfg or {"enabled": False}
        self.gradient_audit_cfg = gradient_audit_cfg or {"enabled": False}
        self.metric_enabled = bool(self.metric_cfg.get("enabled", False))
        self.gene_panels: dict[str, np.ndarray] = {}
        if self.metric_enabled:
            panel_path = Path(str(self.metric_cfg.get("gene_panels_path", "")))
            if not panel_path.is_file():
                raise FileNotFoundError(
                    "Validation conditional scoring requires metric.gene_panels_path; "
                    f"missing {panel_path}."
                )
            panel_payload = json.loads(panel_path.read_text())
            self.gene_panels = {
                str(context): np.asarray(indices, dtype=int)
                for context, indices in panel_payload.items()
            }
        self._validation_cache: dict[str, list] = {}

    @staticmethod
    def _assert_finite_mapping(values: dict, *, label: str) -> None:
        for name, value in values.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                bad = int((~torch.isfinite(value)).sum().detach().cpu())
                raise FloatingPointError(
                    f"Non-finite tensor in {label}/{name}: {bad} elements."
                )

    def forward(self, batch):
        return self.model(
            cell_tokens=batch["cell_tokens"],
            drug_tokens=batch["drug_tokens"],
            drug_mask=batch["drug_mask"],
            control_counts=batch.get(
                "input_control_counts", batch.get("control_counts")
            ),
        )

    def _step(self, batch, stage: str, *, audit_gradients: bool = False):
        out = self.forward(batch)
        self._assert_finite_mapping(out, label=f"{stage}_forward_anchor")
        # Build positive view via dropout on z (data aug)
        out2 = self.forward(batch)
        self._assert_finite_mapping(out2, label=f"{stage}_forward_positive")
        out["z_pos"] = out2["z"]
        if "hn_cell_emb" in batch and batch["hn_cell_emb"].numel() > 0:
            B, N, L_c, d_cell = batch["hn_cell_emb"].shape
            _, _, L_d = batch["hn_drug_mask"].shape
            hn_batch = {
                "cell_tokens": batch["hn_cell_emb"].reshape(B * N, L_c, d_cell),
                "drug_tokens": batch["hn_drug_emb"].reshape(B * N, L_d, -1),
                "drug_mask": batch["hn_drug_mask"].reshape(B * N, L_d),
            }
            hn_out = self.forward(hn_batch)
            self._assert_finite_mapping(hn_out, label=f"{stage}_forward_hard_negative")
            out["z_hard_neg"] = hn_out["z"].reshape(B, N, -1)
        else:
            out["z_hard_neg"] = None
        loss_dict = self.loss_fn(
            out,
            batch,
            include_raw_components=audit_gradients,
        )
        raw_components = loss_dict.pop("_raw_components", None)
        self._assert_finite_mapping(loss_dict, label=f"{stage}_loss")
        if raw_components is not None:
            audit = measure_component_gradient_norms(
                raw_components,
                component_weights(self.loss_fn.cfg),
                list(self.model.named_parameters()),
            )
            for component, values in audit.items():
                self.log(
                    f"gradient_audit/{component}",
                    values["weighted_gradient_l2"],
                    on_step=True,
                    on_epoch=False,
                )
            if self.trainer.is_global_zero:
                output_path = Path(
                    str(
                        self.gradient_audit_cfg.get(
                            "output_path", "logs/gradient_norms.jsonl"
                        )
                    )
                )
                append_gradient_audit(
                    output_path,
                    {
                        "epoch": int(self.current_epoch),
                        "global_step": int(self.global_step),
                        "components": audit,
                    },
                )
        for k, v in loss_dict.items():
            self.log(f"{stage}/{k}", v, prog_bar=(k == "loss"), on_epoch=True)
        if stage == "val" and self.metric_enabled:
            truth_control = batch.get("truth_control_counts")
            if truth_control is None:
                raise ValueError(
                    "Validation conditional scoring requires disjoint truth_control_counts."
                )
            self._validation_cache["pred_treated"].append(
                out["mu"].detach().cpu().numpy()
            )
            self._validation_cache["true_treated"].append(
                batch["treated_counts"].detach().cpu().numpy()
            )
            self._validation_cache["truth_control"].append(
                truth_control.detach().cpu().numpy()
            )
            self._validation_cache["drug_ids"].extend(map(str, batch["drug_ids"]))
            self._validation_cache["context_ids"].extend(map(str, batch["cell_lines"]))
        return loss_dict["loss"]

    def on_before_optimizer_step(self, optimizer) -> None:
        del optimizer
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                bad = int((~torch.isfinite(parameter.grad)).sum().detach().cpu())
                raise FloatingPointError(
                    f"Non-finite optimizer gradient in {name}: {bad} elements."
                )

    def on_before_zero_grad(self, optimizer) -> None:
        del optimizer
        for name, parameter in self.model.named_parameters():
            if not torch.isfinite(parameter).all():
                bad = int((~torch.isfinite(parameter)).sum().detach().cpu())
                raise FloatingPointError(
                    f"Non-finite parameter after optimizer step in {name}: {bad} elements."
                )

    def training_step(self, batch, batch_idx):
        audit_gradients = (
            bool(self.gradient_audit_cfg.get("enabled", False))
            and int(self.global_step) == 0
            and int(batch_idx) == 0
        )
        return self._step(batch, "train", audit_gradients=audit_gradients)

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def on_validation_epoch_start(self) -> None:
        self._validation_cache = {
            "pred_treated": [],
            "true_treated": [],
            "truth_control": [],
            "drug_ids": [],
            "context_ids": [],
        }

    def on_validation_epoch_end(self) -> None:
        if not self.metric_enabled:
            return
        cache = self._validation_cache
        if not cache["pred_treated"]:
            raise RuntimeError(
                "Validation metric is enabled but no validation batches were cached."
            )
        metrics = compute_validation_metrics(
            np.concatenate(cache["pred_treated"]),
            np.concatenate(cache["true_treated"]),
            np.concatenate(cache["truth_control"]),
            cache["drug_ids"],
            cache["context_ids"],
            self.gene_panels,
        )
        self.log(
            "val/conditional_accuracy_drug_macro",
            metrics["conditional_accuracy_drug_macro"],
            prog_bar=True,
            on_epoch=True,
        )
        self.log(
            "val/pair_own_spearman_top50_drug_macro",
            metrics["pair_own_spearman_top50_drug_macro"],
            prog_bar=False,
            on_epoch=True,
        )
        self._validation_cache = {}

    def configure_optimizers(self):
        # Different LR for projections vs bridge
        proj_prefixes = ("backbone.cell_proj",)
        named = list(self.model.named_parameters())
        proj_named = [(n, p) for n, p in named if n.startswith(proj_prefixes)]
        bridge_named = [(n, p) for n, p in named if not n.startswith(proj_prefixes)]
        proj_ids = {id(p) for _, p in proj_named}
        bridge_ids = {id(p) for _, p in bridge_named}
        if proj_ids & bridge_ids or len(proj_ids | bridge_ids) != len(
            {id(p) for _, p in named}
        ):
            raise RuntimeError(
                "Optimizer parameter routing produced overlapping or missing groups."
            )
        param_groups = [
            {"params": [p for _, p in proj_named], "lr": self.opt_cfg["lr_proj"]},
            {"params": [p for _, p in bridge_named], "lr": self.opt_cfg["lr_bridge"]},
        ]
        opt = torch.optim.AdamW(param_groups, weight_decay=self.opt_cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.opt_cfg["max_steps"],
            eta_min=1e-6,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


class LexicographicCheckpoint(pl.Callback):
    """Save best/last checkpoints using the exact preregistered epoch ordering."""

    primary_name = f"val/{PRIMARY}"
    tie_breaker_name = f"val/{TIE_BREAKER}"

    def __init__(self, *, dirpath: Path, state_path: Path, patience: int | None):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.state_path = Path(state_path)
        self.tracker = EpochSelectionTracker(patience=patience)

    @staticmethod
    def _metric_value(metrics: dict, name: str) -> float:
        if name not in metrics:
            raise RuntimeError(f"Validation callback metric is missing: {name}")
        value = metrics[name]
        if torch.is_tensor(value):
            value = value.detach().double().cpu().item()
        return float(value)

    @staticmethod
    def _temporary_path(path: Path) -> Path:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"Refusing to reuse checkpoint temporary path: {temporary}")
        return temporary

    def _save_checkpoint(self, trainer: pl.Trainer, path: Path) -> None:
        temporary = self._temporary_path(path)
        trainer.save_checkpoint(str(temporary))
        os.replace(temporary, path)

    def _copy_checkpoint(self, source: Path, destination: Path) -> None:
        temporary = self._temporary_path(destination)
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)

    def _write_state(self) -> None:
        payload = {
            "schema_version": 1,
            "selection_rule": [PRIMARY, TIE_BREAKER, "earlier_epoch"],
            "best_checkpoint": str(self.dirpath / "best.ckpt"),
            "last_checkpoint": str(self.dirpath / "last.ckpt"),
            **self.tracker.to_dict(),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._temporary_path(self.state_path)
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n")
        os.replace(temporary, self.state_path)

    def state_dict(self) -> dict:
        return {"tracker": self.tracker.to_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        restored = EpochSelectionTracker.from_dict(state_dict["tracker"])
        if restored.patience != self.tracker.patience:
            raise ValueError("Checkpoint selection patience differs from the active config.")
        self.tracker = restored

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if trainer.world_size != 1:
            raise RuntimeError(
                "Lexicographic checkpointing is registered for one GPU per campaign job."
            )

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if trainer.sanity_checking:
            return
        primary = self._metric_value(trainer.callback_metrics, self.primary_name)
        tie_breaker = self._metric_value(
            trainer.callback_metrics, self.tie_breaker_name
        )
        epoch = int(trainer.current_epoch) + 1
        record = self.tracker.update(
            primary=primary,
            tie_breaker=tie_breaker,
            epoch=epoch,
        )
        self.dirpath.mkdir(parents=True, exist_ok=True)
        last_path = self.dirpath / "last.ckpt"
        self._save_checkpoint(trainer, last_path)
        if record["improved"]:
            self._copy_checkpoint(last_path, self.dirpath / "best.ckpt")
        self._write_state()
        if record["should_stop"]:
            trainer.should_stop = True


@hydra.main(version_base=None, config_path="configs", config_name="train/v1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed, workers=True)
    write_run_metadata(
        cfg,
        OmegaConf.select(cfg, "run_metadata_path", default="run_metadata.json"),
    )
    resume_ckpt = resolve_resume_ckpt(cfg)

    dm = CytoBridgeDataModule(**cfg.data)
    lit = LitCytoBridge(
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
        loss_cfg=OmegaConf.to_container(cfg.loss, resolve=True),
        opt_cfg=OmegaConf.to_container(cfg.optim, resolve=True),
        metric_cfg=_config_mapping(cfg, "metric", default={"enabled": False}),
        gradient_audit_cfg=_config_mapping(
            cfg, "gradient_audit", default={"enabled": False}
        ),
    )

    if cfg.wandb.use:
        logger = WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name)
    else:
        # No wandb (e.g. offline run): a None logger makes LearningRateMonitor
        # crash, so fall back to a CSVLogger (metrics under logs/<run_name>/).
        from pytorch_lightning.loggers import CSVLogger

        logger = CSVLogger(save_dir="logs", name=cfg.run_name)
    # Anchor checkpoints at the documented `ckpts/<run_name>/` path so the eval
    # scripts and run_pipeline.sh can find `last.ckpt` without scraping logs.
    ckpt_dir = (
        OmegaConf.select(cfg, "ckpt.dirpath", default=None) or f"ckpts/{cfg.run_name}"
    )
    metric_enabled = bool(OmegaConf.select(cfg, "metric.enabled", default=False))
    final_refit = bool(OmegaConf.select(cfg, "trainer.final_refit", default=False))
    if final_refit and metric_enabled:
        raise ValueError(
            "Final train+validation refit must disable validation model selection."
        )
    early_stopping_patience = OmegaConf.select(
        cfg, "trainer.early_stopping_patience", default=None
    )
    if final_refit and early_stopping_patience is not None:
        raise ValueError("Final fixed-epoch refit must disable early stopping.")
    callbacks: list[pl.Callback] = [LearningRateMonitor("step")]
    enable_checkpointing = True
    if final_refit:
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="epoch{epoch:02d}-final",
                auto_insert_metric_name=False,
                monitor=None,
                save_top_k=0,
                save_last=True,
            ),
        )
    elif metric_enabled:
        selection_state_path = Path(
            str(
                OmegaConf.select(
                    cfg,
                    "metric.selection_state_path",
                    default=f"logs/{cfg.run_name}/selection_state.json",
                )
            )
        )
        callbacks.insert(
            0,
            LexicographicCheckpoint(
                dirpath=Path(str(ckpt_dir)),
                state_path=selection_state_path,
                patience=(
                    int(early_stopping_patience)
                    if early_stopping_patience is not None
                    else None
                ),
            ),
        )
        enable_checkpointing = False
    else:
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="epoch{epoch:02d}-val_loss{val/loss:.4f}",
                auto_insert_metric_name=False,
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            ),
        )
        if early_stopping_patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val/loss",
                    mode="min",
                    patience=int(early_stopping_patience),
                    strict=True,
                )
            )
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.get("accelerator", "auto"),
        devices=cfg.trainer.get("devices", "auto"),
        precision=cfg.trainer.precision,
        deterministic=cfg.trainer.get("deterministic", True),
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        limit_val_batches=0 if final_refit else 1.0,
        num_sanity_val_steps=0 if final_refit else 2,
        enable_checkpointing=enable_checkpointing,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
    )
    trainer.fit(lit, datamodule=dm, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
