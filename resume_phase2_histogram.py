"""
resume_phase2_histogram.py
────────────────────────────────────────────────────────────────────────────
Resumes the interrupted phase-2 (+histogram loss) fine-tuning run from its
last intact checkpoint, restoring optimizer + scheduler + epoch count so it
continues the SAME cosine schedule where it left off — unlike
retrain_phase2_histogram.py, which intentionally starts a fresh schedule.

Use this only to recover from an interruption (crash, disk-full, server
reboot, etc.) of the checkpoints_phase2_hist/ run. Point RESUME_FROM at the
last checkpoint file confirmed intact (check file size / mtime looks sane,
not 0 bytes or truncated).
"""

from __future__ import annotations

from src.training.trainer import PUWDMTrainer, TrainerConfig

# Update this to the last INTACT checkpoint after checking
# `ls -la checkpoints_phase2_hist/` on the server.
RESUME_FROM = "checkpoints_phase2_hist/epoch_0015.pt"


def main() -> None:
    cfg = TrainerConfig(
        checkpoint_dir="checkpoints_phase2_hist",
        log_dir="runs/p_uwdm_phase2_hist",
        total_epochs=120,
        phase1_epochs=0,
        batch_size=16,
        use_amp=True,
        compile_model=True,
        ema_decay=0.999,
        ema_update_every=10,
        save_every_n_epochs=5,
        keep_last_n_checkpoints=3,
    )

    trainer = PUWDMTrainer(cfg)
    trainer.fit(resume_from=RESUME_FROM)


if __name__ == "__main__":
    main()
