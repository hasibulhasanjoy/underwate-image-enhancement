"""
retrain_phase2_histogram.py
────────────────────────────────────────────────────────────────────────────
Fine-tuning extension: adds histogram loss on top of the existing
phase-1/phase-2 checkpoint (best.pt, epoch 257).

Loads MODEL + EMA WEIGHTS ONLY from checkpoints/best.pt — not the optimizer,
scheduler, or epoch count. This run gets its own fresh warmup + cosine LR
decay and its own checkpoint directory, so:
  - the old best.pt / checkpoints/ from the first run are untouched (safe
    fallback if this run doesn't improve things)
  - the LR schedule isn't inherited already-decayed to ~eta_min from the
    end of the previous run

phase1_epochs=0 means the trainer enters phase 2 immediately (diffusion +
perceptual + histogram, gated to t<200) since the loaded model already
completed diffusion pretraining and 17 epochs of phase-2 in the prior run.

Effective total training so far: 257 (previous run) + 120 (this run) = 377
epochs, landing in the 350-400 range. 120 epochs of phase-2 fine-tuning is
already double the length of the original phase-2 (60 epochs), which is
plenty of room for the histogram loss to take effect without overfitting
on the 623-image UIEB train split.

Usage:
    cd ~/underwater_image_enhancement && source venv/bin/activate
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python \\
        retrain_phase2_histogram.py > training_phase2_hist.log 2>&1 &
    echo $! > train_phase2_hist.pid
"""

from __future__ import annotations

from src.training.trainer import PUWDMTrainer, TrainerConfig

SOURCE_CHECKPOINT = "checkpoints/best.pt"  # epoch 257, pre-histogram-loss


def main() -> None:
    cfg = TrainerConfig(
        checkpoint_dir="checkpoints_phase2_hist",
        log_dir="runs/p_uwdm_phase2_hist",
        total_epochs=120,      # new epochs in THIS run (fresh count, see docstring)
        phase1_epochs=0,       # skip phase 1 entirely — go straight to phase 2
        batch_size=16,
        use_amp=True,
        compile_model=True,
        ema_decay=0.999,
        ema_update_every=10,
        save_every_n_epochs=5,
        keep_last_n_checkpoints=3,
    )

    trainer = PUWDMTrainer(cfg)
    trainer.load_weights_from(SOURCE_CHECKPOINT)
    trainer.fit(resume_from=None)


if __name__ == "__main__":
    main()