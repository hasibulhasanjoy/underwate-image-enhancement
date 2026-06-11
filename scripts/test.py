from pathlib import Path

from src.utils.config import DataConfig
from src.data.datamodule import UIEBDataModule


def main():
    print("\n🚀 Starting UIEB Data Pipeline Debug...\n")

    # ------------------------------------------------------------
    # 1. Load config (use your default or YAML-loaded config)
    # ------------------------------------------------------------
    config = DataConfig()

    print("📦 Dataset Config:")
    print("  name      :", config.dataset.name)
    print("  root_dir  :", config.dataset.root_dir)
    print("  raw_subdir:", config.dataset.raw_subdir)
    print("  ref_subdir:", config.dataset.reference_subdir)
    print()

    # ------------------------------------------------------------
    # 2. Create DataModule
    # ------------------------------------------------------------
    dm = UIEBDataModule(config=config, project_root=".")  # change if needed

    # ------------------------------------------------------------
    # 3. Setup pipeline
    # ------------------------------------------------------------
    dm.setup()

    print("\n✅ DataModule setup complete")

    # ------------------------------------------------------------
    # 4. Check dataset sizes
    # ------------------------------------------------------------
    print("\n📊 Dataset splits:")
    print("  Train size:", len(dm.train_dataset))
    print("  Val size  :", len(dm.val_dataset))
    print("  Test size :", len(dm.test_dataset))

    # ------------------------------------------------------------
    # 5. Test one sample from dataset
    # ------------------------------------------------------------
    raw, ref, meta = dm.train_dataset[0]

    print("\n🧪 Sample check (index 0):")
    print("  Raw type :", type(raw))
    print("  Ref type :", type(ref))
    print("  Meta     :", meta)

    # ------------------------------------------------------------
    # 6. Test DataLoader
    # ------------------------------------------------------------
    train_loader = dm.train_dataloader()

    batch = next(iter(train_loader))

    print("\n📦 Batch check:")
    print("  Batch size:", len(batch))
    print("  Raw batch type:", type(batch[0]))
    print("  Ref batch type:", type(batch[1]))
    print("  Meta type:", type(batch[2]))

    print("\n🎉 Everything looks OK if no errors above!\n")


if __name__ == "__main__":
    main()
