from loguru import logger
from torchvision import datasets
from pathlib import Path
from src.dataloader import get_flower_dataloaders


def main() -> None:
    # Ensure data directory exists
    try:
        logger.info("📁 Ensuring data directory exists...")
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Data directory exists: {data_dir}")
    except Exception as e:
        logger.error(f"❌ Could not create data directory: {e}")
        raise

    # Download Flowers102 dataset
    try:
        logger.info("📥 Downloading Flowers102 dataset...")
        datasets.Flowers102(root=data_dir, split="train", download=True)
        datasets.Flowers102(root=data_dir, split="val", download=True)
        datasets.Flowers102(root=data_dir, split="test", download=True)
        logger.info(f"✅ Dataset downloaded to: {data_dir}")
    except Exception as e:
        logger.error(f"❌ Could not download dataset: {e}")
        raise

    # Create dataloaders
    try:
        logger.info("🚚 Creating dataloaders...")
        train_loader, val_loader = get_flower_dataloaders(data_dir=str(data_dir))
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        logger.error(f"❌ Could not create dataloaders: {e}")
        raise


if __name__ == "__main__":
    main()
