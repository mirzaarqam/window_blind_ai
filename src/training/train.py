import torch
from torch.utils.data import DataLoader
from model import CustomPipeline, LightingAdjuster
from configs.model_config import Config


def main():
    cfg = Config()

    # Initialize components
    pipeline = CustomPipeline.from_pretrained(cfg.model_name)
    adjuster = LightingAdjuster()

    # Optimizers
    optimizer = torch.optim.AdamW(
        list(pipeline.parameters()) + list(adjuster.parameters()),
        lr=cfg.learning_rate
    )

    # Training loop
    for epoch in range(cfg.epochs):
        for batch in DataLoader(dataset, batch_size=cfg.batch_size):
            # Training logic
            pass

    # Save model
    torch.save({
        'pipeline': pipeline.state_dict(),
        'adjuster': adjuster.state_dict()
    }, 'models/trained/model.pth')


if __name__ == "__main__":
    main()