import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="configs",
    config_name="train",
    version_base="1.2",
)
def main(config: DictConfig):
    trainer = getattr(
        __import__("usdsgen.trainer", fromlist=[""]), f"{config.task}Trainer"
    )(config)
    if config.mode == "train":
        trainer.fit()
    elif config.mode == "test":
        trainer.test()
    else:
        raise ValueError(f"Invalid mode: {config.mode}")


if __name__ == "__main__":
    main()
