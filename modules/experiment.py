import os
import json
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, dataclass

from modules.data import (
    FootballDataset,
    FootballDataLoader,
    FootballDatasetConfig,
    FootballDataLoaderConfig,
)
from modules.model import FootballNet, FootballNetConfig
from modules.trainer import TrainerConfig, Trainer
from modules.evaluator import EvaluatorConfig, Evaluator


@dataclass
class ExperimentConfig:
    """
    Top-level configuration object for a full experiment run.
    Holds dataset, dataloader, model, training, and evaluation configs.
    """

    # Directory where all outputs (logs, checkpoints, results) will be saved
    output_dir: str

    # Unique experiment name (specififc experiment logging directory)
    experiment_name: str

    # Training dataset and dataloader configuration
    train_dataset_config: FootballDatasetConfig
    train_dataloader_config: FootballDataLoaderConfig

    # Validation dataset and dataloader configuration
    valid_dataset_config: FootballDatasetConfig
    valid_dataloader_config: FootballDataLoaderConfig

    # Test dataset and dataloader configuration
    test_dataset_config: FootballDatasetConfig
    test_dataloader_config: FootballDataLoaderConfig

    # Model architecture configuration
    model_config: FootballNetConfig

    # Training hyperparameters (epochs, optimizer, lr, etc.)
    trainer_config: TrainerConfig

    # Evaluation configuration (IoU threshold, device, etc.)
    evaluator_config: EvaluatorConfig


class Experiment:
    """
    Experiment class to manage complete training and evaluation pipeline.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

        # Setup paths
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.output_path = os.path.join(
            config.output_dir, config.experiment_name, timestamp
        )
        self.output_path = os.path.join(config.output_dir, config.experiment_name)
        self.log_path = os.path.join(self.output_path, "logs")
        self.model_path = os.path.join(self.output_path, "checkpoints")

        # Create directories
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._setup_datasets()
        self._setup_dataloaders()
        self._setup_model()
        self._setup_writer()
        self._setup_trainer()
        self._setup_evaluator()

    def _setup_datasets(self):
        """
        Initialize datasets.
        """
        self.train_datasets = FootballDataset(self.config.train_dataset_config)
        self.valid_datasets = FootballDataset(self.config.valid_dataset_config)
        self.test_datasets = FootballDataset(self.config.test_dataset_config)

    def _setup_dataloaders(self):
        """
        Initialize data loaders.
        """
        if self.train_datasets is None:
            raise RuntimeError(
                "Cannot setup train dataloader without corresponding datasets"
            )
        if self.valid_datasets is None:
            raise RuntimeError(
                "Cannot setup valid dataloader without corresponding datasets"
            )
        if self.test_datasets is None:
            raise RuntimeError(
                "Cannot setup test dataloader without corresponding datasets"
            )
        self.train_loader = FootballDataLoader(
            self.config.train_dataloader_config, self.train_datasets
        )
        self.valid_loader = FootballDataLoader(
            self.config.valid_dataloader_config, self.valid_datasets
        )
        self.test_loader = FootballDataLoader(
            self.config.test_dataloader_config, self.test_datasets
        )

    def _setup_model(self):
        """
        Initialize model.
        """
        self.model = FootballNet(self.config.model_config)

    def _setup_writer(self):
        """
        Initialize tensorboard writer to log experiment info.
        """
        self.writer = SummaryWriter(self.log_path)

    def _setup_trainer(self):
        """
        Initialize trainer with tensorboard writer.
        """
        if self.train_loader is None:
            raise RuntimeError("Cannot set up trainer without a training dataloader.")
        if self.valid_loader is None:
            raise RuntimeError("Cannot set up trainer without a validation dataloader.")
        if self.model is None:
            raise RuntimeError("Cannot set up trainer without a model.")
        if self.writer is None:
            raise RuntimeError("Cannot set up trainer without a tensorboard writer.")

        self.trainer = Trainer(
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            model=self.model,
            writer=self.writer,
            config=self.config.trainer_config,
            model_path=self.model_path,
        )

    def _setup_evaluator(self):
        """
        Initialize evaluator.
        """
        self.evaluator = Evaluator(
            model=self.model,
            config=self.config.evaluator_config,
        )

    def save_config(self):
        """
        Save experiment configuration.
        """
        config_dict = asdict(self.config)
        config_path = os.path.join(self.output_path, "experiment_config.json")

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4, default=str)

    def run(self):
        """
        Run complete experiment: training + evaluation.
        """
        print(f"Starting experiment: {self.config.experiment_name}")
        print(f"Output directory: {self.output_path}")

        # Save configuration
        self.save_config()
        print("Configuration saved.")

        # Train model
        print("\n" + "=" * 50)
        print("TRAINING PHASE")
        print("=" * 50)
        self.trainer.train()

        # Evaluate on all datasets
        print("\n" + "=" * 50)
        print("EVALUATION PHASE")
        print("=" * 50)

        # Evaluate on all three datasets
        train_metrics = self.evaluator.evaluate(self.train_loader)
        valid_metrics = self.evaluator.evaluate(self.valid_loader)
        test_metrics = self.evaluator.evaluate(self.test_loader)

        # Log metrics for all datasets to tensorboard
        datasets = {
            "Train": train_metrics,
            "Validation": valid_metrics,
            "Test": test_metrics,
        }

        all_metrics = {}

        for dataset_name, metrics in datasets.items():
            print(f"\n{dataset_name} Metrics:")
            print("-" * 30)
            for metric_name, metric_value in metrics.items():
                # Log to tensorboard
                self.writer.add_scalar(f"{dataset_name}/{metric_name}", metric_value)
                # Print to console
                print(f"{metric_name}: {metric_value:.4f}")
                # Store for saving
                all_metrics[f"{dataset_name.lower()}_{metric_name}"] = metric_value

        # Save all metrics to JSON file
        metrics_path = os.path.join(self.output_path, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"\nAll metrics saved to: {metrics_path}")

        # Close tensorboard writer
        self.writer.close()

        print(f"\nExperiment completed! Results saved to: {self.output_path}")
        return all_metrics


# Example usage:
# def run_experiment(config_path: str | None = None):
#     """
#     Run experiment from configuration file.

#     Args:
#         config_path: Path to configuration file (optional)
#     """
#     if config_path:
#         # Load config from file
#         with open(config_path, "r") as f:
#             config_data = json.load(f)
#         config = Config(**config_data)
#     else:
#         # Use default config
#         config = Config(
#             train_dataset=FootballDatasetConfig(...),  # Your actual configs
#             train_dataloader=FootballDataLoaderConfig(...),
#             valid_dataset=FootballDatasetConfig(...),
#             valid_dataloader=FootballDataLoaderConfig(...),
#             test_dataset=FootballDatasetConfig(...),
#             test_dataloader=FootballDataLoaderConfig(...),
#             model=ModelConfig(...),
#             trainer=TrainerConfig(),
#             evaluation=EvaluatorConfig(),
#         )

#     # Create and run experiment
#     experiment = Experiment(config)
#     results = experiment.run()

#     return results


# if __name__ == "__main__":
#     # Example of running an experiment
#     results = run_experiment()
#     print(f"Final results: {results}")
