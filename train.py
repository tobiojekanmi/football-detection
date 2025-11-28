from modules.data import FootballDatasetConfig, FootballDataLoaderConfig
from modules.model import FootballNetConfig
from modules.evaluator import EvaluatorConfig
from modules.trainer import TrainerConfig
from modules.experiment import ExperimentConfig, Experiment


# IMAGE_CHANNELS = 3
# IMAGE_HEIGHT = 224
# IMAGE_WIDTH = 256

IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920

DEVICE = "mps"

# -------------------------------------------------------------------------------------------
# Prepare Datasets and Dataloaders
# -------------------------------------------------------------------------------------------
train_set_config = FootballDatasetConfig(
    root_data_path="./datasets/Soccerball",
    dataset_category="train",
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    use_augmentation=False,
    normalize_image=True,
    normalize_bbox=True,
)
train_loader_config = FootballDataLoaderConfig(
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
)

valid_set_config = FootballDatasetConfig(
    root_data_path="./datasets/Soccerball",
    dataset_category="valid",
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    use_augmentation=False,
    normalize_image=True,
    normalize_bbox=True,
)
valid_loader_config = FootballDataLoaderConfig(
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
)

test_set_config = FootballDatasetConfig(
    root_data_path="./datasets/Soccerball",
    dataset_category="test",
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    use_augmentation=False,
    normalize_image=True,
    normalize_bbox=True,
)
test_loader_config = FootballDataLoaderConfig(
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
)


# -------------------------------------------------------------------------------------------
# Prepare Model
# -------------------------------------------------------------------------------------------
model_config = FootballNetConfig(
    input_shape=(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
    out_channels=64,
    num_res_blocks=5,
    output_dim=4,
)


# -------------------------------------------------------------------------------------------
# Prepare Evaluator and Trainer
# -------------------------------------------------------------------------------------------
evaluator_config = EvaluatorConfig(
    iou_threshold=0.5,
    device_type=DEVICE,
)
trainer_config = TrainerConfig(
    device_type=DEVICE,
    learning_rate=1e-3,
    weight_decay=1e-5,
    num_epochs=50,
    evaluate_every=1,
    save_every=5,
    use_scheduler=True,
    scheduler_step_size=5,
    scheduler_gamma=0.5,
    loss_type="mse",
)


# -------------------------------------------------------------------------------------------
# Prepare Experiment
# -------------------------------------------------------------------------------------------
experiment_config = ExperimentConfig(
    output_dir="outputs",
    experiment_name="exp_v1",
    train_dataset_config=train_set_config,
    train_dataloader_config=train_loader_config,
    valid_dataset_config=valid_set_config,
    valid_dataloader_config=valid_loader_config,
    test_dataset_config=test_set_config,
    test_dataloader_config=test_loader_config,
    model_config=model_config,
    trainer_config=trainer_config,
    evaluator_config=evaluator_config,
)


# -------------------------------------------------------------------------------------------
# Run experiment
# -------------------------------------------------------------------------------------------
experiment = Experiment(config=experiment_config)
if __name__ == "__main__":
    experiment.run()
