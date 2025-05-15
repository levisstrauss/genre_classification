import mlflow
import os
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    wandb.config = OmegaConf.to_container(
         config, 
         resolve=True, 
         throw_on_missing=True
    )

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:

        steps_to_execute = list(config["main"]["execute_steps"])

    # Download step -> It download file from the internet and save it as artifact
    if "download" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "download"), # Path to the component
            "main", # Entry point of the component
            parameters={ # Look at the MLproject file to understand the parameters
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet", #  Name of the output artifact
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            },
        )

    if "preprocess" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            "main",
            parameters={
                "input_artifact": "raw_data.parquet:latest", # the input here is the output of the previous step
                "artifact_name": "preprocessed_data.csv", # output
                "artifact_type": "preprocessed_data", # Type
                "artifact_description": "Data with preprocessing applied" # Description
            },
        )

    if "check_data" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            "main",
            parameters={
                "reference_artifact": config["data"]["reference_dataset"],
                "sample_artifact": "preprocessed_data.csv:latest",
                "ks_alpha": config["data"]["ks_alpha"]
            },
        )

    if "segregate" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters={
                "input_artifact": "preprocessed_data.csv:latest",
                "artifact_root": "data", # We are taking root here because this is generating two artifact
                "artifact_type": "segregated_data",
                "test_size": config["data"]["test_size"], # test size
                "stratify": config["data"]["stratify"]
            },
        )

    if "random_forest" in steps_to_execute:
        # Serialize decision tree configuration
        model_config = os.path.abspath("random_forest_config.yml") # Load all the random forest params

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            "main",
            parameters={
                "train_data": "data_train.csv:latest",
                "model_config": model_config,
                "export_artifact": config["random_forest_pipeline"]["export_artifact"],
                "random_seed": config["main"]["random_seed"],
                "val_size": config["data"]["test_size"],
                "stratify": config["data"]["stratify"]
            },
        )

    if "evaluate" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "evaluate"),
            "main",
            parameters={
                "model_export": f"{config['random_forest_pipeline']['export_artifact']}:latest",
                "test_data": "data_test.csv:latest"
            },
        )


if __name__ == "__main__":
    go()
