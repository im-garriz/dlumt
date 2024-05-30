from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from typing import Dict, List
import os
import pytorch_lightning as pl


class Custom_PL_MLFlowLogger(MLFlowLogger):
    """
    A custom PyTorch Lightning MLFlowLogger for enhanced logging functionalities.

    This logger extends the MLFlowLogger to include additional features such as
    logging source code and ensuring successful finalization even when interrupted.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Custom_PL_MLFlowLogger.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.best_checkpoint_logged = False
        mlflow.set_tracking_uri(kwargs["tracking_uri"])

    # No deja sobrescribir parametros
    def log_hyperparams(self, params: Dict) -> None:
        """
        Log hyperparameters, suppressing any errors that occur.

        Args:
            params (Dict): Dictionary of hyperparameters to log.
        """
        try:
            super().log_hyperparams(params)
        except:
            pass

    def after_save_checkpoint(
        self, checkpoint_callback: pl.callbacks.ModelCheckpoint
    ) -> None:
        """
        Log the best and last model checkpoints as MLFlow artifacts.

        Args:
            checkpoint_callback (pl.callbacks.ModelCheckpoint): The checkpoint callback instance.
        """

        if os.path.isfile(checkpoint_callback.best_model_path):
            self.experiment.log_artifact(
                self.run_id,
                checkpoint_callback.best_model_path,
                artifact_path="models/best",
            )
        else:
            print(
                f"[W] Checkpoint best:{checkpoint_callback.best_model_path} does not exist at disk."
            )

        # During epoch 0, only the best is saved, not the last
        if os.path.isfile(checkpoint_callback.last_model_path):
            self.experiment.log_artifact(
                self.run_id,
                checkpoint_callback.last_model_path,
                artifact_path="models/last",
            )
        else:
            print(
                f"[W] Checkpoint last:{checkpoint_callback.last_model_path} does not exist at disk."
            )

        # Log the absolute path of the best checkpoint as a parameter
        if not self.best_checkpoint_logged:
            self.experiment.log_param(
                self.run_id, "run_checkpoint", checkpoint_callback.best_model_path
            )

        return super().after_save_checkpoint(checkpoint_callback)

    def log_source_code(self, path: str, extensions: List[str]) -> None:
        """
        Log source code files as MLFlow artifacts.

        Args:
            path (str): The root directory path to search for source code files.
            extensions (List[str]): List of file extensions to log.
        """
        for root, _, files in os.walk(path):
            for file in files:
                for ext in extensions:
                    if file.endswith(ext):
                        folders = root.replace(path, "")
                        if folders.startswith("/"):
                            folders = folders[1:]

                        artifact_path = os.path.join("SOURCE_CODE", folders)
                        if artifact_path.endswith("/"):
                            artifact_path = artifact_path[:-1]

                        self.experiment.log_artifact(
                            self.run_id,
                            os.path.join(root, file),
                            artifact_path=artifact_path,
                        )

    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        """
        Finalize the logging run with a success status.

        Ensures that the run always ends with a 'success' status, even if interrupted,
        to allow further configuration and data logging.

        Args:
            status (str, optional): The final status of the run. Defaults to "success".
        """
        super().finalize(status="success")
