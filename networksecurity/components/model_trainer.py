
import os
import sys

import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
)
from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)


# --------------------------------------------------
# MLflow: local tracking using SQLite
# --------------------------------------------------

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "NetworkSecurity"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = (
                data_transformation_artifact
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    # --------------------------------------------------
    # Log metrics only; avoid the current skops model
    # serialization error.
    # --------------------------------------------------

    def track_mlflow(
        self,
        classification_metric,
        run_name: str,
    ) -> None:
        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_metrics(
                    {
                        "f1_score": float(
                            classification_metric.f1_score
                        ),
                        "precision": float(
                            classification_metric.precision_score
                        ),
                        "recall_score": float(
                            classification_metric.recall_score
                        ),
                    }
                )

                logging.info(
                    f"MLflow metrics logged for run: {run_name}"
                )

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    # --------------------------------------------------
    # Train, tune, compare, and save models
    # --------------------------------------------------

    def train_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> ModelTrainerArtifact:
        try:
            models = {
                "Random Forest": RandomForestClassifier(
                    random_state=42
                ),
                "Decision Tree": DecisionTreeClassifier(
                    random_state=42
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42
                ),
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
                "AdaBoost": AdaBoostClassifier(
                    random_state=42
                ),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            best_models = {}
            model_report = {}

            # Tune every candidate and retain its fitted estimator.
            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=params[model_name],
                    cv=3,
                    scoring="f1_weighted",
                    n_jobs=-1,
                    refit=True,
                    error_score="raise",
                )

                grid_search.fit(X_train, y_train)

                fitted_model = grid_search.best_estimator_
                best_models[model_name] = fitted_model

                test_predictions = fitted_model.predict(X_test)
                test_metric = get_classification_score(
                    y_true=y_test,
                    y_pred=test_predictions,
                )

                model_report[model_name] = float(
                    test_metric.f1_score
                )

                logging.info(
                    f"{model_name} best parameters: "
                    f"{grid_search.best_params_}"
                )
                logging.info(
                    f"{model_name} test F1 score: "
                    f"{test_metric.f1_score}"
                )

            if not model_report:
                raise ValueError("No models were successfully evaluated.")

            best_model_name = max(
                model_report,
                key=model_report.get,
            )
            best_model = best_models[best_model_name]

            logging.info(f"Best model: {best_model_name}")
            logging.info(f"Model scores: {model_report}")

            # Evaluate the selected, fitted model on training data.
            y_train_pred = best_model.predict(X_train)
            train_metric = get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred,
            )

            self.track_mlflow(
                classification_metric=train_metric,
                run_name="training_metrics",
            )

            # Evaluate the selected, fitted model on test data.
            y_test_pred = best_model.predict(X_test)
            test_metric = get_classification_score(
                y_true=y_test,
                y_pred=y_test_pred,
            )

            self.track_mlflow(
                classification_metric=test_metric,
                run_name="testing_metrics",
            )

            # Load the fitted preprocessing object.
            preprocessor = load_object(
                file_path=(
                    self.data_transformation_artifact
                    .transformed_object_file_path
                )
            )

            # Save the complete model pipeline.
            model_file_path = (
                self.model_trainer_config.trained_model_file_path
            )
            model_dir = os.path.dirname(model_file_path)

            if model_dir:
                os.makedirs(model_dir, exist_ok=True)

            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model,
            )

            save_object(
                file_path=model_file_path,
                obj=network_model,
            )

            # Save separate artifacts for deployment if required.
            os.makedirs("final_model", exist_ok=True)

            save_object(
                file_path="final_model/preprocessor.pkl",
                obj=preprocessor,
            )
            save_object(
                file_path="final_model/model.pkl",
                obj=best_model,
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )

            logging.info(
                f"Model trainer artifact: {model_trainer_artifact}"
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    # --------------------------------------------------
    # Pipeline entry point
    # --------------------------------------------------

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = (
                self.data_transformation_artifact
                .transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact
                .transformed_test_file_path
            )

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            return self.train_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
