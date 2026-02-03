"""
Classical Machine Learning Model Factory.
"""

from typing import Any

from ..enums.models import MacModelType
from .mac import (
    AdaBoostModel,
    AODEModel,
    BaggingModel,
    BayesianNetworkModel,
    C45Model,
    C50Model,
    CARTModel,
    CHAIDModel,
    ConditionalDecisionTreeModel,
    DecisionStumpModel,
    DecisionTreeModel,
    ElasticNetModel,
    GaussianNaiveBayesModel,
    GBRTModel,
    GradientBoostingModel,
    LARSModel,
    LassoRegressionModel,
    LightGBMModel,
    LinearRegressionModel,
    LinearSVMModel,
    LOESSModel,
    LogisticRegressionModel,
    LSSVMModel,
    LWLModel,
    M5Model,
    MARSModel,
    MultinomialNaiveBayesModel,
    NaiveBayesModel,
    NuSVMModel,
    OLSRModel,
    OneClassSVMModel,
    PolynomialRegressionModel,
    RandomForestModel,
    RidgeRegressionModel,
    StackingModel,
    StepwiseRegressionModel,
    SVMModel,
    SVRModel,
    TWSVMModel,
    VotingModel,
    WeightedAverageModel,
    XGBoostModel,
    kNNModel,
)
from .mac.base import ClassicalModel

# List of MAC model names


def create_mac_model(model_name: str, cfg: dict[str, Any]) -> ClassicalModel | None:  # noqa: PLR0911
    """
    Factory function to create classical machine learning models.

    Args:
        model_name: Name of the model to create.
        cfg: Configuration dictionary.

    Returns:
        Instantiated model or None if not a MAC model.
    """
    try:
        model_type = MacModelType(model_name)
    except ValueError:
        return None

    if model_type == MacModelType.LINEAR_REGRESSION:
        return LinearRegressionModel(**cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.RIDGE:
        return RidgeRegressionModel(alpha=cfg.get("alpha", 1.0), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.LASSO:
        return LassoRegressionModel(alpha=cfg.get("alpha", 1.0), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.LARS:
        return LARSModel(
            n_nonzero_coefs=cfg.get("n_nonzero_coefs", 500),
            **cfg.get("model_kwargs", {}),
        )
    elif model_type == MacModelType.ELASTIC_NET:
        return ElasticNetModel(
            alpha=cfg.get("alpha", 1.0),
            l1_ratio=cfg.get("l1_ratio", 0.5),
            **cfg.get("model_kwargs", {}),
        )
    elif model_type == MacModelType.LOGISTIC_REGRESSION:
        return LogisticRegressionModel(**cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.POLYNOMIAL:
        return PolynomialRegressionModel(degree=cfg.get("degree", 2), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.DECISION_TREE:
        return DecisionTreeModel(
            task=cfg.get("task", "regression"),
            max_depth=cfg.get("max_depth", None),
            **cfg.get("model_kwargs", {}),
        )
    elif model_type == MacModelType.CART:
        return CARTModel(task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.ID3:
        from .mac.trees import ID3Model

        return ID3Model(task=cfg.get("task", "classification"), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.C45:
        return C45Model(task=cfg.get("task", "classification"), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.C50:
        return C50Model(task=cfg.get("task", "classification"), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.CHAID:
        return CHAIDModel(task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.DECISION_STUMP:
        return DecisionStumpModel(task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.CONDITIONAL_TREE:
        return ConditionalDecisionTreeModel(task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.M5:
        return M5Model(**cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.RANDOM_FOREST:
        return RandomForestModel(
            task=cfg.get("task", "regression"),
            n_estimators=cfg.get("n_estimators", 100),
            **cfg.get("model_kwargs", {}),
        )
    elif model_type == MacModelType.GRADIENT_BOOSTING:
        return GradientBoostingModel(
            task=cfg.get("task", "regression"),
            n_estimators=cfg.get("n_estimators", 100),
            **cfg.get("model_kwargs", {}),
        )
    elif model_type == MacModelType.GBM:
        return GBRTModel(task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {}))  # Alias GBM -> GBRT
    elif model_type == MacModelType.GBRT:
        return GBRTModel(task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.ADA_BOOST:
        return AdaBoostModel(
            task=cfg.get("task", "regression"),
            n_estimators=cfg.get("n_estimators", 50),
            **cfg.get("model_kwargs", {}),
        )
    elif model_type == MacModelType.BAGGING:
        return BaggingModel(
            task=cfg.get("task", "regression"),
            n_estimators=cfg.get("n_estimators", 10),
            **cfg.get("model_kwargs", {}),
        )
    elif model_type == MacModelType.STACKING:
        return StackingModel(
            task=cfg.get("task", "regression"),
            final_estimator=cfg.get("final_estimator", None),
            **cfg.get("model_kwargs", {}),
        )
    elif model_type == MacModelType.VOTING:
        return VotingModel(
            task=cfg.get("task", "regression"),
            voting=cfg.get("voting", "hard"),
            **cfg.get("model_kwargs", {}),
        )
    elif model_type == MacModelType.WEIGHTED_AVERAGE:
        return WeightedAverageModel(
            task=cfg.get("task", "regression"),
            weights=cfg.get("weights", None),
            **cfg.get("model_kwargs", {}),
        )
    elif model_type == MacModelType.BLENDING:
        return WeightedAverageModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )  # Placeholder if identical
    elif model_type == MacModelType.XGBOOST:
        return XGBoostModel(task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.LIGHTGBM:
        return LightGBMModel(task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.KNN:
        return kNNModel(
            task=cfg.get("task", "regression"),
            n_neighbors=cfg.get("n_neighbors", 5),
            **cfg.get("model_kwargs", {}),
        )
    elif model_type == MacModelType.SVM:
        return SVMModel(
            task=cfg.get("task", "regression"),
            kernel=cfg.get("kernel", "rbf"),
            **cfg.get("model_kwargs", {}),
        )
    elif model_type == MacModelType.SVR:
        return SVRModel(**cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.LINEAR_SVM:
        return LinearSVMModel(task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.NU_SVM:
        return NuSVMModel(task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.ONE_CLASS_SVM:
        return OneClassSVMModel(**cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.LSSVM:
        return LSSVMModel(**cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.TWSVM:
        return TWSVMModel(**cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.NAIVE_BAYES:
        return NaiveBayesModel(nb_type=cfg.get("type", "gaussian"), **cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.GAUSSIAN_NB:
        return GaussianNaiveBayesModel(**cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.MULTINOMIAL_NB:
        return MultinomialNaiveBayesModel(**cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.AODE:
        return AODEModel(**cfg.get("model_kwargs", {}))
    elif model_type in {MacModelType.BAYESIAN_NETWORK, MacModelType.BBN, MacModelType.BN}:
        return BayesianNetworkModel(**cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.OLSR:
        return OLSRModel(**cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.STEPWISE:
        return StepwiseRegressionModel(**cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.MARS:
        return MARSModel(**cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.LOESS:
        return LOESSModel(**cfg.get("model_kwargs", {}))
    elif model_type == MacModelType.LWL:
        return LWLModel(
            task=cfg.get("task", "regression"),
            n_neighbors=cfg.get("n_neighbors", 5),
            **cfg.get("model_kwargs", {}),
        )

    return None
