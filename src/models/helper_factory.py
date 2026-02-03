"""
Factory for Classical and Supplemental ML Models.
"""

from typing import TYPE_CHECKING, Any, ClassVar

from ..enums.models import HelperModelType

if TYPE_CHECKING:
    from .mac.base import ClassicalModel

from .helper.association_rule import (
    AprioriModel,
    EclatModel,
    FPGrowthModel,
)
from .helper.clustering import (
    DBSCANModel,
    EMModel,
    GMMModel,
    HierarchicalClusteringModel,
    KMeansModel,
    KMediansModel,
)
from .helper.dim_reduction import (
    FDAModel,
    LDAModel,
    MDAModel,
    MDSModel,
    PCAModel,
    PCRModel,
    PLSRModel,
    ProjectionPursuitModel,
    QDAModel,
    SammonMappingModel,
    TSNEModel,
    UMAPModel,
)


class HelperModelFactory:
    """
    Factory class to create instances of supplemental ML models.
    Supports Clustering, Dimensionality Reduction, and Association Rule Learning.
    """

    _MODELS: ClassVar[dict[str, type]] = {
        # Clustering
        HelperModelType.KMEANS.value: KMeansModel,
        HelperModelType.HIERARCHICAL.value: HierarchicalClusteringModel,
        HelperModelType.DBSCAN.value: DBSCANModel,
        HelperModelType.GMM.value: GMMModel,
        HelperModelType.EM.value: EMModel,
        HelperModelType.KMEDIANS.value: KMediansModel,
        # Dimensionality Reduction
        HelperModelType.PCA.value: PCAModel,
        HelperModelType.TSNE.value: TSNEModel,
        HelperModelType.LDA.value: LDAModel,
        HelperModelType.PCR.value: PCRModel,
        HelperModelType.PLSR.value: PLSRModel,
        HelperModelType.MDS.value: MDSModel,
        HelperModelType.SAMMON.value: SammonMappingModel,
        HelperModelType.PP.value: ProjectionPursuitModel,
        HelperModelType.MDA.value: MDAModel,
        HelperModelType.QDA.value: QDAModel,
        HelperModelType.FDA.value: FDAModel,
        HelperModelType.UMAP.value: UMAPModel,
        # Association Rule Learning
        HelperModelType.APRIORI.value: AprioriModel,
        HelperModelType.FPGROWTH.value: FPGrowthModel,
        HelperModelType.ECLAT.value: EclatModel,
    }

    @classmethod
    def create_model(cls, model_name: str, **kwargs: Any) -> "ClassicalModel":
        """
        Create a model instance based on the provided name.

        Args:
            model_name: Name of the algorithm (e.g., 'kmeans', 'pca', 'apriori').
            **kwargs: Hyperparameters for the model.

        Returns:
            An instance of ClassicalModel.
        """
        try:
            model_type = HelperModelType(model_name.lower())
        except ValueError:
            raise ValueError(f"Unknown model type: {model_name}. Available: {[m.value for m in HelperModelType]}")

        model_class = cls._MODELS.get(model_type.value)
        if model_class is None:
            # Should not happen if _MODELS covers all Enum values
            raise ValueError(f"Model {model_name} implementation not found.")

        from typing import cast

        return cast("ClassicalModel", model_class(**kwargs))

    @classmethod
    def list_available_models(cls) -> list[str]:
        """Returns a list of all available model names."""
        return list(cls._MODELS.keys())
