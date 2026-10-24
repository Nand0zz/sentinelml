import mlflow
from mlflow.tracking import MlflowClient

from sentinelml.data import clean
from sentinelml.synthetic import make_synthetic
from sentinelml.train import run_training


def test_end_to_end_training_registers_candidate(tmp_path):
    train = clean(make_synthetic(3000, seed=0))
    test = clean(make_synthetic(1200, seed=1, shift=0.2), drop_duplicates=False)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    table, best = run_training(
        train, test, tracking_uri=uri, model_name="test-ids", fast=True, out_dir=tmp_path
    )

    assert len(table) == 3
    assert table["val_fpr"].max() <= 0.06  # threshold respects the FPR budget (+ slack)
    assert (tmp_path / "reports" / "model_comparison.csv").exists()

    client = MlflowClient(tracking_uri=uri)
    mv = client.get_model_version_by_alias("test-ids", "candidate")
    assert str(mv.version) == str(best["registered_version"])
    assert "threshold" in mv.tags

    mlflow.set_tracking_uri(uri)
    model = mlflow.sklearn.load_model("models:/test-ids@candidate")
    assert model.predict_proba(test.drop(columns=["label"]).head(2)).shape == (2, 2)
