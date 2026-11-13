from app.model.client import K8sModelClient


def test_model_client_reads_endpoint_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_ENDPOINT", "http://model.test:9000/generate_step")
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: object(),
    )

    client = K8sModelClient()

    assert client.endpoint_url == "http://model.test:9000/generate_step"