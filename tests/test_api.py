import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from medical_event_chains.api import create_app


def test_real_api_roundtrip():
    with TestClient(create_app()) as client:
        assert client.get('/api/health').json()["backend"]["name"]=="rule"
        payload=client.get('/api/example').json()
        response=client.post('/api/extract',json=payload)
        assert response.status_code==200
        assert len(response.json()["combined_chains"])==1
        assert client.get('/').status_code==200
        assert client.get('/app.js').headers['content-type'].startswith('text/javascript')
        assert client.post('/api/extract',json={"records":[]}).status_code==422
        assert client.post('/api/extract',content='x'*2_000_001).status_code==413
