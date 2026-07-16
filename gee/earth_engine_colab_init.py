"""Inicialização explícita do Google Earth Engine em notebooks Python/Colab.

O Earth Engine passou a exigir um projeto Google Cloud válido em muitas
inicializações interativas. Use este helper antes de chamar qualquer operação
`ee.*` em notebooks Python para evitar o erro:

    EEException: ee.Initialize: no project found. Call with project=

Exemplo no Colab:

    import ee
    from google.colab import userdata
    from gee.earth_engine_colab_init import initialize_earth_engine

    PROJECT_ID = userdata.get('EE_PROJECT_ID')  # ou informe a string diretamente
    initialize_earth_engine(PROJECT_ID)
"""

from __future__ import annotations

import os
from typing import Optional

import ee


def resolve_project_id(project_id: Optional[str] = None) -> str:
    """Resolve o ID do projeto GCP usado pelo Earth Engine.

    A ordem de prioridade é:
    1. argumento explícito `project_id`;
    2. variável de ambiente `EE_PROJECT_ID`;
    3. variável de ambiente `GOOGLE_CLOUD_PROJECT`;
    4. variável de ambiente `GCLOUD_PROJECT`.

    Raises:
        ValueError: se nenhum projeto válido for informado.
    """
    resolved = (
        project_id
        or os.environ.get("EE_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
    )

    if not resolved or not resolved.strip():
        raise ValueError(
            "Informe um projeto Google Cloud habilitado para Earth Engine. "
            "Exemplo: initialize_earth_engine('meu-projeto-gcp') ou defina "
            "a variável de ambiente EE_PROJECT_ID. No Colab, você também pode "
            "salvar o valor em Secrets como EE_PROJECT_ID e recuperá-lo com "
            "google.colab.userdata.get('EE_PROJECT_ID')."
        )

    return resolved.strip()


def initialize_earth_engine(project_id: Optional[str] = None) -> str:
    """Autentica e inicializa o Earth Engine com `project=` obrigatório.

    Returns:
        O ID do projeto usado na inicialização.
    """
    resolved_project_id = resolve_project_id(project_id)
    ee.Authenticate()
    ee.Initialize(project=resolved_project_id)
    print(f"Earth Engine inicializado com o projeto: {resolved_project_id}")
    return resolved_project_id
