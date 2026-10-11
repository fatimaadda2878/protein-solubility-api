"""
server.py — HuggingFace Spaces entry point
=========================================
Ce fichier remplace app.py à la racine pour éviter le conflit avec le package app/.
Python résolvait `import app` vers app.py au lieu du package app/ → tous les tests cassaient.

Renommé de app.py → server.py pour résoudre le conflit d'import.
"""
import uvicorn
from app.main import app  # noqa: F401 — réexporté pour HF Spaces

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )
