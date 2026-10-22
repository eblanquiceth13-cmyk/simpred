# SIMPRED — Tablero de mantenimiento predictivo

Tablero web que estima el riesgo de falla de una máquina, explica la predicción con SHAP y registra la decisión del técnico.

## Archivos

| Archivo | Qué es |
|---|---|
| `app.py` | Tablero web (Streamlit) |
| `simpred_modelo_xgb.json` | Modelo XGBoost entrenado en el notebook (Parte 1), en formato portable |
| `simpred_config.json` | Variables, etiquetas y umbrales del modelo |
| `ficha_modelo.json` | Ficha técnica del modelo: datos, métricas, limitaciones |
| `lecturas_ejemplo.csv` | Lecturas de 8 equipos para el panel |
| `requirements.txt` | Librerías (mismas versiones con que se entrenó el modelo) |
| `registro_decisiones.csv` | Se crea solo al guardar la primera decisión |

## Ejecutar en tu PC

Requiere Python 3.10 a 3.13.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre en el navegador en `http://localhost:8501`.

## Publicar en internet (para el enlace de la entrega)

1. Sube esta carpeta a un repositorio de GitHub.
2. Entra a https://share.streamlit.io con tu cuenta de GitHub → **Create app**.
3. Elige el repositorio, rama `main` y archivo `app.py`. En *Advanced settings* elige Python 3.12.
4. **Deploy**. Te da un enlace público.

En la versión publicada el registro de decisiones se borra cuando la app se reinicia. Usa el botón **Descargar registro** para conservarlo.

## Dónde se aplica el principio de transparencia

Busca `# 🔎 TRANSPARENCIA` en `app.py`:
- Explicación de cada predicción en lenguaje sencillo (SHAP).
- Aviso de limitaciones en cada resultado y en la barra lateral.
- Advertencias cuando una lectura está fuera de lo que el modelo conoce.
- Decisión final del técnico y registro auditable.
- Ficha del modelo adaptada a técnicos, jefes y desarrolladores.
