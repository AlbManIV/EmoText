# EmoText — Detección de emociones en textos (ES)

Proyecto escolar **end-to-end** listo para correr: baseline clásico y **fine-tuning** de BETO (BERT en español) + **demo web con Gradio**.

## Estructura
```
EmoText_EndToEnd/
├─ data/
│  └─ sample_emotions_es.csv        # Dataset de ejemplo (6 clases)
├─ scripts/
│  ├─ baseline_logreg.py            # Baseline TF-IDF + Regresión Logística
│  ├─ finetune_beto.py              # Fine-tuning de BETO con HuggingFace
│  └─ demo_gradio.py                # Demo web (Gradio) para probar el modelo
├─ outputs/                         # Aquí se guardan modelos entrenados
├─ requirements.txt
└─ README.md
```

## Requisitos
- Recomendado: **Google Colab** con GPU activada (Runtime → Change runtime type → T4/A100).
- Python 3.9+

Instala dependencias:
```bash
pip install -r requirements.txt
```

## 1) Baseline (TF-IDF + Regresión Logística)
Entrena y evalúa:
```bash
python scripts/baseline_logreg.py --csv data/sample_emotions_es.csv --text_col text --label_col label --test_size 0.2
```
Guarda el modelo en `outputs/baseline_logreg/` y muestra Accuracy / F1.

## 2) Fine-tuning BETO
Entrena BETO con HuggingFace `Trainer`:
```bash
python scripts/finetune_beto.py --csv data/sample_emotions_es.csv --text_col text --label_col label --epochs 3 --batch_size 16 --out_dir outputs/beto-emotext
```
**Nota:** El dataset de ejemplo es pequeño y sirve solo para la demo. Reemplázalo por uno más grande si desean métricas altas.

## 3) Demo con Gradio
Ejecuta una interfaz web local:
```bash
python scripts/demo_gradio.py --model_dir outputs/beto-emotext
```
Abre el enlace que imprime Gradio, escribe un texto en español y verás la emoción y probabilidades.

## Cambiar a un dataset real
- Reemplaza `data/sample_emotions_es.csv` por tu CSV con columnas `text` y `label`.
- Clases soportadas (libres): `alegria, tristeza, enojo, miedo, sorpresa, neutral` (puedes cambiarlas; los scripts las detectan automáticamente).
- Asegúrate de tener **>500 ejemplos por clase** para resultados sólidos.

## Ética y privacidad
- No incluir PII (nombres, teléfonos, etc.).
- Uso académico; no desplegar en producción sin consentimiento y evaluación de sesgos.
