# Streamlit Image Classification - 7 Types

## Files
Place these files in the same folder:

```text
classification_web/
├─ app.py
├─ best_efficientnet_b0.pth
└─ requirements.txt
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The browser should open automatically. If it does not, open the Local URL shown in the terminal.

## Output
After prediction, the app provides:

- `classified_images.zip`
  - `type_0/`
  - `type_1/`
  - `type_2/`
  - `type_3/`
  - `type_4/`
  - `type_5/`
  - `type_6/`
  - `prediction_results.csv`
  - `prediction_results.xlsx`
- A separate CSV download
- A separate Excel download

CSV columns:

```text
image,predicted,confidence
```

`confidence` is a percentage from 0 to 100.

## Folder upload
The project pins Streamlit 1.64+ because the app uses:

```python
accept_multiple_files="directory"
```

for direct directory uploads.
