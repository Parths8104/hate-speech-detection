# Data

## Sample data (`sample.csv`)

A tiny, fully synthetic set of clean examples (no slurs) across the three classes
(`neither`, `offensive`, `hate`). It exists only to smoke-test the pipeline end to
end in seconds. Do not read anything into metrics produced on it; it is far too
small to be meaningful.

```bash
python -m hsd.train --config config/sample.yaml
```

## Real dataset

This project is built for the public **Davidson et al. (2017) "Hate Speech and
Offensive Language"** dataset, a labeled corpus of ~24,800 tweets across three
classes: `hate`, `offensive`, and `neither`.

- Source: https://github.com/t-davidson/hate-speech-and-offensive-language
- The raw file `labeled_data.csv` has a numeric `class` column (0 = hate,
  1 = offensive, 2 = neither) and a `tweet` column.

To use it, map the columns into the format this project expects (`text`, `label`)
and save as `data/dataset.csv`:

```python
import pandas as pd

df = pd.read_csv("labeled_data.csv")
mapping = {0: "hate", 1: "offensive", 2: "neither"}
out = pd.DataFrame({"text": df["tweet"], "label": df["class"].map(mapping)})
out.to_csv("data/dataset.csv", index=False)
```

Then train on the full dataset:

```bash
python -m hsd.train --config config/config.yaml
```

## Bring your own data

Any CSV works as long as it has a text column and a label column. Point
`config.yaml` at it and set `text_column` / `label_column` accordingly. The
pipeline is label-agnostic, so binary (`toxic` / `clean`) or multilingual,
code-mixed corpora work too; character n-grams make it robust to the spelling
variation common in code-mixed text.

> Dataset files are git-ignored on purpose to keep the repo light and to avoid
> redistributing third-party data. Only `sample.csv` is committed.
