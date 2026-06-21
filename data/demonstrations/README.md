# Demonstrations

Pickle files in this directory follow the project data format:

- `observations`: RGB image sequence with shape `(N, 60, 80, 3)`
- `actions`: integer action labels, where `0=left`, `1=right`, `2=forward`
- `time`: room id for each frame

The default training script loads all demonstration pickle files in this directory:

```bash
python scripts/train_model.py --pattern "*.pickle"
```

To train on a selected subset, provide a narrower glob pattern:

```bash
python scripts/train_model.py --pattern "demonstration-*.pickle"
```
