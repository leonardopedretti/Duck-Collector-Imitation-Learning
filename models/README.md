# Models

`best_model.keras` is the saved neural-network policy used by the runnable demo.
It expects a four-dimensional input vector:

1. largest yellow blob centroid x-coordinate
2. largest yellow blob centroid y-coordinate
3. left/right yellow-pixel skew
4. yellow-pixel fraction

Architecture:

- batch normalization
- dense layer, 32 ReLU units, L2 regularization
- dense layer, 16 ReLU units, L2 regularization
- dense softmax output for the three actions
