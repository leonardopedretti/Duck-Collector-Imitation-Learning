# Project Summary

This project solves a vision-based object-collection task with imitation learning. The
robot receives only the current RGB camera observation and must output one of
three actions: turn left, turn right, or move forward.

## Final Approach

The final policy is deliberately compact:

- detect yellow pixels with RGB thresholds tuned for the pickup objects;
- identify the largest connected yellow component;
- convert the image into four interpretable features: centroid x, centroid y,
  left/right skew, and yellow fraction;
- classify the action with a small Keras MLP trained on human demonstrations.

This design avoids feeding raw images into a large model. It keeps inference
fast, makes the behavior easier to explain, and generalizes across the visual
changes between rooms because the features focus on the target object rather
than the room texture.

## Model Selection

The development process compared a random forest against a neural network. The random
forest is fast and stable but tends to over-predict the majority class
`move_forward`. The neural network is more sensitive to minority actions and was
selected for the final policy after class-weight tuning.

The final hyperparameters are:

- hidden layers: 32 and 16 units
- activation: ReLU
- output: 3-way softmax
- optimizer: Adam, learning rate `1e-3`
- L2 regularization: `1e-4`
- batch size: 128
- early stopping patience: 10
- softened class weighting: `alpha=0.5`

## Generalization

Rooms 0, 1, and 2 are used for training and validation. Room 3 is treated as the
held-out deployment room. The strongest result came from training on the full
multi-room dataset with softened class weights: enough emphasis on turn actions
to avoid a forward-only policy, but not enough to trigger unstable left/right
oscillation.
