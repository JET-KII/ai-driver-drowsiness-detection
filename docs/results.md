# Results and limitations

## Recorded training results

| Epoch | Training accuracy | Validation accuracy | Training loss | Validation loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 97.98% | 66.44% | 0.0503 | 4.5701 |
| 2 | 99.93% | 68.79% | 0.0025 | 6.7350 |
| 3 | 99.95% | 66.82% | 0.0021 | 4.2888 |
| 4 | 99.97% | 74.93% | 0.0011 | 3.2308 |
| 5 | 99.97% | 72.46% | 0.0011 | 6.2726 |
| 6 | 99.98% | 74.35% | 0.0010 | 8.5449 |
| 7 | 99.98% | 72.81% | 0.0008 | 9.4176 |
| 8 | 99.99% | 78.61% | 0.0007 | 5.0924 |

## Interpretation

The model fitted the training data rapidly but validation accuracy stayed below 80% and validation loss remained unstable. The report identifies this as overfitting, likely caused by near-duplicate video frames, a limited number of subjects, and insufficient variation in faces, poses, and lighting.

## Deployment limitations

- No broad field trial with a diverse driver population was completed.
- Event-level microsleep detection was not comprehensively evaluated.
- Reinforcement-learning personalization was proposed but not implemented.
- Steering-data fusion and haptic alerts were proposed but not integrated into the validated prototype.
- The impact of integer quantization on accuracy was not evaluated.
- The original dataset and trained model are not included in this repository.

## Recommended next steps

1. Add more drivers, lighting conditions, poses, glasses, and occlusions.
2. Split data by subject rather than randomly by adjacent frames.
3. Use transfer learning with a lightweight backbone such as MobileNetV2.
4. Add brightness, contrast, and occlusion augmentation.
5. Benchmark latency, CPU use, memory, and temperature on the Raspberry Pi.
6. Evaluate false alarms, missed detections, and event-level sensitivity in controlled trials.

