# Method

This project studies recovery from manipulation failures caused by visual, physical, and task-level anomalies.

The core loop is:

1. Execute a nominal manipulation policy.
2. Estimate anomaly or failure risk from observations, critic values, and optional VLM judgments.
3. Trigger a recovery policy when risk exceeds a threshold.
4. Evaluate recovery success, task completion, and intervention cost.

## Method Families

- Rule-based recovery for transparent baselines.
- Actor-critic recovery using learned risk and value estimates.
- VLM/VLA recovery using language-grounded visual failure diagnosis.
- Hybrid recovery that combines the above signals.
