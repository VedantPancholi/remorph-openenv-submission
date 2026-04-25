# ReMorph Benchmark Report

- Trainer: `supervised_structured_policy`
- Seed: `42`
- Train scenarios: `10`
- Eval scenarios: `6`
- Training examples: `22`

## Eval Metrics

| Policy | Success Rate | Avg Raw Reward | Avg Capped Episode Return | Avg Steps |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.0000 | -19.6667 | 0.9956 | 4.5000 |
| replay | 0.0000 | -19.6667 | 0.9956 | 4.5000 |
| supervised | 1.0000 | 19.5000 | 1.0000 | 2.0000 |
| oracle | 1.0000 | 19.5000 | 1.0000 | 2.0000 |

## Model Config

```json
{
  "learner": "supervised_structured_policy",
  "route_strategy": "max_confidence_candidate",
  "payload_strategy": "expected_request_body",
  "auth_strategy": "required_headers",
  "abstain_strategy": "partition_gate",
  "average_route_confidence": 0.885,
  "payload_hint_coverage": 1.0,
  "observed_tenant_aliases": [
    "east",
    "north",
    "west"
  ],
  "counts": {
    "route_examples": 8,
    "payload_examples": 7,
    "auth_examples": 6,
    "abstain_examples": 1,
    "training_example_count": 22
  }
}
```