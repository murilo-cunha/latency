# Latency

Based off example from [modal-example](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/stable_lm)
## Modal

```bash
curl $MODEL_APP_ENDPOINT \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Generate a list of 20 great names for sentient cheesecakes that teach SQL",
    "stream": true,
    "max_tokens": 64
  }'
```
