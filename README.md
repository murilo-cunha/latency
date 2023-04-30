# Latency

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
