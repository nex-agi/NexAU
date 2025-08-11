Add Langfuse Environment Variables to .env file

```
LANGFUSE_SECRET_KEY=sk-lf-xxxx
LANGFUSE_PUBLIC_KEY=pk-lf-xxxx
LANGFUSE_HOST="http://***REMOVED***/"
```

First upload the dataset by
```bash
dotenv run uv run examples/prompt_hacking_expt_langfuse/upload_dataset.py
```

Then run the evaluation by
```bash
dotenv run uv run examples/prompt_hacking_expt_langfuse/run_eval.py
```