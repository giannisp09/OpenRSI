from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

from .eval_exploitbench import evaluate_exploitbench
from .eval_deepred import evaluate_deepred

app = FastAPI(title="CTFGym Verifier Router")
logger = logging.getLogger(__name__)

class EvaluateRequest(BaseModel):
    benchmark: str
    task_id: str
    logs: str

@app.post("/evaluate/exploitbench")
def handle_exploitbench(req: EvaluateRequest):
    try:
        return evaluate_exploitbench(req.task_id, req.logs)
    except Exception as e:
        logger.error(f"Error evaluating exploitbench: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/deepred")
def handle_deepred(req: EvaluateRequest):
    try:
        return evaluate_deepred(req.task_id, req.logs)
    except Exception as e:
        logger.error(f"Error evaluating deepred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
