import os
import subprocess
import requests
import time
import json
from typing import Any, Dict, Optional, Tuple

from dojo.core.tasks.base import Task
from dojo.config_dataclasses.task.base import TaskConfig


class BaseCTFTask(Task):
    """Generic CTF Task that all benchmark-specific tasks inherit from."""
    
    def __init__(self, cfg: TaskConfig) -> None:
        super().__init__(cfg)
        self.benchmark_name = cfg.get("benchmark_name", "unknown")
        self.task_id = cfg.get("task_id", "unknown")
        self.timeout = cfg.get("timeout", 60)
        self.router_url = cfg.get("router_url", "http://localhost:8000")

    def prepare(self, **task_args: Optional[Dict]) -> Dict:
        """Prepare the task (implemented by subclass)"""
        return {}

    def setup_sandbox(self) -> Any:
        """Setup the docker sandbox (implemented by subclass)"""
        raise NotImplementedError

    def run_in_sandbox(self, code: str) -> str:
        """Run the candidate code in sandbox and return logs"""
        raise NotImplementedError
        
    def step_task(self, state: Dict, code: str) -> Tuple[Dict, Dict]:
        # 1. Run the candidate code in the sandbox (sandbox is typically setup in prepare or lazily)
        execution_logs = self.run_in_sandbox(code)
        
        # 2. Send to the Verifier Router
        verifier_payload = self.call_verifier_service(
            benchmark=self.benchmark_name, 
            task_id=self.task_id, 
            logs=execution_logs
        )
        
        # 3. Standardize the feedback string for the LLM
        feedback = self.construct_feedback_string(verifier_payload)
        
        return state, {"score": verifier_payload.get("score", 0), "feedback": feedback, "achieved": verifier_payload.get("achieved", []), "weakest_component": verifier_payload.get("weakest_component", "")}

    def call_verifier_service(self, benchmark: str, task_id: str, logs: str) -> dict:
        try:
            payload = {
                "benchmark": benchmark,
                "task_id": task_id,
                "logs": logs
            }
            response = requests.post(f"{self.router_url}/evaluate/{benchmark}", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to call verifier service: {e}")
            return {
                "score": 0.0,
                "achieved": "none",
                "weakest_component": "verifier call failed",
                "traceback": str(e)
            }

    def construct_feedback_string(self, payload: dict) -> str:
        """
        Generic template that prevents the 'partial-credit trap'.
        """
        return f"""
        Score: {payload.get('score', 0)}
        Achieved Milestones: {payload.get('achieved', 'None')}
        Failed At: {payload.get('weakest_component', 'Unknown')}
        
        CRITICAL PRESERVATION CONSTRAINT: You must maintain the logic that achieved 
        {payload.get('achieved', 'None')}. Do not break this while attempting to fix {payload.get('weakest_component', 'Unknown')}.
        
        Execution Logs/Traceback:
        {payload.get('traceback', 'None')}
        """
        
    def evaluate_fitness(
        self,
        solution: Optional[Dict] = None,
        state: Optional[Dict] = None,
        interpreter: Optional[Dict] = None,
        aux_info: Dict[str, Any] = None,
    ) -> Any:
        pass

    def close(self, state: Dict) -> None:
        pass
