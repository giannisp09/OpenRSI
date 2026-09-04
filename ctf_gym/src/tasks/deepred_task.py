import os
import subprocess
import tempfile
import uuid
import time
from typing import Dict, Optional
from ...src.base_task import BaseCTFTask

class DeepRedTask(BaseCTFTask):
    """
    Task implementation for DeepRed VM-based networked CTFs.
    Spins up a target service container and an attacker container in the same docker network.
    """
    
    def prepare(self, **task_args: Optional[Dict]) -> Dict:
        """
        Build the target and attacker containers, set up network.
        """
        self.target_image = f"ctfgym_{self.benchmark_name}_{self.task_id.lower()}_target"
        self.target_dir = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "deepred", self.task_id)
        
        # Build target container
        subprocess.run(
            ["docker", "build", "-t", self.target_image, self.target_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Build attacker container
        self.attacker_image = f"ctfgym_{self.benchmark_name}_{self.task_id.lower()}_attacker"
        subprocess.run([
            "docker", "build", "-t", self.attacker_image, "-"
        ], input=b"FROM ubuntu:24.04\nRUN apt-get update && apt-get install -y curl nmap netcat-openbsd\n", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self.network_name = f"deepred_net_{uuid.uuid4().hex[:8]}"
        subprocess.run(["docker", "network", "create", self.network_name], check=True, stdout=subprocess.DEVNULL)
        return {}

    def setup_sandbox(self):
        pass

    def run_in_sandbox(self, code: str) -> str:
        """
        Run the candidate shell script in the attacker container.
        """
        target_name = f"deepred_target_{uuid.uuid4().hex[:8]}"
        
        # Start target
        subprocess.run([
            "docker", "run", "-d", "--rm",
            "--name", target_name,
            "--network", self.network_name,
            self.target_image
        ], check=True, stdout=subprocess.DEVNULL)
        
        time.sleep(2) # Wait for service to start
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(code)
            temp_path = f.name
            
        try:
            attacker_name = f"deepred_attacker_{uuid.uuid4().hex[:8]}"
            # Attacker container
            result = subprocess.run([
                "docker", "run", "--rm",
                "--name", attacker_name,
                "--network", self.network_name,
                "-v", f"{temp_path}:/exploit.sh:ro",
                "-e", f"TARGET_HOST={target_name}",
                self.attacker_image,
                "bash", "/exploit.sh"
            ], capture_output=True, text=True, timeout=self.timeout)
            return result.stdout + "\n" + result.stderr
        except subprocess.TimeoutExpired:
            return "Execution timed out."
        except Exception as e:
            return str(e)
        finally:
            os.remove(temp_path)
            # Stop target container
            subprocess.run(["docker", "stop", target_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
    def close(self, state: Dict) -> None:
        """Clean up the network after task is done."""
        if hasattr(self, 'network_name'):
            subprocess.run(["docker", "network", "rm", self.network_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
