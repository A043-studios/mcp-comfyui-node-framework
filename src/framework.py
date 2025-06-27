"""
MCP Multi-Agent Framework Core
Main framework class for orchestrating agent execution
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from .utils import create_directory, save_json, load_json
except ImportError:
    from utils import create_directory, save_json, load_json


class MCPFramework:
    """
    Core framework class for managing multi-agent ComfyUI node generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MCP Framework
        
        Args:
            config: Framework configuration dictionary
        """
        self.config = config
        self.start_time = time.time()
        self.execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup output directories
        self.output_dir = Path(config["output_directory"])
        self.logs_dir = self.output_dir / "logs"
        self.artifacts_dir = self.output_dir / "artifacts"
        
        # Create directories
        create_directory(self.output_dir)
        create_directory(self.logs_dir)
        create_directory(self.artifacts_dir)
        
        # Initialize execution tracking
        self.execution_log = {
            "execution_id": self.execution_id,
            "start_time": datetime.now().isoformat(),
            "config": config,
            "agents": [],
            "status": "initialized"
        }
    
    def log_agent_start(self, agent_name: str, agent_config: Dict[str, Any]):
        """Log the start of an agent execution"""
        agent_log = {
            "name": agent_name,
            "start_time": datetime.now().isoformat(),
            "config": agent_config,
            "status": "running"
        }
        self.execution_log["agents"].append(agent_log)
        
        # Save intermediate log
        self._save_execution_log()
    
    def log_agent_completion(self, agent_name: str, result: Dict[str, Any]):
        """Log the completion of an agent execution"""
        # Find the agent in the log
        for agent_log in self.execution_log["agents"]:
            if agent_log["name"] == agent_name and agent_log["status"] == "running":
                agent_log["end_time"] = datetime.now().isoformat()
                agent_log["status"] = result.get("status", "completed")
                agent_log["result"] = result
                break
        
        # Save intermediate log
        self._save_execution_log()
    
    def log_agent_error(self, agent_name: str, error: str):
        """Log an agent execution error"""
        # Find the agent in the log
        for agent_log in self.execution_log["agents"]:
            if agent_log["name"] == agent_name and agent_log["status"] == "running":
                agent_log["end_time"] = datetime.now().isoformat()
                agent_log["status"] = "error"
                agent_log["error"] = error
                break
        
        # Update overall status
        self.execution_log["status"] = "error"
        
        # Save intermediate log
        self._save_execution_log()
    
    def generate_report(self, context: Dict[str, Any]):
        """
        Generate comprehensive execution report
        
        Args:
            context: Execution context with results from all agents
        """
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        # Update execution log
        self.execution_log["end_time"] = datetime.now().isoformat()
        self.execution_log["execution_time_seconds"] = execution_time
        self.execution_log["status"] = "completed"
        self.execution_log["context"] = context
        
        # Generate summary statistics
        summary = self._generate_summary(context, execution_time)
        self.execution_log["summary"] = summary
        
        # Save final execution log
        self._save_execution_log()
        
        # Generate detailed report
        self._generate_detailed_report(context, summary)
        
        # Generate metrics report
        self._generate_metrics_report(context)
    
    def _save_execution_log(self):
        """Save execution log to file"""
        log_file = self.logs_dir / f"execution_{self.execution_id}.json"
        save_json(self.execution_log, log_file)
    
    def _generate_summary(self, context: Dict[str, Any], execution_time: float) -> Dict[str, Any]:
        """Generate execution summary statistics"""
        agents_completed = context.get("agents_completed", [])
        artifacts = context.get("artifacts", {})
        metrics = context.get("metrics", {})
        
        # Count generated files
        total_files = 0
        for agent_artifacts in artifacts.values():
            if isinstance(agent_artifacts, dict):
                total_files += len(agent_artifacts.get("files", []))
        
        # Calculate success rate
        total_agents = len(self.execution_log["agents"])
        successful_agents = len([a for a in self.execution_log["agents"] if a.get("status") == "completed"])
        success_rate = (successful_agents / total_agents * 100) if total_agents > 0 else 0
        
        summary = {
            "execution_time_minutes": round(execution_time / 60, 2),
            "total_agents": total_agents,
            "successful_agents": successful_agents,
            "success_rate_percent": round(success_rate, 1),
            "total_files_generated": total_files,
            "output_directory": str(self.output_dir),
            "quality_level": context.get("quality_level", "unknown")
        }
        
        return summary
    
    def _generate_detailed_report(self, context: Dict[str, Any], summary: Dict[str, Any]):
        """Generate detailed markdown report"""
        report_content = f"""# MCP Framework Execution Report

## 📊 Execution Summary

- **Execution ID**: {self.execution_id}
- **Start Time**: {self.execution_log['start_time']}
- **End Time**: {self.execution_log.get('end_time', 'N/A')}
- **Total Time**: {summary['execution_time_minutes']} minutes
- **Success Rate**: {summary['success_rate_percent']}%

## 🎯 Configuration

- **Input Source**: {context.get('input_source', 'N/A')}
- **Input Type**: {context.get('input_type', 'N/A')}
- **Output Directory**: {summary['output_directory']}
- **Quality Level**: {summary['quality_level']}
- **Focus Areas**: {', '.join(context.get('focus_areas', []) or ['None'])}

## 🤖 Agent Execution Results

"""
        
        # Add agent results
        for agent_log in self.execution_log["agents"]:
            agent_name = agent_log["name"]
            status = agent_log.get("status", "unknown")
            status_emoji = "✅" if status == "completed" else "❌" if status == "error" else "🔄"
            
            report_content += f"""### {status_emoji} {agent_name}

- **Status**: {status}
- **Start Time**: {agent_log.get('start_time', 'N/A')}
- **End Time**: {agent_log.get('end_time', 'N/A')}
"""
            
            if "result" in agent_log:
                result = agent_log["result"]
                if "summary" in result:
                    report_content += f"- **Summary**: {result['summary']}\n"
                if "files_generated" in result:
                    report_content += f"- **Files Generated**: {result['files_generated']}\n"
            
            if "error" in agent_log:
                report_content += f"- **Error**: {agent_log['error']}\n"
            
            report_content += "\n"
        
        # Add artifacts summary
        artifacts = context.get("artifacts", {})
        if artifacts:
            report_content += "## 📁 Generated Artifacts\n\n"
            for agent_name, agent_artifacts in artifacts.items():
                if isinstance(agent_artifacts, dict) and "files" in agent_artifacts:
                    files = agent_artifacts["files"]
                    report_content += f"### {agent_name.title()} Agent\n\n"
                    for file_info in files:
                        report_content += f"- `{file_info.get('path', 'unknown')}` ({file_info.get('type', 'unknown')})\n"
                    report_content += "\n"
        
        # Add metrics summary
        metrics = context.get("metrics", {})
        if metrics:
            report_content += "## 📈 Performance Metrics\n\n"
            for agent_name, agent_metrics in metrics.items():
                if isinstance(agent_metrics, dict):
                    report_content += f"### {agent_name.title()} Agent\n\n"
                    for metric_name, metric_value in agent_metrics.items():
                        report_content += f"- **{metric_name}**: {metric_value}\n"
                    report_content += "\n"
        
        # Add next steps
        report_content += """## 🚀 Next Steps

1. **Review Generated Nodes**: Check the generated ComfyUI nodes in the output directory
2. **Install Dependencies**: Run `pip install -r requirements.txt` to install required packages
3. **Test Integration**: Copy nodes to ComfyUI custom_nodes directory and test
4. **Documentation**: Review the generated documentation and examples
5. **Community Sharing**: Consider sharing successful nodes with the ComfyUI community

## 📞 Support

If you encounter any issues:

1. Check the execution logs in the `logs/` directory
2. Review the agent-specific error messages above
3. Consult the framework documentation
4. Open an issue on the GitHub repository

---

*Report generated by MCP Multi-Agent Framework*  
*Execution ID: {self.execution_id}*
"""
        
        # Save report
        report_file = self.output_dir / "execution_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
    
    def _generate_metrics_report(self, context: Dict[str, Any]):
        """Generate metrics report in JSON format"""
        metrics = context.get("metrics", {})
        
        metrics_report = {
            "execution_id": self.execution_id,
            "timestamp": datetime.now().isoformat(),
            "framework_metrics": {
                "total_execution_time": time.time() - self.start_time,
                "agents_executed": len(self.execution_log["agents"]),
                "success_rate": len([a for a in self.execution_log["agents"] if a.get("status") == "completed"]) / len(self.execution_log["agents"]) if self.execution_log["agents"] else 0
            },
            "agent_metrics": metrics
        }
        
        # Save metrics
        metrics_file = self.logs_dir / f"metrics_{self.execution_id}.json"
        save_json(metrics_report, metrics_file)
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        return {
            "execution_id": self.execution_id,
            "status": self.execution_log["status"],
            "agents_completed": len([a for a in self.execution_log["agents"] if a.get("status") == "completed"]),
            "total_agents": len(self.execution_log["agents"]),
            "execution_time": time.time() - self.start_time
        }
    
    def cleanup(self):
        """Cleanup framework resources"""
        # Save final state
        self._save_execution_log()
        
        # Any additional cleanup can be added here
        pass
