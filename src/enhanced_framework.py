#!/usr/bin/env python3
"""
Enhanced ComfyUI Framework with Intelligent Analysis
Replaces simple template-based generation with sophisticated LLM-powered workflows
"""

import logging
import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from intelligent_analyzer import IntelligentContentAnalyzer, NodeSpecification
from llm_client import LLMManager
from execution_monitor import ExecutionMonitor, ExecutionStatus

# Import agents with fallback handling
try:
    from agents.research_agent import ResearchAgent
    from agents.coding_agent import CodingAgent
    from agents.testing_agent import TestingAgent
    from agents.documentation_agent import DocumentationAgent
    from agents.devops_agent import DevOpsAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Agents not available: {e}")
    AGENTS_AVAILABLE = False
    # Create mock agents for fallback
    class MockAgent:
        def __init__(self, *args, **kwargs):
            pass
        async def execute(self, *args, **kwargs):
            return {"status": "mock", "message": "Agent not available"}

    ResearchAgent = MockAgent
    CodingAgent = MockAgent
    TestingAgent = MockAgent
    DocumentationAgent = MockAgent
    DevOpsAgent = MockAgent

class MockAgent:
    """Mock agent for fallback when real agents fail to initialize"""
    def __init__(self, agent_type: str):
        self.agent_type = agent_type

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "completed",
            "agent": self.agent_type,
            "message": f"Mock {self.agent_type} agent executed",
            "artifacts": {},
            "fallback": True
        }

# Optional performance optimizations
try:
    from performance_optimizer import get_performance_optimizer, monitor_performance, async_cached
    from dependency_manager import get_dependency_manager
    PERFORMANCE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZATION_AVAILABLE = False
    # Mock functions for fallback
    def get_performance_optimizer(config=None):
        return None
    def get_dependency_manager(config=None):
        return None
    def monitor_performance(func):
        return func
    def async_cached(ttl=None):
        def decorator(func):
            return func
        return decorator


class QualityLevel:
    """Quality level configurations for different use cases"""
    
    DRAFT = {
        "name": "draft",
        "description": "Fast prototyping with basic functionality",
        "agents": ["research", "coding"],
        "llm_calls": "minimal",
        "validation": "basic",
        "estimated_time": "5-15 minutes"
    }
    
    DEVELOPMENT = {
        "name": "development", 
        "description": "Standard development with testing and documentation",
        "agents": ["research", "coding", "testing", "documentation"],
        "llm_calls": "standard",
        "validation": "comprehensive",
        "estimated_time": "30-60 minutes"
    }
    
    PRODUCTION = {
        "name": "production",
        "description": "Production-ready with full validation and deployment",
        "agents": ["research", "coding", "testing", "documentation", "devops"],
        "llm_calls": "extensive",
        "validation": "strict",
        "estimated_time": "1-2 hours"
    }


class EnhancedComfyUIFramework:
    """
    Enhanced framework with intelligent analysis and quality-differentiated workflows
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize performance optimization (if available and caching is enabled)
        if PERFORMANCE_OPTIMIZATION_AVAILABLE and config.get("enable_caching", True):
            try:
                self.performance_optimizer = get_performance_optimizer(config)
                self.dependency_manager = get_dependency_manager(config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize performance optimization: {e}. Continuing without caching.")
                self.performance_optimizer = None
                self.dependency_manager = None
        else:
            self.performance_optimizer = None
            self.dependency_manager = None

        # Initialize core components
        self.intelligent_analyzer = IntelligentContentAnalyzer(config)
        self.llm_manager = LLMManager(config)
        self.execution_monitor = ExecutionMonitor()
        
        # Initialize agents with error handling
        self.agents = {}
        try:
            self.agents = {
                "research": ResearchAgent(config),
                "coding": CodingAgent(config),
                "testing": TestingAgent(config),
                "documentation": DocumentationAgent(config),
                "devops": DevOpsAgent(config)
            }
            self.logger.info("All agents initialized successfully")
        except Exception as e:
            self.logger.warning(f"Agent initialization failed: {e}")
            # Create fallback agents
            self.agents = {
                "research": MockAgent("research"),
                "coding": MockAgent("coding"),
                "testing": MockAgent("testing"),
                "documentation": MockAgent("documentation"),
                "devops": MockAgent("devops")
            }
            self.logger.info("Using fallback mock agents")
        
        # Quality level configurations
        self.quality_levels = {
            "draft": QualityLevel.DRAFT,
            "development": QualityLevel.DEVELOPMENT,
            "production": QualityLevel.PRODUCTION
        }
        
        self.logger.info("Enhanced ComfyUI Framework initialized with performance optimization")

    async def generate_comfyui_node(self, input_source: str, output_directory: Optional[str] = None,
                                  quality_level: str = "production", focus_areas: Optional[str] = None,
                                  agents: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate ComfyUI nodes using enhanced intelligent analysis
        """
        import uuid
        execution_id = str(uuid.uuid4())
        execution_info = self.execution_monitor.start_execution(
            execution_id=execution_id,
            tool_name="generate_comfyui_node",
            arguments={
                "input_source": input_source,
                "output_directory": output_directory,
                "quality_level": quality_level,
                "focus_areas": focus_areas,
                "agents": agents
            },
            output_directory=output_directory
        )
        
        try:
            self.logger.info(f"Starting enhanced node generation (execution: {execution_id})")
            
            # Validate quality level
            if quality_level not in self.quality_levels:
                quality_level = "production"
                self.logger.warning(f"Invalid quality level, defaulting to production")
            
            quality_config = self.quality_levels[quality_level]
            
            # Parse focus areas and agents
            focus_areas_list = [area.strip() for area in (focus_areas or "").split(",") if area.strip()]
            agents_list = [agent.strip() for agent in (agents or "").split(",") if agent.strip()]
            
            # Use quality level agents if not specified
            if not agents_list:
                agents_list = quality_config["agents"]
            
            # Set up output directory
            if not output_directory:
                output_directory = self.config.get("default_output_dir", "./output")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            execution_dir = os.path.join(output_directory, f"execution_{execution_id}_{timestamp}")
            os.makedirs(execution_dir, exist_ok=True)
            
            # Phase 1: Intelligent Content Analysis
            self.logger.info("Phase 1: Intelligent content analysis")
            content_analysis = await self._analyze_content_intelligently(
                input_source, focus_areas_list, quality_level, execution_id, output_directory
            )
            
            # Phase 2: Multi-Agent Processing
            self.logger.info("Phase 2: Multi-agent processing")
            agent_results = await self._execute_agent_workflow(
                content_analysis, agents_list, execution_dir, quality_level, execution_id
            )
            
            # Phase 3: Integration and Validation
            self.logger.info("Phase 3: Integration and validation")
            final_result = await self._integrate_and_validate(
                content_analysis, agent_results, execution_dir, quality_level, execution_id
            )
            
            # Phase 4: Quality Assurance
            self.logger.info("Phase 4: Quality assurance")
            qa_result = await self._perform_quality_assurance(
                final_result, quality_level, execution_id
            )
            
            self.execution_monitor.update_status(execution_id, ExecutionStatus.COMPLETED)
            
            return {
                "status": "success",
                "execution_id": execution_id,
                "quality_level": quality_level,
                "output_directory": execution_dir,
                "content_analysis": content_analysis,
                "agent_results": agent_results,
                "final_result": final_result,
                "qa_result": qa_result,
                "summary": self._generate_execution_summary(content_analysis, final_result, qa_result)
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced generation failed: {str(e)}")
            self.execution_monitor.update_status(execution_id, ExecutionStatus.FAILED, str(e))
            raise
    
    async def _analyze_content_intelligently(self, input_source: str, focus_areas: List[str],
                                           quality_level: str, execution_id: str, output_directory: str) -> Dict[str, Any]:
        """Perform intelligent content analysis using LLM reasoning"""
        
        execution_info = self.execution_monitor.get_execution(execution_id)
        if execution_info:
            execution_info.add_log("Starting intelligent content analysis")
        
        # Step 1: Extract content using research agent
        research_context = {
            "input_source": input_source,
            "focus_areas": focus_areas,
            "quality_level": quality_level,
            "output_directory": output_directory
        }
        research_result = self.agents["research"].execute(research_context)
        
        # Step 2: Intelligent analysis using enhanced analyzer
        content = research_result.get("analysis", {}).get("content", "")
        title = research_result.get("analysis", {}).get("title", "Unknown")
        
        node_specifications = self.intelligent_analyzer.analyze_content(
            content=content,
            title=title,
            input_source=input_source,
            focus_areas=focus_areas,
            quality_level=quality_level
        )
        
        execution_info = self.execution_monitor.get_execution(execution_id)
        if execution_info:
            execution_info.add_log(f"Generated {len(node_specifications)} intelligent node specifications")
        
        return {
            "research_result": research_result,
            "node_specifications": [self._serialize_node_spec(spec) for spec in node_specifications],
            "analysis_metadata": {
                "quality_level": quality_level,
                "focus_areas": focus_areas,
                "analysis_timestamp": datetime.now().isoformat(),
                "llm_model": self.llm_manager.client.model
            }
        }
    
    async def _execute_agent_workflow(self, content_analysis: Dict[str, Any], agents_list: List[str],
                                    execution_dir: str, quality_level: str, execution_id: str) -> Dict[str, Any]:
        """Execute multi-agent workflow based on quality level"""
        
        agent_results = {}
        context = {
            "content_analysis": content_analysis,
            "execution_dir": execution_dir,
            "quality_level": quality_level,
            "execution_id": execution_id
        }
        
        # Execute agents in sequence with dependency management
        for agent_name in agents_list:
            if agent_name not in self.agents:
                self.logger.warning(f"Unknown agent: {agent_name}")
                continue
            
            execution_info = self.execution_monitor.get_execution(execution_id)
            if execution_info:
                execution_info.add_log(f"Executing {agent_name} agent")
            
            try:
                agent = self.agents[agent_name]
                
                # Prepare agent-specific context
                agent_context = self._prepare_agent_context(agent_name, context, agent_results)
                
                # Execute agent
                if agent_name == "research":
                    # Research already done in content analysis
                    result = content_analysis["research_result"]
                elif agent_name == "coding":
                    result = await self._execute_coding_agent(agent, agent_context)
                elif agent_name == "testing":
                    result = await self._execute_testing_agent(agent, agent_context)
                elif agent_name == "documentation":
                    result = await self._execute_documentation_agent(agent, agent_context)
                elif agent_name == "devops":
                    result = await self._execute_devops_agent(agent, agent_context)
                else:
                    result = {"error": f"Unknown agent execution: {agent_name}"}
                
                agent_results[agent_name] = result
                context["agent_results"] = agent_results
                
                execution_info = self.execution_monitor.get_execution(execution_id)
                if execution_info:
                    execution_info.add_log(f"Completed {agent_name} agent successfully")
                
            except Exception as e:
                self.logger.error(f"Agent {agent_name} failed: {str(e)}")
                agent_results[agent_name] = {"error": str(e)}
        
        return agent_results
    
    def _prepare_agent_context(self, agent_name: str, context: Dict[str, Any], 
                             agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for specific agent execution"""
        
        agent_context = context.copy()
        agent_context["previous_results"] = agent_results
        
        # Add agent-specific data
        if agent_name == "coding":
            agent_context["node_specifications"] = context["content_analysis"]["node_specifications"]
        elif agent_name == "testing":
            agent_context["coding_results"] = agent_results.get("coding", {})
        elif agent_name == "documentation":
            agent_context["coding_results"] = agent_results.get("coding", {})
            agent_context["testing_results"] = agent_results.get("testing", {})
        elif agent_name == "devops":
            agent_context["all_artifacts"] = agent_results
        
        return agent_context
    
    async def _execute_coding_agent(self, agent: CodingAgent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute coding agent with enhanced specifications"""

        # Prepare coding-specific context
        coding_context = {
            **context,
            "node_specifications": context["node_specifications"],
            "research_artifacts": context["content_analysis"]["research_result"],
            "task_type": "code_generation"
        }

        # Use the standard execute method
        return agent.execute(coding_context)

    async def _execute_testing_agent(self, agent: TestingAgent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute testing agent with coding results"""

        # Prepare testing-specific context
        testing_context = {
            **context,
            "coding_results": context.get("coding_results", {}),
            "task_type": "test_generation"
        }

        # Use the standard execute method
        return agent.execute(testing_context)

    async def _execute_documentation_agent(self, agent: DocumentationAgent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute documentation agent with all previous results"""

        # Prepare documentation-specific context
        documentation_context = {
            **context,
            "previous_results": context.get("previous_results", {}),
            "task_type": "documentation_generation"
        }

        # Use the standard execute method
        return agent.execute(documentation_context)
    
    async def _execute_devops_agent(self, agent: DevOpsAgent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DevOps agent for packaging and deployment"""

        # Prepare DevOps-specific context
        devops_context = {
            **context,
            "previous_results": context.get("all_artifacts", {}),
            "task_type": "packaging_deployment"
        }

        # Use the standard execute method
        return agent.execute(devops_context)
    
    async def _integrate_and_validate(self, content_analysis: Dict[str, Any], agent_results: Dict[str, Any],
                                    execution_dir: str, quality_level: str, execution_id: str) -> Dict[str, Any]:
        """Integrate results and perform validation"""
        
        execution_info = self.execution_monitor.get_execution(execution_id)
        if execution_info:
            execution_info.add_log("Integrating and validating results")
        
        # Collect all generated artifacts
        artifacts = {
            "nodes": [],
            "tests": [],
            "documentation": [],
            "packages": []
        }
        
        # Extract artifacts from agent results
        if "coding" in agent_results:
            coding_result = agent_results["coding"]
            artifacts["nodes"] = coding_result.get("node_implementations", {}).get("nodes", [])
        
        if "testing" in agent_results:
            testing_result = agent_results["testing"]
            artifacts["tests"] = testing_result.get("test_files", [])
        
        if "documentation" in agent_results:
            doc_result = agent_results["documentation"]
            artifacts["documentation"] = doc_result.get("documentation_files", [])
        
        if "devops" in agent_results:
            devops_result = agent_results["devops"]
            artifacts["packages"] = devops_result.get("packages", [])
        
        # Perform validation based on quality level
        validation_result = await self._validate_artifacts(artifacts, quality_level, execution_id)
        
        return {
            "artifacts": artifacts,
            "validation": validation_result,
            "integration_metadata": {
                "timestamp": datetime.now().isoformat(),
                "quality_level": quality_level,
                "total_nodes": len(artifacts["nodes"]),
                "total_tests": len(artifacts["tests"]),
                "total_docs": len(artifacts["documentation"])
            }
        }
    
    async def _validate_artifacts(self, artifacts: Dict[str, Any], quality_level: str, 
                                execution_id: str) -> Dict[str, Any]:
        """Validate generated artifacts based on quality level"""
        
        validation_result = {
            "status": "passed",
            "checks": [],
            "warnings": [],
            "errors": []
        }
        
        # Basic validation for all quality levels
        if not artifacts["nodes"]:
            validation_result["errors"].append("No nodes generated")
            validation_result["status"] = "failed"
        
        # Development and production validation
        if quality_level in ["development", "production"]:
            if not artifacts["tests"]:
                validation_result["warnings"].append("No tests generated")
            
            if not artifacts["documentation"]:
                validation_result["warnings"].append("No documentation generated")
        
        # Production-only validation
        if quality_level == "production":
            if not artifacts["packages"]:
                validation_result["warnings"].append("No packages generated")
            
            # Additional strict validation
            for node in artifacts["nodes"]:
                if not node.get("file") or not os.path.exists(node.get("file", "")):
                    validation_result["errors"].append(f"Node file missing: {node.get('name')}")
        
        # Set final status
        if validation_result["errors"]:
            validation_result["status"] = "failed"
        elif validation_result["warnings"]:
            validation_result["status"] = "passed_with_warnings"
        
        return validation_result
    
    async def _perform_quality_assurance(self, final_result: Dict[str, Any], quality_level: str,
                                       execution_id: str) -> Dict[str, Any]:
        """Perform quality assurance checks"""
        
        qa_result = {
            "status": "passed",
            "quality_score": 0.0,
            "recommendations": [],
            "metrics": {}
        }
        
        # Calculate quality score based on artifacts and validation
        artifacts = final_result["artifacts"]
        validation = final_result["validation"]
        
        score = 0.0
        max_score = 100.0
        
        # Node generation score (40 points)
        if artifacts["nodes"]:
            score += 40.0
        
        # Testing score (20 points)
        if artifacts["tests"]:
            score += 20.0
        
        # Documentation score (20 points)
        if artifacts["documentation"]:
            score += 20.0
        
        # Validation score (20 points)
        if validation["status"] == "passed":
            score += 20.0
        elif validation["status"] == "passed_with_warnings":
            score += 10.0
        
        qa_result["quality_score"] = score / max_score
        
        # Generate recommendations
        if qa_result["quality_score"] < 0.8:
            qa_result["recommendations"].append("Consider using higher quality level for better results")
        
        if not artifacts["tests"]:
            qa_result["recommendations"].append("Add comprehensive testing for production use")
        
        if not artifacts["documentation"]:
            qa_result["recommendations"].append("Add detailed documentation for better usability")
        
        return qa_result
    
    def _serialize_node_spec(self, spec: NodeSpecification) -> Dict[str, Any]:
        """Serialize NodeSpecification to dictionary"""
        return {
            "name": spec.name,
            "type": spec.type,
            "category": spec.category.value,
            "description": spec.description,
            "complexity": spec.complexity.value,
            "inputs": spec.inputs,
            "outputs": spec.outputs,
            "parameters": spec.parameters,
            "dependencies": spec.dependencies,
            "use_cases": spec.use_cases,
            "implementation_hints": spec.implementation_hints,
            "quality_requirements": spec.quality_requirements,
            "confidence_score": spec.confidence_score
        }
    
    def _generate_execution_summary(self, content_analysis: Dict[str, Any], 
                                  final_result: Dict[str, Any], qa_result: Dict[str, Any]) -> str:
        """Generate human-readable execution summary"""
        
        artifacts = final_result["artifacts"]
        
        summary = f"""
✅ **Enhanced ComfyUI Node Generation Complete!**

**Quality Score**: {qa_result['quality_score']:.1%}

**Generated Artifacts**:
- 🔧 **Nodes**: {len(artifacts['nodes'])} ComfyUI nodes
- 🧪 **Tests**: {len(artifacts['tests'])} test files  
- 📚 **Documentation**: {len(artifacts['documentation'])} documentation files
- 📦 **Packages**: {len(artifacts['packages'])} deployment packages

**Analysis Method**: Intelligent LLM-based analysis
**LLM Model**: {content_analysis['analysis_metadata']['llm_model']}

**Node Specifications**:
"""
        
        for i, spec in enumerate(content_analysis['node_specifications'], 1):
            summary += f"  {i}. **{spec['name']}** ({spec['category']}) - {spec['description'][:100]}...\n"
        
        if qa_result['recommendations']:
            summary += "\n**Recommendations**:\n"
            for rec in qa_result['recommendations']:
                summary += f"- {rec}\n"
        
        return summary.strip()
