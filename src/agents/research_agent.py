"""
Research Agent for MCP Multi-Agent Framework
Analyzes research papers, repositories, and technical specifications
"""

import os
import re
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

# Professional import structure with fallback handling
import sys
import os
from pathlib import Path

# Add parent directory to path for absolute imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import with proper error handling
try:
    from agents.base_agent import BaseAgent
    from utils import save_text, save_json, extract_pdf_text, clone_repository
    from llm_client import LLMManager
    from web_scraper import AdvancedWebScraper, ScrapedContent
except ImportError as e:
    # Fallback for different execution contexts
    try:
        from .base_agent import BaseAgent
        from ..utils import save_text, save_json, extract_pdf_text, clone_repository
        from ..llm_client import LLMManager
        from ..web_scraper import AdvancedWebScraper, ScrapedContent
    except ImportError:
        # Final fallback - create minimal implementations
        print(f"Warning: Could not import dependencies: {e}")

        class BaseAgent:
            def __init__(self, config): pass
            def execute(self, context): return {"status": "error", "message": "Dependencies not available"}

        class LLMManager:
            def __init__(self, config): pass
            def generate(self, *args, **kwargs): return type('Response', (), {'content': 'Mock response'})()

        class AdvancedWebScraper:
            def __init__(self, config): pass
            def scrape_url(self, url, method="auto"): return type('Content', (), {'content': '', 'title': '', 'url': url})()

        ScrapedContent = type('ScrapedContent', (), {})

        def save_text(content, path): pass
        def save_json(data, path): pass
        def extract_pdf_text(path): return ""
        def clone_repository(url, path): return path

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False


class ResearchAgent(BaseAgent):
    """
    Agent responsible for research analysis and requirement extraction
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.supported_sources = ["arxiv", "github", "pdf", "directory"]

        # Initialize LLM manager for intelligent analysis (if not already done by base class)
        if not hasattr(self, 'llm_manager') or self.llm_manager is None:
            self.llm_manager = LLMManager(config)
        self._log_info(f"Initialized ResearchAgent with LLM model: {self.llm_manager.client.model}")

        # Initialize advanced web scraper
        scraper_config = {
            'request_delay': config.get('scraper_delay', 1.0),
            'timeout': config.get('scraper_timeout', 30)
        }
        self.web_scraper = AdvancedWebScraper(scraper_config)
        self._log_info("Initialized advanced web scraper")
        
    def _process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main research processing method
        
        Args:
            context: Execution context containing input source
            
        Returns:
            Research analysis results
        """
        input_source = context["input_source"]
        input_type = context["input_type"]
        focus_areas = context.get("focus_areas", [])
        
        self._log_info(f"Analyzing {input_type} source: {input_source}")
        
        # Analyze input based on type
        if input_type == "arxiv":
            analysis = self._analyze_arxiv_paper(input_source, focus_areas)
        elif input_type == "github":
            analysis = self._analyze_github_repository(input_source, focus_areas)
        elif input_type == "url":
            analysis = self._analyze_web_url(input_source, focus_areas)
        elif input_type == "pdf":
            analysis = self._analyze_pdf_file(input_source, focus_areas)
        elif input_type == "directory":
            analysis = self._analyze_directory(input_source, focus_areas)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        
        # Generate implementation requirements
        requirements = self._generate_requirements(analysis, focus_areas)
        
        # Create research artifacts
        artifacts = self._create_research_artifacts(analysis, requirements)
        
        # Calculate metrics
        metrics = self._calculate_metrics(analysis, requirements)
        
        return {
            "analysis": analysis,
            "requirements": requirements,
            "artifacts": artifacts,
            "metrics": metrics,
            "summary": f"Analyzed {input_type} source and generated {len(requirements.get('nodes', []))} node specifications"
        }
    
    def _analyze_arxiv_paper(self, arxiv_url: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Analyze arXiv paper using advanced web scraping"""
        self._log_info("Analyzing arXiv paper with advanced scraping")

        try:
            # Use advanced web scraper for arXiv
            scraped_content = self.web_scraper.scrape_url(arxiv_url, method="arxiv")

            if scraped_content.content_type == "error":
                raise ValueError(f"Failed to scrape arXiv paper: {scraped_content.content}")

            # Analyze content using LLM
            analysis = self._analyze_paper_content(scraped_content.content, focus_areas)

            # Enhance with scraped metadata
            analysis["source_type"] = "arxiv"
            analysis["title"] = scraped_content.title
            analysis["authors"] = scraped_content.authors
            analysis["publication_date"] = scraped_content.publication_date
            analysis["abstract"] = scraped_content.abstract
            analysis["arxiv_metadata"] = scraped_content.metadata
            analysis["arxiv_url"] = arxiv_url

            # Add content quality metrics
            analysis["content_quality"] = {
                "content_length": len(scraped_content.content),
                "has_abstract": bool(scraped_content.abstract),
                "has_authors": bool(scraped_content.authors),
                "scraping_method": "advanced_arxiv"
            }

            self._log_info(f"Successfully analyzed arXiv paper: {scraped_content.title}")
            return analysis

        except Exception as e:
            self._log_error(f"Advanced arXiv scraping failed: {str(e)}")
            # Fallback to basic method
            return self._analyze_arxiv_paper_fallback(arxiv_url, focus_areas)

    def _analyze_arxiv_paper_fallback(self, arxiv_url: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Fallback method for arXiv paper analysis"""
        self._log_info("Using fallback arXiv analysis method")

        # Extract arXiv ID
        arxiv_id = self._extract_arxiv_id(arxiv_url)
        if not arxiv_id:
            raise ValueError("Invalid arXiv URL")

        # Download paper using basic method
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_path = f"{self.agent_output_dir}/paper_{arxiv_id}.pdf"

        try:
            response = requests.get(pdf_url)
            response.raise_for_status()

            with open(pdf_path, 'wb') as f:
                f.write(response.content)

            self._log_info(f"Downloaded paper to {pdf_path}")

            # Extract text and analyze
            paper_text = extract_pdf_text(pdf_path)
            analysis = self._analyze_paper_content(paper_text, focus_areas)
            analysis["source_type"] = "arxiv"
            analysis["arxiv_id"] = arxiv_id
            analysis["pdf_path"] = pdf_path
            analysis["scraping_method"] = "fallback"

            return analysis

        except Exception as e:
            self._log_error(f"Fallback arXiv analysis failed: {e}")
            raise

    def _analyze_github_repository(self, github_url: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Analyze GitHub repository with enhanced web scraping"""
        self._log_info("Analyzing GitHub repository with advanced scraping")

        try:
            # First, scrape GitHub page for metadata
            scraped_content = self.web_scraper.scrape_url(github_url, method="selenium")

            # Clone repository for code analysis
            repo_dir = f"{self.agent_output_dir}/repository"
            try:
                clone_repository(github_url, repo_dir)
                self._log_info(f"Cloned repository to {repo_dir}")

                # Analyze repository structure and code
                code_analysis = self._analyze_repository_content(repo_dir, focus_areas)

                # Combine web scraping and code analysis
                analysis = code_analysis
                analysis["source_type"] = "github"
                analysis["repository_url"] = github_url
                analysis["repository_path"] = repo_dir

                # Add scraped metadata
                if scraped_content.content_type != "error":
                    analysis["github_metadata"] = {
                        "page_title": scraped_content.title,
                        "description": scraped_content.content[:500] + "..." if len(scraped_content.content) > 500 else scraped_content.content,
                        "scraped_content": scraped_content.metadata
                    }

                # Extract additional GitHub-specific information
                analysis.update(self._extract_github_metadata(github_url, repo_dir))

                return analysis

            except Exception as e:
                self._log_error(f"Failed to clone repository: {e}")
                # Fallback to web scraping only
                return self._analyze_github_web_only(github_url, scraped_content, focus_areas)

        except Exception as e:
            self._log_error(f"GitHub analysis failed: {str(e)}")
            raise

    def _analyze_github_web_only(self, github_url: str, scraped_content: ScrapedContent, focus_areas: List[str]) -> Dict[str, Any]:
        """Analyze GitHub repository using only web scraping"""
        self._log_info("Analyzing GitHub repository using web scraping only")

        # Analyze scraped content
        analysis = self._analyze_paper_content(scraped_content.content, focus_areas)
        analysis["source_type"] = "github_web"
        analysis["repository_url"] = github_url
        analysis["title"] = scraped_content.title
        analysis["scraping_method"] = "web_only"

        return analysis

    def _extract_github_metadata(self, github_url: str, repo_dir: str) -> Dict[str, Any]:
        """Extract GitHub-specific metadata"""
        metadata = {}

        try:
            # Extract repository info from URL
            parts = github_url.rstrip('/').split('/')
            if len(parts) >= 2:
                metadata["owner"] = parts[-2]
                metadata["repo_name"] = parts[-1]

            # Check for README files
            readme_files = []
            for filename in ["README.md", "README.rst", "README.txt", "readme.md"]:
                readme_path = os.path.join(repo_dir, filename)
                if os.path.exists(readme_path):
                    readme_files.append(filename)
                    try:
                        with open(readme_path, 'r', encoding='utf-8') as f:
                            metadata[f"readme_content"] = f.read()[:2000]  # First 2000 chars
                    except Exception:
                        pass

            metadata["readme_files"] = readme_files

            # Check for common files
            common_files = ["requirements.txt", "setup.py", "package.json", "Cargo.toml", "go.mod"]
            found_files = []
            for filename in common_files:
                if os.path.exists(os.path.join(repo_dir, filename)):
                    found_files.append(filename)

            metadata["dependency_files"] = found_files

            # Count files by extension
            file_counts = {}
            for root, dirs, files in os.walk(repo_dir):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext:
                        file_counts[ext] = file_counts.get(ext, 0) + 1

            metadata["file_types"] = dict(sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        except Exception as e:
            self._log_warning(f"Failed to extract GitHub metadata: {str(e)}")

        return metadata

    def _analyze_web_url(self, url: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Analyze any web URL using advanced scraping"""
        self._log_info(f"Analyzing web URL with advanced scraping: {url}")

        try:
            # Use advanced web scraper with auto method selection
            scraped_content = self.web_scraper.scrape_url(url, method="auto")

            if scraped_content.content_type == "error":
                raise ValueError(f"Failed to scrape URL: {scraped_content.content}")

            # Analyze content using LLM
            analysis = self._analyze_paper_content(scraped_content.content, focus_areas)

            # Add web-specific metadata
            analysis["source_type"] = "web"
            analysis["url"] = url
            analysis["title"] = scraped_content.title
            analysis["authors"] = scraped_content.authors
            analysis["publication_date"] = scraped_content.publication_date
            analysis["content_type"] = scraped_content.content_type
            analysis["web_metadata"] = scraped_content.metadata

            # Add content quality metrics
            analysis["content_quality"] = {
                "content_length": len(scraped_content.content),
                "has_title": bool(scraped_content.title),
                "has_authors": bool(scraped_content.authors),
                "scraping_method": "advanced_web"
            }

            self._log_info(f"Successfully analyzed web URL: {scraped_content.title}")
            return analysis

        except Exception as e:
            self._log_error(f"Web URL analysis failed: {str(e)}")
            raise

    def _analyze_pdf_file(self, pdf_path: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Analyze local PDF file"""
        self._log_info(f"Analyzing PDF file: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract text and analyze
        paper_text = extract_pdf_text(pdf_path)
        analysis = self._analyze_paper_content(paper_text, focus_areas)
        analysis["source_type"] = "pdf"
        analysis["pdf_path"] = pdf_path
        
        return analysis
    
    def _analyze_directory(self, directory_path: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Analyze local directory"""
        self._log_info(f"Analyzing directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Analyze directory structure and content
        analysis = self._analyze_repository_content(directory_path, focus_areas)
        analysis["source_type"] = "directory"
        analysis["directory_path"] = directory_path
        
        return analysis
    
    def _analyze_paper_content(self, text: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Analyze paper text content using LLM-powered analysis"""
        self._log_info("Performing LLM-powered analysis of paper content")

        # Truncate text if too long (keep first 50k characters for analysis)
        if len(text) > 50000:
            text = text[:50000] + "\n\n[Content truncated for analysis]"

        # Create comprehensive analysis prompt
        analysis_prompt = self._create_paper_analysis_prompt(text, focus_areas)

        # Get LLM analysis
        try:
            response = self.llm_manager.generate(
                prompt=analysis_prompt,
                system_prompt="You are an expert research analyst specializing in computer vision, AI, and ComfyUI node development. Provide detailed, technical analysis.",
                max_tokens=4000,
                temperature=0.1
            )

            # Parse LLM response into structured data
            analysis = self._parse_llm_analysis_response(response.content)

            # Add metadata
            analysis["llm_model"] = self.llm_manager.client.model
            analysis["analysis_tokens"] = response.usage.get("total_tokens", 0)

            self._log_info(f"LLM analysis completed using {analysis['llm_model']}")

        except Exception as e:
            self._log_error(f"LLM analysis failed: {str(e)}")
            # Fallback to basic extraction
            analysis = self._fallback_paper_analysis(text, focus_areas)

        return analysis

    def _create_paper_analysis_prompt(self, text: str, focus_areas: List[str]) -> str:
        """Create comprehensive analysis prompt for LLM"""

        focus_text = ", ".join(focus_areas) if focus_areas else "general computer vision and AI techniques"

        prompt = f"""
Please analyze this research paper and extract detailed information for ComfyUI node development.

Focus Areas: {focus_text}

Paper Content:
{text}

Please provide a comprehensive analysis in the following JSON format:

{{
    "title": "Paper title",
    "abstract": "Key points from abstract",
    "methodology": {{
        "main_approach": "Primary methodology described",
        "key_algorithms": ["algorithm1", "algorithm2"],
        "novel_techniques": ["technique1", "technique2"],
        "mathematical_foundations": "Key mathematical concepts"
    }},
    "implementation_details": {{
        "architecture": "Model/system architecture description",
        "input_requirements": ["input type 1", "input type 2"],
        "output_formats": ["output type 1", "output type 2"],
        "processing_steps": ["step1", "step2", "step3"],
        "parameters": ["param1", "param2"]
    }},
    "comfyui_opportunities": {{
        "potential_nodes": [
            {{
                "name": "NodeName",
                "function": "What it does",
                "inputs": ["input1", "input2"],
                "outputs": ["output1"],
                "complexity": "low/medium/high",
                "implementation_priority": "high/medium/low"
            }}
        ],
        "integration_points": ["existing ComfyUI features this could connect to"],
        "workflow_applications": ["use case 1", "use case 2"]
    }},
    "technical_requirements": {{
        "dependencies": ["library1", "library2"],
        "computational_requirements": "GPU/CPU requirements",
        "memory_requirements": "Memory needs",
        "performance_considerations": "Speed/efficiency notes"
    }},
    "implementation_complexity": {{
        "overall_difficulty": "low/medium/high",
        "key_challenges": ["challenge1", "challenge2"],
        "estimated_development_time": "time estimate",
        "required_expertise": ["skill1", "skill2"]
    }}
}}

Provide only the JSON response, no additional text.
"""
        return prompt

    def _parse_llm_analysis_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response into structured analysis data"""

        try:
            # Try to extract JSON from response
            import json

            # Find JSON content (handle cases where LLM adds extra text)
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_content = response_content[start_idx:end_idx]
                analysis = json.loads(json_content)

                # Validate required fields
                required_fields = ["title", "methodology", "implementation_details", "comfyui_opportunities"]
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = {}

                return analysis
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            self._log_error(f"Failed to parse LLM response: {str(e)}")
            # Return structured fallback
            return {
                "title": "Analysis Failed",
                "abstract": response_content[:500] + "..." if len(response_content) > 500 else response_content,
                "methodology": {"main_approach": "LLM parsing failed"},
                "implementation_details": {"architecture": "Unknown"},
                "comfyui_opportunities": {"potential_nodes": []},
                "technical_requirements": {"dependencies": []},
                "implementation_complexity": {"overall_difficulty": "unknown"},
                "parse_error": str(e)
            }

    def _fallback_paper_analysis(self, text: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Fallback analysis when LLM fails"""
        self._log_info("Using fallback analysis method")

        # Basic text extraction
        title = self._extract_title_basic(text)
        abstract = self._extract_section_basic(text, "abstract")

        return {
            "title": title,
            "abstract": abstract,
            "methodology": {"main_approach": "Basic extraction - LLM unavailable"},
            "implementation_details": {"architecture": "Unknown"},
            "comfyui_opportunities": {
                "potential_nodes": [{
                    "name": "BasicProcessingNode",
                    "function": "Generic processing based on paper",
                    "inputs": ["image"],
                    "outputs": ["processed_image"],
                    "complexity": "medium",
                    "implementation_priority": "medium"
                }]
            },
            "technical_requirements": {"dependencies": ["torch", "numpy"]},
            "implementation_complexity": {"overall_difficulty": "medium"},
            "fallback_used": True
        }

    def _extract_title_basic(self, text: str) -> str:
        """Basic title extraction"""
        lines = text.split('\n')[:10]  # Check first 10 lines
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200 and not line.lower().startswith('abstract'):
                return line
        return "Unknown Title"

    def _extract_section_basic(self, text: str, section: str) -> str:
        """Basic section extraction"""
        text_lower = text.lower()
        section_start = text_lower.find(section.lower())
        if section_start != -1:
            # Find next section or take next 1000 characters
            next_section_patterns = ['introduction', 'method', 'results', 'conclusion', 'references']
            section_end = len(text)

            for pattern in next_section_patterns:
                if pattern != section.lower():
                    pattern_pos = text_lower.find(pattern, section_start + len(section))
                    if pattern_pos != -1 and pattern_pos < section_end:
                        section_end = pattern_pos

            # Limit to reasonable length
            section_end = min(section_end, section_start + 2000)
            return text[section_start:section_end].strip()

        return f"No {section} section found"

    def _analyze_repository_content(self, repo_path: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Analyze repository structure and code"""
        self._log_info("Analyzing repository structure and code")
        
        # Analyze file structure
        file_structure = self._analyze_file_structure(repo_path)
        
        # Find key files
        key_files = self._identify_key_files(repo_path)
        
        # Analyze code patterns
        code_patterns = self._analyze_code_patterns(repo_path)
        
        # Extract documentation
        documentation = self._extract_documentation(repo_path)
        
        # Identify models and architectures
        models = self._identify_models_in_code(repo_path)
        
        analysis = {
            "file_structure": file_structure,
            "key_files": key_files,
            "code_patterns": code_patterns,
            "documentation": documentation,
            "models": models,
            "focus_alignment": self._assess_code_focus_alignment(repo_path, focus_areas),
            "complexity_assessment": self._assess_code_complexity(repo_path),
            "dependencies": self._extract_dependencies_from_code(repo_path)
        }
        
        return analysis
    
    def _generate_requirements(self, analysis: Dict[str, Any], focus_areas: List[str]) -> Dict[str, Any]:
        """Generate implementation requirements from analysis"""
        self._log_info("Generating implementation requirements")
        
        # Generate node specifications
        nodes = self._generate_node_specifications(analysis, focus_areas)
        
        # Identify dependencies
        dependencies = self._consolidate_dependencies(analysis)
        
        # Create implementation roadmap
        roadmap = self._create_implementation_roadmap(nodes, dependencies)
        
        # Assess feasibility
        feasibility = self._assess_implementation_feasibility(nodes, dependencies)
        
        requirements = {
            "nodes": nodes,
            "dependencies": dependencies,
            "roadmap": roadmap,
            "feasibility": feasibility,
            "estimated_complexity": self._estimate_implementation_complexity(nodes),
            "recommended_approach": self._recommend_implementation_approach(analysis, nodes)
        }
        
        return requirements
    
    def _generate_node_specifications(self, analysis: Dict[str, Any], focus_areas: List[str]) -> List[Dict[str, Any]]:
        """Generate ComfyUI node specifications"""
        nodes = []
        
        # Based on analysis, generate appropriate node types
        if "models" in analysis and analysis["models"]:
            # Model loader node
            nodes.append({
                "name": "ModelLoader",
                "type": "loader",
                "description": "Load and initialize the main model",
                "inputs": ["model_path", "device", "precision"],
                "outputs": ["model", "status"],
                "complexity": "medium"
            })
        
        # Main processing node
        nodes.append({
            "name": "MainProcessor", 
            "type": "processor",
            "description": "Main processing functionality",
            "inputs": ["model", "image", "parameters"],
            "outputs": ["result", "status"],
            "complexity": "high"
        })
        
        # Add focus-specific nodes
        if focus_areas:
            for focus in focus_areas:
                if focus.lower() in ["segmentation", "mask"]:
                    nodes.append({
                        "name": "SegmentationProcessor",
                        "type": "processor", 
                        "description": "Segmentation and masking functionality",
                        "inputs": ["model", "image", "prompts"],
                        "outputs": ["masks", "segments"],
                        "complexity": "medium"
                    })
                elif focus.lower() in ["3d", "reconstruction"]:
                    nodes.append({
                        "name": "3DReconstructor",
                        "type": "processor",
                        "description": "3D reconstruction functionality", 
                        "inputs": ["model", "image", "parameters"],
                        "outputs": ["mesh", "pointcloud"],
                        "complexity": "high"
                    })
        
        # Utility nodes
        nodes.append({
            "name": "ResultExporter",
            "type": "exporter",
            "description": "Export results in various formats",
            "inputs": ["data", "format", "path"],
            "outputs": ["file_path", "status"],
            "complexity": "low"
        })
        
        return nodes
    
    def _create_research_artifacts(self, analysis: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create research artifacts"""
        artifacts = {}
        
        # Save analysis report
        analysis_file = f"{self.agent_output_dir}/research_analysis.json"
        save_json(analysis, analysis_file)
        artifacts["analysis_report"] = analysis_file
        
        # Save requirements specification
        requirements_file = f"{self.agent_output_dir}/implementation_requirements.json"
        save_json(requirements, requirements_file)
        artifacts["requirements_spec"] = requirements_file
        
        # Create summary report
        summary = self._create_summary_report(analysis, requirements)
        summary_file = f"{self.agent_output_dir}/research_summary.md"
        save_text(summary, summary_file)
        artifacts["summary_report"] = summary_file
        
        return artifacts
    
    def _calculate_metrics(self, analysis: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate research metrics"""
        return {
            "nodes_identified": len(requirements.get("nodes", [])),
            "dependencies_count": len(requirements.get("dependencies", [])),
            "complexity_score": requirements.get("estimated_complexity", 0),
            "feasibility_score": requirements.get("feasibility", {}).get("score", 0),
            "focus_alignment_score": analysis.get("focus_alignment", {}).get("score", 0)
        }
    
    # Helper methods (simplified implementations)
    def _extract_arxiv_id(self, url: str) -> Optional[str]:
        """Extract arXiv ID from URL"""
        match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', url)
        return match.group(1) if match else None
    
    def _extract_section(self, text: str, section: str) -> str:
        """Extract specific section from paper text"""
        # Simplified implementation
        pattern = rf'{section}.*?(?=\n[A-Z]|\n\d+\.|\Z)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(0) if match else ""
    
    def _extract_title(self, text: str) -> str:
        """Extract paper title"""
        lines = text.split('\n')[:10]  # Check first 10 lines
        for line in lines:
            if len(line.strip()) > 10 and not line.strip().startswith(('Abstract', 'Introduction')):
                return line.strip()
        return "Unknown Title"
    
    def _extract_algorithms(self, text: str) -> List[str]:
        """Extract algorithm names from text"""
        # Simplified pattern matching
        algorithms = []
        patterns = [r'Algorithm \d+', r'[A-Z][a-z]+Net', r'[A-Z][a-z]+GAN']
        for pattern in patterns:
            matches = re.findall(pattern, text)
            algorithms.extend(matches)
        return list(set(algorithms))
    
    def _extract_models(self, text: str) -> List[str]:
        """Extract model names from text"""
        # Simplified implementation
        models = []
        common_models = ['ResNet', 'VGG', 'BERT', 'GPT', 'Transformer', 'CNN', 'RNN', 'LSTM']
        for model in common_models:
            if model.lower() in text.lower():
                models.append(model)
        return models
    
    def _extract_datasets(self, text: str) -> List[str]:
        """Extract dataset names from text"""
        # Simplified implementation
        datasets = []
        common_datasets = ['ImageNet', 'COCO', 'MNIST', 'CIFAR', 'Pascal VOC']
        for dataset in common_datasets:
            if dataset.lower() in text.lower():
                datasets.append(dataset)
        return datasets
    
    def _identify_implementation_opportunities(self, text: str, focus_areas: List[str]) -> List[str]:
        """Identify implementation opportunities"""
        opportunities = []
        
        # Look for implementation hints
        if 'real-time' in text.lower():
            opportunities.append("Real-time processing capability")
        if 'interactive' in text.lower():
            opportunities.append("Interactive user interface")
        if 'batch' in text.lower():
            opportunities.append("Batch processing support")
        
        return opportunities
    
    def _assess_focus_alignment(self, text: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Assess alignment with focus areas"""
        if not focus_areas:
            return {"score": 1.0, "details": "No specific focus areas"}
        
        alignment_score = 0
        for focus in focus_areas:
            if focus.lower() in text.lower():
                alignment_score += 1
        
        score = alignment_score / len(focus_areas)
        return {"score": score, "aligned_areas": alignment_score, "total_areas": len(focus_areas)}
    
    def _assess_complexity(self, text: str) -> Dict[str, Any]:
        """Assess implementation complexity"""
        complexity_indicators = ['neural network', 'deep learning', 'optimization', 'training']
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in text.lower())
        
        if complexity_score >= 3:
            level = "high"
        elif complexity_score >= 1:
            level = "medium"
        else:
            level = "low"
        
        return {"level": level, "score": complexity_score, "indicators": complexity_indicators}
    
    def _identify_dependencies(self, text: str) -> List[str]:
        """Identify potential dependencies"""
        dependencies = []
        
        # Common ML libraries
        ml_libs = ['pytorch', 'tensorflow', 'numpy', 'opencv', 'scikit-learn', 'pandas']
        for lib in ml_libs:
            if lib in text.lower():
                dependencies.append(lib)
        
        return dependencies
    
    def _create_summary_report(self, analysis: Dict[str, Any], requirements: Dict[str, Any]) -> str:
        """Create markdown summary report"""
        return f"""# Research Analysis Summary

## Source Information
- **Type**: {analysis.get('source_type', 'Unknown')}
- **Title**: {analysis.get('title', 'Unknown')}

## Key Findings
- **Models Identified**: {', '.join(analysis.get('models', []))}
- **Algorithms**: {', '.join(analysis.get('algorithms', []))}
- **Complexity**: {analysis.get('complexity_assessment', {}).get('level', 'Unknown')}

## Implementation Requirements
- **Nodes to Implement**: {len(requirements.get('nodes', []))}
- **Dependencies**: {len(requirements.get('dependencies', []))}
- **Estimated Complexity**: {requirements.get('estimated_complexity', 'Unknown')}

## Recommended Approach
{requirements.get('recommended_approach', 'Standard implementation approach')}
"""
    
    # Additional helper methods would be implemented here
    def _analyze_file_structure(self, path: str) -> Dict[str, Any]:
        """Analyze repository file structure"""
        return {"files": [], "directories": [], "total_files": 0}
    
    def _identify_key_files(self, path: str) -> List[str]:
        """Identify key files in repository"""
        return []
    
    def _analyze_code_patterns(self, path: str) -> Dict[str, Any]:
        """Analyze code patterns"""
        return {"patterns": [], "frameworks": []}
    
    def _extract_documentation(self, path: str) -> Dict[str, Any]:
        """Extract documentation from repository"""
        return {"readme": "", "docs": []}
    
    def _identify_models_in_code(self, path: str) -> List[str]:
        """Identify models in code"""
        return []
    
    def _assess_code_focus_alignment(self, path: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Assess code alignment with focus areas"""
        return {"score": 0.5}
    
    def _assess_code_complexity(self, path: str) -> Dict[str, Any]:
        """Assess code complexity"""
        return {"level": "medium"}
    
    def _extract_dependencies_from_code(self, path: str) -> List[str]:
        """Extract dependencies from code"""
        return []
    
    def _consolidate_dependencies(self, analysis: Dict[str, Any]) -> List[str]:
        """Consolidate all identified dependencies"""
        return analysis.get("dependencies", [])
    
    def _create_implementation_roadmap(self, nodes: List[Dict[str, Any]], dependencies: List[str]) -> Dict[str, Any]:
        """Create implementation roadmap"""
        return {"phases": [], "timeline": ""}
    
    def _assess_implementation_feasibility(self, nodes: List[Dict[str, Any]], dependencies: List[str]) -> Dict[str, Any]:
        """Assess implementation feasibility"""
        return {"score": 0.8, "challenges": [], "recommendations": []}
    
    def _estimate_implementation_complexity(self, nodes: List[Dict[str, Any]]) -> str:
        """Estimate overall implementation complexity"""
        high_complexity_nodes = [n for n in nodes if n.get("complexity") == "high"]
        if len(high_complexity_nodes) > 2:
            return "high"
        elif len(high_complexity_nodes) > 0:
            return "medium"
        else:
            return "low"
    
    def _recommend_implementation_approach(self, analysis: Dict[str, Any], nodes: List[Dict[str, Any]]) -> str:
        """Recommend implementation approach"""
        return "Implement nodes in order of complexity, starting with utility nodes and building up to main processing nodes."
