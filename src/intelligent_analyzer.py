#!/usr/bin/env python3
"""
Intelligent Content Analyzer for ComfyUI Framework
Replaces simple keyword matching with sophisticated LLM-based analysis
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from llm_client import LLMManager


class NodeComplexity(Enum):
    """Node complexity levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ADVANCED = "advanced"


class NodeCategory(Enum):
    """Extended node categories"""
    IMAGE_PROCESSING = "image/processing"
    IMAGE_GENERATION = "image/generation"
    IMAGE_ENHANCEMENT = "image/enhancement"
    TEXT_PROCESSING = "text/processing"
    TEXT_GENERATION = "text/generation"
    AUDIO_PROCESSING = "audio/processing"
    AUDIO_GENERATION = "audio/generation"
    VIDEO_PROCESSING = "video/processing"
    MODEL_LOADING = "model/loading"
    MODEL_INFERENCE = "model/inference"
    DATA_PROCESSING = "data/processing"
    UTILITY = "utility"
    WORKFLOW = "workflow"
    CUSTOM = "custom"


@dataclass
class NodeSpecification:
    """Enhanced node specification with detailed metadata"""
    name: str
    type: str
    category: NodeCategory
    description: str
    complexity: NodeComplexity
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    parameters: Dict[str, Any]
    dependencies: List[str]
    use_cases: List[str]
    implementation_hints: List[str]
    quality_requirements: Dict[str, Any]
    confidence_score: float


class IntelligentContentAnalyzer:
    """
    Advanced content analyzer using LLM reasoning instead of simple keyword matching
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_manager = LLMManager(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Quality level configurations
        self.quality_configs = {
            "draft": {
                "analysis_depth": "basic",
                "llm_temperature": 0.3,
                "max_tokens": 2000,
                "validation_level": "minimal"
            },
            "development": {
                "analysis_depth": "standard",
                "llm_temperature": 0.1,
                "max_tokens": 3000,
                "validation_level": "standard"
            },
            "production": {
                "analysis_depth": "comprehensive",
                "llm_temperature": 0.05,
                "max_tokens": 4000,
                "validation_level": "strict"
            }
        }
    
    def analyze_content(self, content: str, title: str, input_source: str, 
                       focus_areas: List[str], quality_level: str = "production") -> List[NodeSpecification]:
        """
        Perform intelligent content analysis using LLM reasoning
        """
        self.logger.info(f"Starting intelligent analysis with quality level: {quality_level}")
        
        # Get quality configuration
        quality_config = self.quality_configs.get(quality_level, self.quality_configs["production"])
        
        # Step 1: Initial content classification
        classification = self._classify_content_with_llm(content, title, input_source, quality_config)
        
        # Step 2: Extract technical requirements
        technical_analysis = self._analyze_technical_requirements(content, classification, quality_config)
        
        # Step 3: Generate node specifications
        node_specs = self._generate_node_specifications(
            content, title, classification, technical_analysis, focus_areas, quality_config
        )
        
        # Step 4: Validate and enhance specifications
        validated_specs = self._validate_and_enhance_specs(node_specs, quality_config)
        
        self.logger.info(f"Generated {len(validated_specs)} node specifications")
        return validated_specs
    
    def _classify_content_with_llm(self, content: str, title: str, input_source: str, 
                                  quality_config: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to classify content and determine primary purpose"""
        
        classification_prompt = f"""
        Analyze the following content and classify it for ComfyUI node generation:

        Title: {title}
        Source: {input_source}
        Content Preview: {content[:2000]}...

        Please provide a detailed classification including:

        1. **Primary Domain**: What is the main field/domain? (computer vision, NLP, audio processing, etc.)
        2. **Technical Approach**: What algorithms, models, or techniques are used?
        3. **Input/Output Types**: What types of data does it process? (images, text, audio, etc.)
        4. **Complexity Level**: How complex is the implementation? (simple, medium, complex, advanced)
        5. **Key Features**: What are the main capabilities and features?
        6. **Dependencies**: What libraries, models, or external resources are needed?
        7. **Use Cases**: What are the practical applications?
        8. **ComfyUI Suitability**: How well does this fit into ComfyUI workflows?

        Respond in JSON format with detailed analysis.
        """
        
        try:
            response = self.llm_manager.generate(
                prompt=classification_prompt,
                system_prompt="You are an expert software architect specializing in ComfyUI node development and AI/ML systems analysis.",
                max_tokens=quality_config["max_tokens"],
                temperature=quality_config["llm_temperature"]
            )
            
            # Parse JSON response
            classification = self._parse_json_response(response.content)
            classification["llm_confidence"] = self._calculate_confidence(response.content)
            
            return classification
            
        except Exception as e:
            self.logger.error(f"LLM classification failed: {str(e)}")
            return self._fallback_classification(content, title)
    
    def _analyze_technical_requirements(self, content: str, classification: Dict[str, Any], 
                                      quality_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze technical implementation requirements"""
        
        technical_prompt = f"""
        Based on the content classification, analyze the technical implementation requirements:

        Classification: {json.dumps(classification, indent=2)}
        Content: {content[:3000]}...

        Provide detailed technical analysis:

        1. **Architecture Requirements**: What software architecture patterns are needed?
        2. **Data Flow**: How should data flow through the processing pipeline?
        3. **Performance Considerations**: What are the performance requirements and bottlenecks?
        4. **Error Handling**: What error conditions need to be handled?
        5. **Configuration Options**: What parameters should be configurable?
        6. **Integration Points**: How does this integrate with ComfyUI's node system?
        7. **Testing Strategy**: How should this be tested?
        8. **Scalability**: What are the scalability considerations?

        Respond in JSON format with implementation guidance.
        """
        
        try:
            response = self.llm_manager.generate(
                prompt=technical_prompt,
                system_prompt="You are a senior software engineer with expertise in ComfyUI architecture and AI/ML system implementation.",
                max_tokens=quality_config["max_tokens"],
                temperature=quality_config["llm_temperature"]
            )
            
            return self._parse_json_response(response.content)
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {str(e)}")
            return self._fallback_technical_analysis(classification)
    
    def _generate_node_specifications(self, content: str, title: str, classification: Dict[str, Any],
                                    technical_analysis: Dict[str, Any], focus_areas: List[str],
                                    quality_config: Dict[str, Any]) -> List[NodeSpecification]:
        """Generate detailed node specifications using LLM reasoning"""
        
        spec_prompt = f"""
        Generate ComfyUI node specifications based on the analysis:

        Title: {title}
        Classification: {json.dumps(classification, indent=2)}
        Technical Analysis: {json.dumps(technical_analysis, indent=2)}
        Focus Areas: {focus_areas}

        Generate 1-3 node specifications that would best implement this functionality in ComfyUI.
        For each node, provide:

        1. **Node Name**: Clear, descriptive name
        2. **Node Type**: Primary function type
        3. **Category**: ComfyUI category (use standard categories)
        4. **Description**: Detailed description of functionality
        5. **Complexity**: simple/medium/complex/advanced
        6. **Inputs**: Dictionary of input parameters with types
        7. **Outputs**: Dictionary of output parameters with types
        8. **Parameters**: Configuration parameters with defaults
        9. **Dependencies**: Required libraries/models
        10. **Use Cases**: Specific use cases and examples
        11. **Implementation Hints**: Key implementation guidance
        12. **Quality Requirements**: Testing and validation requirements

        Respond with a JSON array of node specifications.
        """
        
        try:
            response = self.llm_manager.generate(
                prompt=spec_prompt,
                system_prompt="You are an expert ComfyUI node developer with deep knowledge of the ComfyUI ecosystem and best practices.",
                max_tokens=quality_config["max_tokens"],
                temperature=quality_config["llm_temperature"]
            )
            
            specs_data = self._parse_json_response(response.content)
            
            # Convert to NodeSpecification objects
            node_specs = []
            for spec_data in specs_data.get("nodes", []):
                try:
                    node_spec = self._create_node_specification(spec_data, classification)
                    node_specs.append(node_spec)
                except Exception as e:
                    self.logger.warning(f"Failed to create node spec: {str(e)}")
            
            return node_specs
            
        except Exception as e:
            self.logger.error(f"Node specification generation failed: {str(e)}")
            return self._fallback_node_specifications(classification, title)
    
    def _create_node_specification(self, spec_data: Dict[str, Any], 
                                 classification: Dict[str, Any]) -> NodeSpecification:
        """Create NodeSpecification object from LLM response data"""
        
        # Map complexity
        complexity_map = {
            "simple": NodeComplexity.SIMPLE,
            "medium": NodeComplexity.MEDIUM,
            "complex": NodeComplexity.COMPLEX,
            "advanced": NodeComplexity.ADVANCED
        }
        
        # Map category
        category_str = spec_data.get("category", "custom").lower()
        category = self._map_category(category_str)
        
        return NodeSpecification(
            name=spec_data.get("name", "CustomNode"),
            type=spec_data.get("type", "processor"),
            category=category,
            description=spec_data.get("description", "Custom ComfyUI node"),
            complexity=complexity_map.get(spec_data.get("complexity", "medium"), NodeComplexity.MEDIUM),
            inputs=spec_data.get("inputs", {}),
            outputs=spec_data.get("outputs", {}),
            parameters=spec_data.get("parameters", {}),
            dependencies=spec_data.get("dependencies", []),
            use_cases=spec_data.get("use_cases", []),
            implementation_hints=spec_data.get("implementation_hints", []),
            quality_requirements=spec_data.get("quality_requirements", {}),
            confidence_score=classification.get("llm_confidence", 0.8)
        )
    
    def _map_category(self, category_str: str) -> NodeCategory:
        """Map category string to NodeCategory enum"""
        category_mapping = {
            "image": NodeCategory.IMAGE_PROCESSING,
            "image/processing": NodeCategory.IMAGE_PROCESSING,
            "image/generation": NodeCategory.IMAGE_GENERATION,
            "image/enhancement": NodeCategory.IMAGE_ENHANCEMENT,
            "text": NodeCategory.TEXT_PROCESSING,
            "text/processing": NodeCategory.TEXT_PROCESSING,
            "text/generation": NodeCategory.TEXT_GENERATION,
            "audio": NodeCategory.AUDIO_PROCESSING,
            "audio/processing": NodeCategory.AUDIO_PROCESSING,
            "audio/generation": NodeCategory.AUDIO_GENERATION,
            "video": NodeCategory.VIDEO_PROCESSING,
            "model": NodeCategory.MODEL_LOADING,
            "model/loading": NodeCategory.MODEL_LOADING,
            "model/inference": NodeCategory.MODEL_INFERENCE,
            "data": NodeCategory.DATA_PROCESSING,
            "utility": NodeCategory.UTILITY,
            "workflow": NodeCategory.WORKFLOW,
            "custom": NodeCategory.CUSTOM
        }
        
        return category_mapping.get(category_str, NodeCategory.CUSTOM)
    
    def _parse_json_response(self, response_content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with error handling"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Try parsing the entire response
                return json.loads(response_content)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response: {str(e)}")
            return {"error": "Failed to parse LLM response", "raw_response": response_content}
    
    def _calculate_confidence(self, response_content: str) -> float:
        """Calculate confidence score based on response quality"""
        # Simple heuristic - can be enhanced
        if "error" in response_content.lower():
            return 0.3
        elif len(response_content) < 100:
            return 0.5
        elif "uncertain" in response_content.lower() or "unclear" in response_content.lower():
            return 0.6
        else:
            return 0.8
    
    def _validate_and_enhance_specs(self, node_specs: List[NodeSpecification], 
                                  quality_config: Dict[str, Any]) -> List[NodeSpecification]:
        """Validate and enhance node specifications based on quality level"""
        validated_specs = []
        
        for spec in node_specs:
            try:
                # Validate required fields
                if not spec.name or not spec.description:
                    self.logger.warning(f"Skipping invalid spec: missing required fields")
                    continue
                
                # Enhance based on quality level
                if quality_config["validation_level"] == "strict":
                    spec = self._enhance_spec_for_production(spec)
                elif quality_config["validation_level"] == "standard":
                    spec = self._enhance_spec_for_development(spec)
                
                validated_specs.append(spec)
                
            except Exception as e:
                self.logger.warning(f"Failed to validate spec {spec.name}: {str(e)}")
        
        return validated_specs
    
    def _enhance_spec_for_production(self, spec: NodeSpecification) -> NodeSpecification:
        """Enhance specification for production quality"""
        # Add comprehensive error handling requirements
        if "error_handling" not in spec.quality_requirements:
            spec.quality_requirements["error_handling"] = "comprehensive"
        
        # Add performance requirements
        if "performance" not in spec.quality_requirements:
            spec.quality_requirements["performance"] = "optimized"
        
        # Add documentation requirements
        if "documentation" not in spec.quality_requirements:
            spec.quality_requirements["documentation"] = "complete"
        
        return spec
    
    def _enhance_spec_for_development(self, spec: NodeSpecification) -> NodeSpecification:
        """Enhance specification for development quality"""
        # Add basic requirements
        if "testing" not in spec.quality_requirements:
            spec.quality_requirements["testing"] = "unit_tests"
        
        return spec
    
    def _fallback_classification(self, content: str, title: str) -> Dict[str, Any]:
        """Fallback classification when LLM fails"""
        return {
            "primary_domain": "unknown",
            "technical_approach": "custom",
            "input_output_types": ["unknown"],
            "complexity_level": "medium",
            "key_features": ["custom processing"],
            "dependencies": [],
            "use_cases": ["general purpose"],
            "comfyui_suitability": "medium",
            "llm_confidence": 0.3
        }
    
    def _fallback_technical_analysis(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback technical analysis when LLM fails"""
        return {
            "architecture_requirements": ["basic_processing"],
            "data_flow": ["input -> process -> output"],
            "performance_considerations": ["standard"],
            "error_handling": ["basic"],
            "configuration_options": ["default_parameters"],
            "integration_points": ["comfyui_standard"],
            "testing_strategy": ["unit_tests"],
            "scalability": ["single_instance"]
        }
    
    def _fallback_node_specifications(self, classification: Dict[str, Any], title: str) -> List[NodeSpecification]:
        """Fallback node specifications when LLM fails"""
        return [
            NodeSpecification(
                name=f"{title.replace(' ', '')}Node",
                type="processor",
                category=NodeCategory.CUSTOM,
                description=f"Custom node based on {title}",
                complexity=NodeComplexity.MEDIUM,
                inputs={"input": "any"},
                outputs={"output": "any"},
                parameters={"enabled": True},
                dependencies=[],
                use_cases=["general processing"],
                implementation_hints=["implement basic processing logic"],
                quality_requirements={"testing": "basic"},
                confidence_score=0.3
            )
        ]
