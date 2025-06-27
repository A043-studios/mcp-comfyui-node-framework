#!/usr/bin/env python3
"""
Enhanced Template System for ComfyUI Framework
Provides comprehensive template coverage for various node types
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

from intelligent_analyzer import NodeSpecification, NodeCategory, NodeComplexity


class TemplateType(Enum):
    """Template types for different node categories"""
    IMAGE_PROCESSOR = "image_processor"
    IMAGE_GENERATOR = "image_generator"
    IMAGE_ENHANCER = "image_enhancer"
    TEXT_PROCESSOR = "text_processor"
    TEXT_GENERATOR = "text_generator"
    AUDIO_PROCESSOR = "audio_processor"
    AUDIO_GENERATOR = "audio_generator"
    VIDEO_PROCESSOR = "video_processor"
    MODEL_LOADER = "model_loader"
    MODEL_INFERENCE = "model_inference"
    DATA_PROCESSOR = "data_processor"
    UTILITY = "utility"
    WORKFLOW = "workflow"
    CUSTOM_ML = "custom_ml"


@dataclass
class NodeTemplate:
    """Enhanced node template with comprehensive metadata"""
    template_type: TemplateType
    name: str
    description: str
    base_code: str
    required_imports: List[str]
    optional_imports: List[str]
    default_inputs: Dict[str, str]
    default_outputs: Dict[str, str]
    default_parameters: Dict[str, Any]
    complexity_variants: Dict[NodeComplexity, Dict[str, Any]]
    use_case_examples: List[str]
    implementation_notes: List[str]


class EnhancedTemplateSystem:
    """
    Enhanced template system with broad node type coverage and adaptive selection
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize comprehensive template library"""
        
        # Image Processing Templates
        self.templates[TemplateType.IMAGE_PROCESSOR] = NodeTemplate(
            template_type=TemplateType.IMAGE_PROCESSOR,
            name="ImageProcessor",
            description="Generic image processing node",
            base_code=self._get_image_processor_template(),
            required_imports=["torch", "torchvision", "PIL"],
            optional_imports=["cv2", "numpy", "skimage"],
            default_inputs={"image": "IMAGE", "parameters": "DICT"},
            default_outputs={"processed_image": "IMAGE", "metadata": "DICT"},
            default_parameters={"strength": 1.0, "enabled": True},
            complexity_variants={
                NodeComplexity.SIMPLE: {"processing_steps": 1, "validation": "basic"},
                NodeComplexity.MEDIUM: {"processing_steps": 3, "validation": "standard"},
                NodeComplexity.COMPLEX: {"processing_steps": 5, "validation": "comprehensive"},
                NodeComplexity.ADVANCED: {"processing_steps": 10, "validation": "strict"}
            },
            use_case_examples=["Image enhancement", "Style transfer", "Noise reduction"],
            implementation_notes=["Handle different image formats", "Implement batch processing", "Add progress callbacks"]
        )
        
        # Text Processing Templates
        self.templates[TemplateType.TEXT_PROCESSOR] = NodeTemplate(
            template_type=TemplateType.TEXT_PROCESSOR,
            name="TextProcessor",
            description="Generic text processing node",
            base_code=self._get_text_processor_template(),
            required_imports=["transformers", "torch"],
            optional_imports=["nltk", "spacy", "regex"],
            default_inputs={"text": "STRING", "model": "MODEL", "parameters": "DICT"},
            default_outputs={"processed_text": "STRING", "embeddings": "TENSOR", "metadata": "DICT"},
            default_parameters={"max_length": 512, "temperature": 0.7},
            complexity_variants={
                NodeComplexity.SIMPLE: {"model_size": "small", "features": ["basic_processing"]},
                NodeComplexity.MEDIUM: {"model_size": "medium", "features": ["tokenization", "embedding"]},
                NodeComplexity.COMPLEX: {"model_size": "large", "features": ["sentiment", "ner", "summarization"]},
                NodeComplexity.ADVANCED: {"model_size": "xl", "features": ["custom_training", "fine_tuning"]}
            },
            use_case_examples=["Text summarization", "Sentiment analysis", "Language translation"],
            implementation_notes=["Handle multiple languages", "Implement streaming", "Add caching"]
        )
        
        # Audio Processing Templates
        self.templates[TemplateType.AUDIO_PROCESSOR] = NodeTemplate(
            template_type=TemplateType.AUDIO_PROCESSOR,
            name="AudioProcessor",
            description="Generic audio processing node",
            base_code=self._get_audio_processor_template(),
            required_imports=["librosa", "torch", "torchaudio"],
            optional_imports=["soundfile", "scipy", "numpy"],
            default_inputs={"audio": "AUDIO", "sample_rate": "INT", "parameters": "DICT"},
            default_outputs={"processed_audio": "AUDIO", "features": "TENSOR", "metadata": "DICT"},
            default_parameters={"sample_rate": 22050, "n_fft": 2048},
            complexity_variants={
                NodeComplexity.SIMPLE: {"features": ["basic_filters"]},
                NodeComplexity.MEDIUM: {"features": ["spectral_analysis", "noise_reduction"]},
                NodeComplexity.COMPLEX: {"features": ["source_separation", "enhancement"]},
                NodeComplexity.ADVANCED: {"features": ["real_time_processing", "ml_enhancement"]}
            },
            use_case_examples=["Audio enhancement", "Music separation", "Speech recognition"],
            implementation_notes=["Handle different audio formats", "Implement real-time processing", "Add GPU acceleration"]
        )
        
        # Model Inference Templates
        self.templates[TemplateType.MODEL_INFERENCE] = NodeTemplate(
            template_type=TemplateType.MODEL_INFERENCE,
            name="ModelInference",
            description="Generic ML model inference node",
            base_code=self._get_model_inference_template(),
            required_imports=["torch", "transformers"],
            optional_imports=["onnx", "tensorrt", "openvino"],
            default_inputs={"input_data": "ANY", "model": "MODEL", "config": "DICT"},
            default_outputs={"predictions": "TENSOR", "probabilities": "TENSOR", "metadata": "DICT"},
            default_parameters={"batch_size": 1, "device": "auto"},
            complexity_variants={
                NodeComplexity.SIMPLE: {"optimization": "none", "precision": "fp32"},
                NodeComplexity.MEDIUM: {"optimization": "basic", "precision": "fp16"},
                NodeComplexity.COMPLEX: {"optimization": "advanced", "precision": "mixed"},
                NodeComplexity.ADVANCED: {"optimization": "custom", "precision": "int8"}
            },
            use_case_examples=["Image classification", "Object detection", "Text generation"],
            implementation_notes=["Support multiple model formats", "Implement model caching", "Add batch processing"]
        )
        
        # Video Processing Templates
        self.templates[TemplateType.VIDEO_PROCESSOR] = NodeTemplate(
            template_type=TemplateType.VIDEO_PROCESSOR,
            name="VideoProcessor",
            description="Generic video processing node",
            base_code=self._get_video_processor_template(),
            required_imports=["cv2", "torch", "torchvision"],
            optional_imports=["ffmpeg", "moviepy", "decord"],
            default_inputs={"video": "VIDEO", "fps": "INT", "parameters": "DICT"},
            default_outputs={"processed_video": "VIDEO", "frames": "IMAGE", "metadata": "DICT"},
            default_parameters={"target_fps": 30, "quality": "high"},
            complexity_variants={
                NodeComplexity.SIMPLE: {"processing": ["resize", "crop"]},
                NodeComplexity.MEDIUM: {"processing": ["stabilization", "enhancement"]},
                NodeComplexity.COMPLEX: {"processing": ["object_tracking", "style_transfer"]},
                NodeComplexity.ADVANCED: {"processing": ["real_time", "ml_enhancement"]}
            },
            use_case_examples=["Video enhancement", "Object tracking", "Style transfer"],
            implementation_notes=["Handle different video formats", "Implement memory-efficient processing", "Add GPU acceleration"]
        )
        
        # Data Processing Templates
        self.templates[TemplateType.DATA_PROCESSOR] = NodeTemplate(
            template_type=TemplateType.DATA_PROCESSOR,
            name="DataProcessor",
            description="Generic data processing node",
            base_code=self._get_data_processor_template(),
            required_imports=["pandas", "numpy"],
            optional_imports=["scipy", "sklearn", "plotly"],
            default_inputs={"data": "ANY", "schema": "DICT", "parameters": "DICT"},
            default_outputs={"processed_data": "ANY", "statistics": "DICT", "visualization": "IMAGE"},
            default_parameters={"normalize": True, "handle_missing": "auto"},
            complexity_variants={
                NodeComplexity.SIMPLE: {"operations": ["basic_stats", "filtering"]},
                NodeComplexity.MEDIUM: {"operations": ["transformation", "aggregation"]},
                NodeComplexity.COMPLEX: {"operations": ["feature_engineering", "outlier_detection"]},
                NodeComplexity.ADVANCED: {"operations": ["ml_preprocessing", "custom_pipelines"]}
            },
            use_case_examples=["Data cleaning", "Feature engineering", "Statistical analysis"],
            implementation_notes=["Handle large datasets", "Implement streaming processing", "Add visualization"]
        )
        
        # Custom ML Templates
        self.templates[TemplateType.CUSTOM_ML] = NodeTemplate(
            template_type=TemplateType.CUSTOM_ML,
            name="CustomMLNode",
            description="Custom machine learning node",
            base_code=self._get_custom_ml_template(),
            required_imports=["torch", "torch.nn"],
            optional_imports=["transformers", "diffusers", "accelerate"],
            default_inputs={"input_data": "ANY", "model_config": "DICT", "training_data": "ANY"},
            default_outputs={"output": "ANY", "model_state": "DICT", "metrics": "DICT"},
            default_parameters={"learning_rate": 0.001, "epochs": 10},
            complexity_variants={
                NodeComplexity.SIMPLE: {"architecture": "linear", "training": "basic"},
                NodeComplexity.MEDIUM: {"architecture": "cnn", "training": "standard"},
                NodeComplexity.COMPLEX: {"architecture": "transformer", "training": "advanced"},
                NodeComplexity.ADVANCED: {"architecture": "custom", "training": "distributed"}
            },
            use_case_examples=["Custom model training", "Transfer learning", "Model fine-tuning"],
            implementation_notes=["Support distributed training", "Implement checkpointing", "Add monitoring"]
        )
        
        self.logger.info(f"Initialized {len(self.templates)} enhanced templates")
    
    def get_template_for_specification(self, spec: NodeSpecification) -> Optional[NodeTemplate]:
        """Get the most appropriate template for a node specification"""
        
        # Map node category to template type
        category_mapping = {
            NodeCategory.IMAGE_PROCESSING: TemplateType.IMAGE_PROCESSOR,
            NodeCategory.IMAGE_GENERATION: TemplateType.IMAGE_GENERATOR,
            NodeCategory.IMAGE_ENHANCEMENT: TemplateType.IMAGE_ENHANCER,
            NodeCategory.TEXT_PROCESSING: TemplateType.TEXT_PROCESSOR,
            NodeCategory.TEXT_GENERATION: TemplateType.TEXT_GENERATOR,
            NodeCategory.AUDIO_PROCESSING: TemplateType.AUDIO_PROCESSOR,
            NodeCategory.AUDIO_GENERATION: TemplateType.AUDIO_GENERATOR,
            NodeCategory.VIDEO_PROCESSING: TemplateType.VIDEO_PROCESSOR,
            NodeCategory.MODEL_LOADING: TemplateType.MODEL_LOADER,
            NodeCategory.MODEL_INFERENCE: TemplateType.MODEL_INFERENCE,
            NodeCategory.DATA_PROCESSING: TemplateType.DATA_PROCESSOR,
            NodeCategory.UTILITY: TemplateType.UTILITY,
            NodeCategory.WORKFLOW: TemplateType.WORKFLOW,
            NodeCategory.CUSTOM: TemplateType.CUSTOM_ML
        }
        
        template_type = category_mapping.get(spec.category, TemplateType.CUSTOM_ML)
        
        # Get base template
        template = self.templates.get(template_type)
        if not template:
            # Fallback to custom ML template
            template = self.templates.get(TemplateType.CUSTOM_ML)
        
        return template
    
    def generate_code_from_template(self, spec: NodeSpecification, template: NodeTemplate) -> str:
        """Generate code from template and specification"""
        
        # Get complexity-specific configuration
        complexity_config = template.complexity_variants.get(spec.complexity, {})
        
        # Start with base template
        code = template.base_code
        
        # Replace placeholders with specification data
        replacements = {
            "{{NODE_NAME}}": spec.name,
            "{{NODE_DESCRIPTION}}": spec.description,
            "{{NODE_CATEGORY}}": spec.category.value,
            "{{INPUTS}}": self._format_inputs(spec.inputs),
            "{{OUTPUTS}}": self._format_outputs(spec.outputs),
            "{{PARAMETERS}}": self._format_parameters(spec.parameters),
            "{{REQUIRED_IMPORTS}}": self._format_imports(template.required_imports),
            "{{OPTIONAL_IMPORTS}}": self._format_optional_imports(template.optional_imports),
            "{{COMPLEXITY_CONFIG}}": str(complexity_config),
            "{{IMPLEMENTATION_HINTS}}": self._format_implementation_hints(spec.implementation_hints)
        }
        
        for placeholder, value in replacements.items():
            code = code.replace(placeholder, value)
        
        return code
    
    def _format_inputs(self, inputs: Dict[str, str]) -> str:
        """Format inputs for template replacement"""
        if not inputs:
            return '"input": ("ANY",)'
        
        formatted = []
        for name, type_str in inputs.items():
            formatted.append(f'"{name}": ("{type_str}",)')
        
        return ",\n            ".join(formatted)
    
    def _format_outputs(self, outputs: Dict[str, str]) -> str:
        """Format outputs for template replacement"""
        if not outputs:
            return '"output"'
        
        return ", ".join(f'"{name}"' for name in outputs.keys())
    
    def _format_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format parameters for template replacement"""
        if not parameters:
            return ""
        
        formatted = []
        for name, default_value in parameters.items():
            if isinstance(default_value, str):
                formatted.append(f'"{name}": "{default_value}"')
            else:
                formatted.append(f'"{name}": {default_value}')
        
        return ", ".join(formatted)
    
    def _format_imports(self, imports: List[str]) -> str:
        """Format required imports"""
        return "\n".join(f"import {imp}" for imp in imports)
    
    def _format_optional_imports(self, imports: List[str]) -> str:
        """Format optional imports with try/except"""
        formatted = []
        for imp in imports:
            formatted.append(f"""try:
    import {imp}
    HAS_{imp.upper().replace('-', '_')} = True
except ImportError:
    HAS_{imp.upper().replace('-', '_')} = False""")
        
        return "\n".join(formatted)
    
    def _format_implementation_hints(self, hints: List[str]) -> str:
        """Format implementation hints as comments"""
        if not hints:
            return "# No specific implementation hints"
        
        return "\n        ".join(f"# {hint}" for hint in hints)
    
    # Template code generators
    def _get_image_processor_template(self) -> str:
        return '''{{REQUIRED_IMPORTS}}
{{OPTIONAL_IMPORTS}}

class {{NODE_NAME}}:
    """
    {{NODE_DESCRIPTION}}
    Category: {{NODE_CATEGORY}}
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                {{INPUTS}}
            },
            "optional": {
                "parameters": ("DICT", {"default": {{{PARAMETERS}}}})
            }
        }
    
    RETURN_TYPES = ({{OUTPUTS}},)
    RETURN_NAMES = ({{OUTPUTS}},)
    FUNCTION = "process"
    CATEGORY = "{{NODE_CATEGORY}}"
    
    def process(self, image, parameters=None):
        """
        Process image with enhanced capabilities
        {{IMPLEMENTATION_HINTS}}
        """
        try:
            if parameters is None:
                parameters = {}
            
            # Implementation based on complexity: {{COMPLEXITY_CONFIG}}
            processed_image = self._process_image(image, parameters)
            metadata = self._generate_metadata(image, processed_image, parameters)
            
            return (processed_image, metadata)
            
        except Exception as e:
            print(f"Error in {{NODE_NAME}}: {str(e)}")
            raise e
    
    def _process_image(self, image, parameters):
        """Core image processing logic"""
        # Implement specific processing based on requirements
        return image
    
    def _generate_metadata(self, input_image, output_image, parameters):
        """Generate processing metadata"""
        return {
            "processing_time": 0.0,
            "parameters_used": parameters,
            "input_shape": getattr(input_image, 'shape', None),
            "output_shape": getattr(output_image, 'shape', None)
        }'''
    
    def _get_text_processor_template(self) -> str:
        return '''{{REQUIRED_IMPORTS}}
{{OPTIONAL_IMPORTS}}

class {{NODE_NAME}}:
    """
    {{NODE_DESCRIPTION}}
    Category: {{NODE_CATEGORY}}
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                {{INPUTS}}
            },
            "optional": {
                "parameters": ("DICT", {"default": {{{PARAMETERS}}}})
            }
        }
    
    RETURN_TYPES = ({{OUTPUTS}},)
    RETURN_NAMES = ({{OUTPUTS}},)
    FUNCTION = "process"
    CATEGORY = "{{NODE_CATEGORY}}"
    
    def process(self, text, model=None, parameters=None):
        """
        Process text with NLP capabilities
        {{IMPLEMENTATION_HINTS}}
        """
        try:
            if parameters is None:
                parameters = {}
            
            # Implementation based on complexity: {{COMPLEXITY_CONFIG}}
            processed_text = self._process_text(text, model, parameters)
            embeddings = self._generate_embeddings(text, model, parameters)
            metadata = self._generate_metadata(text, processed_text, parameters)
            
            return (processed_text, embeddings, metadata)
            
        except Exception as e:
            print(f"Error in {{NODE_NAME}}: {str(e)}")
            raise e
    
    def _process_text(self, text, model, parameters):
        """Core text processing logic"""
        # Implement specific text processing
        return text
    
    def _generate_embeddings(self, text, model, parameters):
        """Generate text embeddings"""
        # Implement embedding generation
        return None
    
    def _generate_metadata(self, input_text, output_text, parameters):
        """Generate processing metadata"""
        return {
            "input_length": len(input_text),
            "output_length": len(output_text),
            "parameters_used": parameters
        }'''
    
    def _get_audio_processor_template(self) -> str:
        return '''{{REQUIRED_IMPORTS}}
{{OPTIONAL_IMPORTS}}

class {{NODE_NAME}}:
    """
    {{NODE_DESCRIPTION}}
    Category: {{NODE_CATEGORY}}
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                {{INPUTS}}
            },
            "optional": {
                "parameters": ("DICT", {"default": {{{PARAMETERS}}}})
            }
        }
    
    RETURN_TYPES = ({{OUTPUTS}},)
    RETURN_NAMES = ({{OUTPUTS}},)
    FUNCTION = "process"
    CATEGORY = "{{NODE_CATEGORY}}"
    
    def process(self, audio, sample_rate=22050, parameters=None):
        """
        Process audio with advanced capabilities
        {{IMPLEMENTATION_HINTS}}
        """
        try:
            if parameters is None:
                parameters = {}
            
            # Implementation based on complexity: {{COMPLEXITY_CONFIG}}
            processed_audio = self._process_audio(audio, sample_rate, parameters)
            features = self._extract_features(audio, sample_rate, parameters)
            metadata = self._generate_metadata(audio, processed_audio, parameters)
            
            return (processed_audio, features, metadata)
            
        except Exception as e:
            print(f"Error in {{NODE_NAME}}: {str(e)}")
            raise e
    
    def _process_audio(self, audio, sample_rate, parameters):
        """Core audio processing logic"""
        # Implement specific audio processing
        return audio
    
    def _extract_features(self, audio, sample_rate, parameters):
        """Extract audio features"""
        # Implement feature extraction
        return None
    
    def _generate_metadata(self, input_audio, output_audio, parameters):
        """Generate processing metadata"""
        return {
            "sample_rate": parameters.get("sample_rate", 22050),
            "duration": len(input_audio) / parameters.get("sample_rate", 22050),
            "parameters_used": parameters
        }'''
    
    def _get_model_inference_template(self) -> str:
        return '''{{REQUIRED_IMPORTS}}
{{OPTIONAL_IMPORTS}}

class {{NODE_NAME}}:
    """
    {{NODE_DESCRIPTION}}
    Category: {{NODE_CATEGORY}}
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                {{INPUTS}}
            },
            "optional": {
                "parameters": ("DICT", {"default": {{{PARAMETERS}}}})
            }
        }
    
    RETURN_TYPES = ({{OUTPUTS}},)
    RETURN_NAMES = ({{OUTPUTS}},)
    FUNCTION = "process"
    CATEGORY = "{{NODE_CATEGORY}}"
    
    def process(self, input_data, model, config=None, parameters=None):
        """
        Perform model inference
        {{IMPLEMENTATION_HINTS}}
        """
        try:
            if parameters is None:
                parameters = {}
            if config is None:
                config = {}
            
            # Implementation based on complexity: {{COMPLEXITY_CONFIG}}
            predictions = self._run_inference(input_data, model, config, parameters)
            probabilities = self._calculate_probabilities(predictions, parameters)
            metadata = self._generate_metadata(input_data, predictions, parameters)
            
            return (predictions, probabilities, metadata)
            
        except Exception as e:
            print(f"Error in {{NODE_NAME}}: {str(e)}")
            raise e
    
    def _run_inference(self, input_data, model, config, parameters):
        """Core inference logic"""
        # Implement model inference
        return input_data
    
    def _calculate_probabilities(self, predictions, parameters):
        """Calculate prediction probabilities"""
        # Implement probability calculation
        return None
    
    def _generate_metadata(self, input_data, predictions, parameters):
        """Generate inference metadata"""
        return {
            "inference_time": 0.0,
            "model_info": "unknown",
            "parameters_used": parameters
        }'''
    
    def _get_video_processor_template(self) -> str:
        return '''{{REQUIRED_IMPORTS}}
{{OPTIONAL_IMPORTS}}

class {{NODE_NAME}}:
    """
    {{NODE_DESCRIPTION}}
    Category: {{NODE_CATEGORY}}
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                {{INPUTS}}
            },
            "optional": {
                "parameters": ("DICT", {"default": {{{PARAMETERS}}}})
            }
        }
    
    RETURN_TYPES = ({{OUTPUTS}},)
    RETURN_NAMES = ({{OUTPUTS}},)
    FUNCTION = "process"
    CATEGORY = "{{NODE_CATEGORY}}"
    
    def process(self, video, fps=30, parameters=None):
        """
        Process video with advanced capabilities
        {{IMPLEMENTATION_HINTS}}
        """
        try:
            if parameters is None:
                parameters = {}
            
            # Implementation based on complexity: {{COMPLEXITY_CONFIG}}
            processed_video = self._process_video(video, fps, parameters)
            frames = self._extract_frames(video, fps, parameters)
            metadata = self._generate_metadata(video, processed_video, parameters)
            
            return (processed_video, frames, metadata)
            
        except Exception as e:
            print(f"Error in {{NODE_NAME}}: {str(e)}")
            raise e
    
    def _process_video(self, video, fps, parameters):
        """Core video processing logic"""
        # Implement specific video processing
        return video
    
    def _extract_frames(self, video, fps, parameters):
        """Extract video frames"""
        # Implement frame extraction
        return None
    
    def _generate_metadata(self, input_video, output_video, parameters):
        """Generate processing metadata"""
        return {
            "fps": parameters.get("target_fps", 30),
            "frame_count": 0,
            "parameters_used": parameters
        }'''
    
    def _get_data_processor_template(self) -> str:
        return '''{{REQUIRED_IMPORTS}}
{{OPTIONAL_IMPORTS}}

class {{NODE_NAME}}:
    """
    {{NODE_DESCRIPTION}}
    Category: {{NODE_CATEGORY}}
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                {{INPUTS}}
            },
            "optional": {
                "parameters": ("DICT", {"default": {{{PARAMETERS}}}})
            }
        }
    
    RETURN_TYPES = ({{OUTPUTS}},)
    RETURN_NAMES = ({{OUTPUTS}},)
    FUNCTION = "process"
    CATEGORY = "{{NODE_CATEGORY}}"
    
    def process(self, data, schema=None, parameters=None):
        """
        Process data with advanced analytics
        {{IMPLEMENTATION_HINTS}}
        """
        try:
            if parameters is None:
                parameters = {}
            if schema is None:
                schema = {}
            
            # Implementation based on complexity: {{COMPLEXITY_CONFIG}}
            processed_data = self._process_data(data, schema, parameters)
            statistics = self._calculate_statistics(data, processed_data, parameters)
            visualization = self._create_visualization(data, processed_data, parameters)
            
            return (processed_data, statistics, visualization)
            
        except Exception as e:
            print(f"Error in {{NODE_NAME}}: {str(e)}")
            raise e
    
    def _process_data(self, data, schema, parameters):
        """Core data processing logic"""
        # Implement specific data processing
        return data
    
    def _calculate_statistics(self, input_data, output_data, parameters):
        """Calculate data statistics"""
        # Implement statistical analysis
        return {}
    
    def _create_visualization(self, input_data, output_data, parameters):
        """Create data visualization"""
        # Implement visualization
        return None'''
    
    def _get_custom_ml_template(self) -> str:
        return '''{{REQUIRED_IMPORTS}}
{{OPTIONAL_IMPORTS}}

class {{NODE_NAME}}:
    """
    {{NODE_DESCRIPTION}}
    Category: {{NODE_CATEGORY}}
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                {{INPUTS}}
            },
            "optional": {
                "parameters": ("DICT", {"default": {{{PARAMETERS}}}})
            }
        }
    
    RETURN_TYPES = ({{OUTPUTS}},)
    RETURN_NAMES = ({{OUTPUTS}},)
    FUNCTION = "process"
    CATEGORY = "{{NODE_CATEGORY}}"
    
    def process(self, input_data, model_config=None, training_data=None, parameters=None):
        """
        Custom ML processing
        {{IMPLEMENTATION_HINTS}}
        """
        try:
            if parameters is None:
                parameters = {}
            if model_config is None:
                model_config = {}
            
            # Implementation based on complexity: {{COMPLEXITY_CONFIG}}
            output = self._ml_process(input_data, model_config, training_data, parameters)
            model_state = self._get_model_state(parameters)
            metrics = self._calculate_metrics(input_data, output, parameters)
            
            return (output, model_state, metrics)
            
        except Exception as e:
            print(f"Error in {{NODE_NAME}}: {str(e)}")
            raise e
    
    def _ml_process(self, input_data, model_config, training_data, parameters):
        """Core ML processing logic"""
        # Implement custom ML logic
        return input_data
    
    def _get_model_state(self, parameters):
        """Get current model state"""
        # Implement model state tracking
        return {}
    
    def _calculate_metrics(self, input_data, output, parameters):
        """Calculate performance metrics"""
        # Implement metrics calculation
        return {}'''
