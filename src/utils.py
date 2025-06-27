"""
Utility functions for MCP Multi-Agent Framework
"""

import os
import json
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration - SAFE VERSION for read-only environments"""
    # Return a simple logger that only uses console output
    # This completely bypasses any file system operations

    logger = logging.getLogger('mcp_framework')

    # Only add console handler if not already present
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(console_handler)
        logger.setLevel(getattr(logging, level.upper()))

    return logger


def create_directory(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist, handle read-only file systems gracefully"""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except (OSError, PermissionError) as e:
        # In read-only environments (like Augment), we can't create directories
        # Return the path anyway - the calling code should handle this gracefully
        # Use basic logging to avoid recursive setup_logging calls
        import logging
        logger = logging.getLogger('mcp_framework')
        logger.warning(f"Cannot create directory {path} (read-only environment): {e}")
        return path


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save data as JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_text(text: str, file_path: Union[str, Path]) -> None:
    """Save text to file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)


def load_text(file_path: Union[str, Path]) -> str:
    """Load text from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def validate_input(input_source: str) -> Dict[str, Any]:
    """
    Validate input source and determine type
    
    Args:
        input_source: Input source (URL, file path, or directory)
        
    Returns:
        Validation result with type and validity
    """
    # Check if it's a URL
    parsed = urlparse(input_source)
    if parsed.scheme in ['http', 'https']:
        if 'arxiv.org' in parsed.netloc:
            return {"valid": True, "type": "arxiv", "url": input_source}
        elif 'github.com' in parsed.netloc:
            return {"valid": True, "type": "github", "url": input_source}
        else:
            return {"valid": True, "type": "url", "url": input_source}
    
    # Check if it's a local file or directory
    if os.path.exists(input_source):
        if os.path.isfile(input_source):
            if input_source.lower().endswith('.pdf'):
                return {"valid": True, "type": "pdf", "path": input_source}
            else:
                return {"valid": True, "type": "file", "path": input_source}
        elif os.path.isdir(input_source):
            return {"valid": True, "type": "directory", "path": input_source}
    
    return {"valid": False, "error": f"Invalid input source: {input_source}"}


def create_output_structure(output_dir: str) -> Dict[str, str]:
    """
    Create standard output directory structure

    Args:
        output_dir: Base output directory

    Returns:
        Dictionary of created directories
    """
    base_dir = Path(output_dir)

    directories = {
        "base": str(base_dir),
        "nodes": str(base_dir / "nodes"),
        "docs": str(base_dir / "docs"),
        "examples": str(base_dir / "examples"),
        "tests": str(base_dir / "tests"),
        "logs": str(base_dir / "logs"),
        "artifacts": str(base_dir / "artifacts")
    }

    # Create all directories (gracefully handle read-only file systems)
    import logging
    logger = logging.getLogger('mcp_framework')
    created_count = 0
    for dir_name, dir_path in directories.items():
        try:
            create_directory(dir_path)
            created_count += 1
        except Exception as e:
            logger.warning(f"Could not create {dir_name} directory {dir_path}: {e}")

    if created_count > 0:
        logger.info(f"Successfully created {created_count}/{len(directories)} directories")
    else:
        logger.warning("Running in read-only environment - no directories created")

    return directories


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from PDF file
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
    """
    try:
        # Try using PyPDF2 first
        import PyPDF2
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        return text
        
    except ImportError:
        # Fallback to pdfplumber if available
        try:
            import pdfplumber
            
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            return text
            
        except ImportError:
            # Last resort: try using subprocess with pdftotext
            try:
                result = subprocess.run(
                    ['pdftotext', pdf_path, '-'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return result.stdout
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise ImportError(
                    "No PDF processing library available. "
                    "Please install PyPDF2, pdfplumber, or pdftotext"
                )


def clone_repository(repo_url: str, target_dir: str) -> None:
    """
    Clone Git repository
    
    Args:
        repo_url: Repository URL
        target_dir: Target directory for cloning
    """
    try:
        # Remove existing directory if it exists
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        # Clone repository
        subprocess.run(
            ['git', 'clone', repo_url, target_dir],
            check=True,
            capture_output=True
        )
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone repository: {e}")
    except FileNotFoundError:
        raise RuntimeError("Git is not installed or not in PATH")


def run_command(command: List[str], cwd: Optional[str] = None, timeout: int = 300) -> Dict[str, Any]:
    """
    Run shell command and return result
    
    Args:
        command: Command to run as list of strings
        cwd: Working directory
        timeout: Timeout in seconds
        
    Returns:
        Command execution result
    """
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": " ".join(command)
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Command timed out",
            "command": " ".join(command)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "command": " ".join(command)
        }


def find_files(directory: str, pattern: str = "*", recursive: bool = True) -> List[str]:
    """
    Find files matching pattern in directory
    
    Args:
        directory: Directory to search
        pattern: File pattern (glob style)
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    from glob import glob
    
    if recursive:
        search_pattern = os.path.join(directory, "**", pattern)
        return glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(directory, pattern)
        return glob(search_pattern)


def copy_file(src: str, dst: str) -> None:
    """Copy file from source to destination"""
    # Create destination directory if it doesn't exist
    dst_dir = os.path.dirname(dst)
    if dst_dir:
        create_directory(dst_dir)
    
    shutil.copy2(src, dst)


def copy_directory(src: str, dst: str) -> None:
    """Copy directory from source to destination"""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    return os.path.getsize(file_path)


def get_directory_size(directory: str) -> int:
    """Get total size of directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(file_path)
            except (OSError, FileNotFoundError):
                pass  # Skip files that can't be accessed
    return total_size


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def validate_python_syntax(code: str) -> Dict[str, Any]:
    """
    Validate Python code syntax
    
    Args:
        code: Python code string
        
    Returns:
        Validation result
    """
    try:
        compile(code, '<string>', 'exec')
        return {"valid": True}
    except SyntaxError as e:
        return {
            "valid": False,
            "error": str(e),
            "line": e.lineno,
            "offset": e.offset
        }


def extract_imports(code: str) -> List[str]:
    """
    Extract import statements from Python code
    
    Args:
        code: Python code string
        
    Returns:
        List of import statements
    """
    import ast
    
    try:
        tree = ast.parse(code)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        
        return imports
        
    except SyntaxError:
        return []


def generate_requirements_txt(imports: List[str], output_file: str) -> None:
    """
    Generate requirements.txt from import list
    
    Args:
        imports: List of imported modules
        output_file: Output requirements.txt file path
    """
    # Map common imports to package names
    package_mapping = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'requests': 'requests',
        'numpy': 'numpy',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'tensorflow': 'tensorflow',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy'
    }
    
    packages = set()
    for imp in imports:
        base_module = imp.split('.')[0]
        if base_module in package_mapping:
            packages.add(package_mapping[base_module])
        elif not base_module.startswith('_') and base_module not in ['os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'typing', 'abc']:
            packages.add(base_module)
    
    # Write requirements.txt
    with open(output_file, 'w') as f:
        for package in sorted(packages):
            f.write(f"{package}\n")


def validate_comfyui_node(node_code: str) -> Dict[str, Any]:
    """
    Validate ComfyUI node structure
    
    Args:
        node_code: Node code string
        
    Returns:
        Validation result
    """
    required_attributes = ['INPUT_TYPES', 'RETURN_TYPES', 'FUNCTION']
    required_methods = ['INPUT_TYPES']
    
    validation = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check syntax first
    syntax_check = validate_python_syntax(node_code)
    if not syntax_check["valid"]:
        validation["valid"] = False
        validation["errors"].append(f"Syntax error: {syntax_check['error']}")
        return validation
    
    # Check for required attributes and methods
    import ast
    
    try:
        tree = ast.parse(node_code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check class attributes
                class_attributes = []
                class_methods = []
                
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                class_attributes.append(target.id)
                    elif isinstance(item, ast.FunctionDef):
                        class_methods.append(item.name)
                
                # Validate required attributes
                for attr in required_attributes:
                    if attr not in class_attributes:
                        validation["errors"].append(f"Missing required attribute: {attr}")
                        validation["valid"] = False
                
                # Validate required methods
                for method in required_methods:
                    if method not in class_methods:
                        validation["errors"].append(f"Missing required method: {method}")
                        validation["valid"] = False
                
                # Check for INPUT_TYPES as classmethod
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES":
                        if not any(isinstance(decorator, ast.Name) and decorator.id == "classmethod" 
                                 for decorator in item.decorator_list):
                            validation["warnings"].append("INPUT_TYPES should be a classmethod")
    
    except Exception as e:
        validation["valid"] = False
        validation["errors"].append(f"Analysis error: {str(e)}")
    
    return validation
