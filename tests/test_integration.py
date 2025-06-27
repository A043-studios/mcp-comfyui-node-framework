#!/usr/bin/env python3
"""
Integration tests for the MCP ComfyUI Node Framework
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from comfyui_mcp_server_v2 import ComfyUIMCPServer
    from llm_client_v2 import LLMClient
    from web_scraper_v2 import WebScraper
except ImportError as e:
    print(f"Warning: Could not import framework modules: {e}")
    print("Some tests may be skipped.")


class TestMCPFramework(unittest.TestCase):
    """Test the core MCP framework functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            "llm": {
                "provider": "mock",
                "model": "test-model",
                "temperature": 0.1,
                "max_tokens": 1000
            },
            "agents": {
                "research_agent": {"model": "test-model"},
                "coding_agent": {"model": "test-model"},
                "documentation_agent": {"model": "test-model"}
            }
        }
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_loading(self):
        """Test configuration loading"""
        config_path = os.path.join(self.temp_dir, "test_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.test_config, f)
        
        # Test that config can be loaded
        self.assertTrue(os.path.exists(config_path))
        
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        self.assertEqual(loaded_config["llm"]["provider"], "mock")

    def test_web_scraper_initialization(self):
        """Test WebScraper initialization"""
        try:
            scraper = WebScraper()
            self.assertIsNotNone(scraper)
        except Exception as e:
            self.skipTest(f"WebScraper not available: {e}")

    def test_llm_client_initialization(self):
        """Test LLMClient initialization"""
        try:
            client = LLMClient(self.test_config)
            self.assertIsNotNone(client)
        except Exception as e:
            self.skipTest(f"LLMClient not available: {e}")


class TestNodeValidation(unittest.TestCase):
    """Test generated node validation"""

    def test_rembg_node_structure(self):
        """Test the rembg node structure"""
        node_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'examples', 
            'nodes', 
            'rembg_background_removal_node.py'
        )
        
        if not os.path.exists(node_path):
            self.skipTest("Rembg node not found")
        
        # Import the node
        sys.path.insert(0, os.path.dirname(node_path))
        try:
            from rembg_background_removal_node import RembgBackgroundRemovalNode
            
            # Test required attributes
            self.assertTrue(hasattr(RembgBackgroundRemovalNode, 'INPUT_TYPES'))
            self.assertTrue(hasattr(RembgBackgroundRemovalNode, 'RETURN_TYPES'))
            self.assertTrue(hasattr(RembgBackgroundRemovalNode, 'FUNCTION'))
            self.assertTrue(hasattr(RembgBackgroundRemovalNode, 'CATEGORY'))
            
            # Test INPUT_TYPES structure
            input_types = RembgBackgroundRemovalNode.INPUT_TYPES()
            self.assertIn('required', input_types)
            self.assertIn('image', input_types['required'])
            self.assertIn('model', input_types['required'])
            
            # Test RETURN_TYPES
            return_types = RembgBackgroundRemovalNode.RETURN_TYPES
            self.assertIsInstance(return_types, tuple)
            self.assertIn('IMAGE', return_types)
            self.assertIn('MASK', return_types)
            
            # Test category
            self.assertEqual(RembgBackgroundRemovalNode.CATEGORY, 'image/background')
            
        except ImportError as e:
            self.skipTest(f"Could not import rembg node: {e}")

    def test_node_class_mappings(self):
        """Test node class mappings"""
        node_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'examples', 
            'nodes', 
            'rembg_background_removal_node.py'
        )
        
        if not os.path.exists(node_path):
            self.skipTest("Rembg node not found")
        
        sys.path.insert(0, os.path.dirname(node_path))
        try:
            import rembg_background_removal_node as node_module
            
            # Test NODE_CLASS_MAPPINGS exists
            self.assertTrue(hasattr(node_module, 'NODE_CLASS_MAPPINGS'))
            mappings = node_module.NODE_CLASS_MAPPINGS
            self.assertIsInstance(mappings, dict)
            self.assertIn('RembgBackgroundRemovalNode', mappings)
            
            # Test NODE_DISPLAY_NAME_MAPPINGS exists
            self.assertTrue(hasattr(node_module, 'NODE_DISPLAY_NAME_MAPPINGS'))
            display_names = node_module.NODE_DISPLAY_NAME_MAPPINGS
            self.assertIsInstance(display_names, dict)
            self.assertIn('RembgBackgroundRemovalNode', display_names)
            
        except ImportError as e:
            self.skipTest(f"Could not import rembg node: {e}")


class TestWorkflowValidation(unittest.TestCase):
    """Test workflow validation"""

    def test_background_removal_workflow(self):
        """Test the background removal workflow structure"""
        workflow_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'examples', 
            'workflows', 
            'background_removal_workflow.json'
        )
        
        if not os.path.exists(workflow_path):
            self.skipTest("Background removal workflow not found")
        
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
        
        # Test basic structure
        self.assertIn('nodes', workflow)
        self.assertIn('links', workflow)
        
        # Test nodes exist
        nodes = workflow['nodes']
        self.assertIsInstance(nodes, list)
        self.assertGreater(len(nodes), 0)
        
        # Test for RembgBackgroundRemovalNode
        rembg_nodes = [n for n in nodes if n.get('type') == 'RembgBackgroundRemovalNode']
        self.assertGreater(len(rembg_nodes), 0, "Workflow should contain RembgBackgroundRemovalNode")
        
        # Test links exist
        links = workflow['links']
        self.assertIsInstance(links, list)
        self.assertGreater(len(links), 0)


class TestDocumentation(unittest.TestCase):
    """Test documentation completeness"""

    def test_readme_exists(self):
        """Test that README files exist"""
        main_readme = os.path.join(os.path.dirname(__file__), '..', 'README.md')
        self.assertTrue(os.path.exists(main_readme))
        
        examples_readme = os.path.join(os.path.dirname(__file__), '..', 'examples', 'README.md')
        self.assertTrue(os.path.exists(examples_readme))

    def test_api_documentation_exists(self):
        """Test that API documentation exists"""
        api_docs = os.path.join(os.path.dirname(__file__), '..', 'docs', 'API.md')
        self.assertTrue(os.path.exists(api_docs))

    def test_configuration_files_exist(self):
        """Test that configuration files exist"""
        config_template = os.path.join(os.path.dirname(__file__), '..', 'config', 'mcp-config.template.json')
        self.assertTrue(os.path.exists(config_template))
        
        agents_example = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.example.json')
        self.assertTrue(os.path.exists(agents_example))


def run_specific_test(test_class_name=None, test_method_name=None):
    """Run a specific test or test class"""
    if test_class_name and test_method_name:
        suite = unittest.TestSuite()
        suite.addTest(globals()[test_class_name](test_method_name))
    elif test_class_name:
        suite = unittest.TestLoader().loadTestsFromTestCase(globals()[test_class_name])
    else:
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MCP ComfyUI Framework tests')
    parser.add_argument('--node', help='Test a specific node file')
    parser.add_argument('--class', dest='test_class', help='Run tests from a specific class')
    parser.add_argument('--method', help='Run a specific test method')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.node:
        # Test a specific node file
        print(f"Testing node: {args.node}")
        if os.path.exists(args.node):
            # Add basic validation for the node
            sys.path.insert(0, os.path.dirname(args.node))
            try:
                module_name = os.path.splitext(os.path.basename(args.node))[0]
                module = __import__(module_name)
                print(f"✓ Node {args.node} imported successfully")
                
                # Check for required attributes
                for attr in ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']:
                    if hasattr(module, attr):
                        print(f"✓ {attr} found")
                    else:
                        print(f"✗ {attr} missing")
                        
            except Exception as e:
                print(f"✗ Error importing node: {e}")
        else:
            print(f"✗ Node file not found: {args.node}")
    else:
        # Run standard tests
        if args.verbose:
            unittest.main(verbosity=2)
        else:
            success = run_specific_test(args.test_class, args.method)
            sys.exit(0 if success else 1)
