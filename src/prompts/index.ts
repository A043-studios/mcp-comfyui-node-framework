import { Prompt } from '@modelcontextprotocol/sdk/types.js';

export interface PromptMessage {
  role: 'user' | 'assistant';
  content: string;
}

/**
 * Generate node from research paper - guided workflow
 */
export function generateNodeFromPaper(paperUrl: string, focusArea?: string): PromptMessage[] {
  return [
    {
      role: 'user',
      content: `I want to generate a ComfyUI node from this research paper: ${paperUrl}

${focusArea ? `Focus area: ${focusArea}` : 'Please help me identify the best focus areas from the paper.'}

Can you guide me through the process step by step?`,
    },
    {
      role: 'assistant',
      content: `I'll help you generate a ComfyUI node from that research paper! Let me break this down into a systematic process:

## Step-by-Step Node Generation Process

### 1. **Research Analysis** 📚
First, I'll analyze the paper content to understand:
- Key algorithms and techniques
- Implementation requirements
- Potential ComfyUI applications

### 2. **Node Design** 🎯
Based on the research, I'll help you design:
- Node input/output structure
- Parameter configuration
- Processing workflow

### 3. **Code Generation** 💻
I'll generate:
- Complete ComfyUI node implementation
- Proper error handling and validation
- Optimized code structure

### 4. **Testing & Documentation** ✅
Finally, I'll create:
- Comprehensive test suite
- Usage documentation
- Installation instructions

Let me start by analyzing the paper content. I'll use the **web_scraper_scrape** tool to extract and analyze the research content.

Would you like me to proceed with the analysis?`,
    },
    {
      role: 'user',
      content: `Yes, please proceed with analyzing the paper. Extract the key information and identify the most suitable approaches for ComfyUI node implementation.`,
    },
    {
      role: 'assistant',
      content: `Perfect! I'll analyze the paper now using our research tools.

**Next Steps:**
1. Extract paper content and metadata
2. Identify key algorithms and techniques  
3. Analyze implementation complexity
4. Suggest optimal node structure

${focusArea ? `I'll pay special attention to **${focusArea}** aspects as you requested.` : 'I\'ll identify the most promising areas for ComfyUI integration.'}

After the analysis, I'll provide you with:
- **Summary** of key findings
- **Recommended approach** for node implementation
- **Technical requirements** and dependencies
- **Estimated complexity** and development time

Let me start the analysis now...`,
    },
  ];
}

/**
 * Debug node generation - troubleshooting workflow
 */
export function debugNodeGeneration(executionId?: string, errorDescription?: string): PromptMessage[] {
  return [
    {
      role: 'user',
      content: `I'm having issues with ComfyUI node generation.

${executionId ? `Execution ID: ${executionId}` : ''}
${errorDescription ? `Error: ${errorDescription}` : ''}

Can you help me debug this issue?`,
    },
    {
      role: 'assistant',
      content: `I'll help you debug the node generation issue! Let me systematically investigate what went wrong.

## Debugging Approach 🔍

### 1. **Information Gathering**
${executionId ? `I'll check the logs and artifacts for execution: **${executionId}**` : 'I\'ll help you identify the issue step by step'}

### 2. **Error Analysis**
I'll analyze:
- Execution logs for error patterns
- Generated artifacts (if any)
- Input validation issues
- Agent-specific failures

### 3. **Root Cause Identification**
Common issues I'll check:
- **Input Source Problems**: URL accessibility, format issues
- **Agent Configuration**: Timeout, dependency issues
- **Resource Constraints**: Memory, disk space, network
- **Code Generation**: Syntax errors, missing dependencies

### 4. **Solution Recommendations**
I'll provide:
- Specific fixes for identified issues
- Alternative approaches if needed
- Prevention strategies for future runs

${executionId ? 'Let me check the execution logs and artifacts first...' : 'Let\'s start by identifying what type of issue you\'re experiencing.'}

**Debugging Checklist:**
- ✅ Input source accessibility
- ✅ Agent configuration
- ✅ Output directory permissions  
- ✅ Dependencies and environment
- ✅ Network connectivity

${executionId ? 'I\'ll use the **logs://' + executionId + '** resource to examine what happened.' : 'Please provide more details about when the error occurred and what you were trying to generate.'}`,
    },
    {
      role: 'user',
      content: `${executionId ? 'Please check the logs and artifacts to identify the issue.' : 'The error occurred during the coding phase. The research phase completed successfully, but the coding agent failed to generate the node.'}`,
    },
    {
      role: 'assistant',
      content: `${executionId ? 
        `I'll examine the execution details now. Let me check both the logs and artifacts to understand what happened during the workflow.

**Checking:**
- Execution logs for error messages
- Agent artifacts to see what was generated
- Failure points in the pipeline

This will help me identify whether it was a configuration issue, resource problem, or code generation error.` :
        
        `A coding phase failure after successful research suggests the issue is likely in the code generation process. Let me help you diagnose this:

**Common Coding Agent Issues:**
1. **Research Data Format**: The research output might not be in the expected format
2. **Node Specification**: Missing or invalid node specifications
3. **Template Issues**: Problems with code templates or patterns
4. **Dependency Conflicts**: Missing or incompatible dependencies

**Immediate Actions:**
1. Check the research agent output format
2. Validate the node specification parameters
3. Review any error messages from the coding agent
4. Verify the target output directory is writable

Would you like me to run a validation check on your research results and node specifications?`}`,
    },
  ];
}

/**
 * Optimize workflow - performance improvement guidance
 */
export function optimizeWorkflow(currentAgents?: string, performanceIssues?: string): PromptMessage[] {
  return [
    {
      role: 'user',
      content: `I want to optimize my ComfyUI node generation workflow for better performance.

${currentAgents ? `Current agents: ${currentAgents}` : ''}
${performanceIssues ? `Performance issues: ${performanceIssues}` : ''}

How can I improve the efficiency and quality of the generation process?`,
    },
    {
      role: 'assistant',
      content: `Excellent question! Workflow optimization can significantly improve both speed and quality. Let me analyze your current setup and provide targeted recommendations.

## Workflow Optimization Strategy 🚀

### 1. **Current Setup Analysis**
${currentAgents ? `Your current pipeline: **${currentAgents}**` : 'I\'ll help you design an optimal agent pipeline'}

${performanceIssues ? `Addressing: **${performanceIssues}**` : 'I\'ll identify potential bottlenecks and optimization opportunities'}

### 2. **Performance Optimization Areas**

#### **Agent Pipeline Optimization** ⚡
- **Sequential vs Parallel**: Some agents can run in parallel
- **Agent Ordering**: Optimal sequence for dependencies
- **Selective Execution**: Skip unnecessary agents for specific use cases

#### **Resource Optimization** 💾
- **Memory Management**: Efficient artifact handling
- **Caching Strategy**: Reuse research results
- **Concurrent Limits**: Balance speed vs resource usage

#### **Quality vs Speed Trade-offs** ⚖️
- **Quality Levels**: Draft → Development → Production
- **Incremental Updates**: Build on previous results
- **Smart Defaults**: Optimize common use cases

### 3. **Recommended Optimizations**

${currentAgents ? getSpecificOptimizations(currentAgents) : getGeneralOptimizations()}

### 4. **Custom Workflow Creation**
I can help you create optimized workflow templates for:
- **Rapid Prototyping**: Research + Coding only
- **Production Ready**: Full pipeline with all quality checks
- **Research Focus**: Deep analysis with minimal code generation
- **Batch Processing**: Optimized for multiple papers

Would you like me to create a custom optimized workflow based on your specific needs?`,
    },
    {
      role: 'user',
      content: `${currentAgents ? 
        'Please analyze my current setup and suggest specific optimizations.' : 
        'I\'d like to create a custom workflow optimized for rapid prototyping during research phases.'}`,
    },
    {
      role: 'assistant',
      content: `${currentAgents ?
        `Perfect! Let me analyze your current agent configuration and provide specific optimization recommendations.

**Current Pipeline Analysis:**
${analyzeCurrentPipeline(currentAgents)}

**Optimization Recommendations:**
${getOptimizationRecommendations(currentAgents, performanceIssues)}

**Implementation Steps:**
1. Use the **workflow_orchestrator_template** tool to create an optimized workflow
2. Configure parallel execution where possible
3. Set appropriate timeouts and retry policies
4. Enable result caching for repeated research sources

Would you like me to create the optimized workflow configuration now?` :

        `Excellent choice! Rapid prototyping workflows are perfect for research phases. Let me design an optimized configuration for you.

**Rapid Prototyping Workflow Design:**

**Agents:** Research + Coding (minimal pipeline)
**Execution:** Sequential (for dependency flow)
**Quality Level:** Development (faster iteration)
**Timeouts:** Reduced for quick feedback

**Key Optimizations:**
- ⚡ **Skip Testing**: Focus on proof-of-concept
- 📝 **Minimal Documentation**: Basic README only  
- 🔄 **Fast Iteration**: Enable quick modifications
- 💾 **Cache Research**: Reuse analysis results
- 🎯 **Focus Areas**: Specify to reduce processing time

**Estimated Time:** 15-30 minutes (vs 1-2 hours for full pipeline)

I'll create this optimized workflow template using the **workflow_orchestrator_template** tool. This will give you a reusable configuration for rapid prototyping.`}`,
    },
  ];
}

/**
 * Best practices guidance - comprehensive development workflow
 */
export function bestPracticesGuidance(nodeType?: string, experience?: string): PromptMessage[] {
  return [
    {
      role: 'user',
      content: `I want to learn the best practices for ComfyUI node development using this framework.

${nodeType ? `Node type: ${nodeType}` : ''}
${experience ? `My experience level: ${experience}` : ''}

What should I know to create high-quality, professional nodes?`,
    },
    {
      role: 'assistant',
      content: `Great question! Following best practices is crucial for creating professional, maintainable ComfyUI nodes. Let me provide you with comprehensive guidance.

## ComfyUI Node Development Best Practices 🏆

### 1. **Planning & Design** 📋

#### **Before You Start:**
- 📚 **Research Thoroughly**: Understand the underlying algorithms
- 🎯 **Define Clear Scope**: What exactly will your node do?
- 👥 **Consider Users**: Who will use this and how?
- 🔄 **Plan for Iteration**: Design for future improvements

#### **Node Design Principles:**
- **Single Responsibility**: One node, one clear purpose
- **Intuitive Interface**: Self-explanatory parameters
- **Consistent Naming**: Follow ComfyUI conventions
- **Proper Categories**: Organize nodes logically

### 2. **Implementation Best Practices** 💻

#### **Code Quality:**
\`\`\`python
# ✅ Good: Clear, documented, validated
class WellDesignedNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.1,
                    "tooltip": "Processing strength factor"
                }),
            }
        }
    
    def process(self, image, strength):
        # Validate inputs
        if image is None:
            raise ValueError("Image input is required")
        
        # Process with error handling
        try:
            result = self._apply_processing(image, strength)
            return (result,)
        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")
\`\`\`

#### **Performance Optimization:**
- 🚀 **Memory Efficient**: Minimize tensor copying
- ⚡ **GPU Optimized**: Use appropriate device placement
- 🔄 **Batch Support**: Handle multiple images efficiently
- 📊 **Progress Tracking**: For long operations

### 3. **Testing Strategy** ✅

${experience === 'beginner' ? getBeginnerTestingGuidance() : getAdvancedTestingGuidance()}

### 4. **Documentation Standards** 📖

- **README**: Clear installation and usage instructions
- **API Docs**: Complete parameter documentation
- **Examples**: Working code samples
- **Troubleshooting**: Common issues and solutions

### 5. **Deployment & Maintenance** 🚀

- **Version Control**: Proper Git workflow
- **CI/CD Pipeline**: Automated testing and deployment
- **User Feedback**: Channels for bug reports and features
- **Regular Updates**: Keep dependencies current

${nodeType ? getNodeTypeSpecificGuidance(nodeType) : ''}

Would you like me to dive deeper into any of these areas, or shall I help you apply these practices to your specific node development?`,
    },
  ];
}

// Helper methods for prompt generation
function getSpecificOptimizations(agents: string): string {
  const agentList = agents.split(',').map(a => a.trim());
  let optimizations = '';
  
  if (agentList.includes('research') && agentList.includes('coding')) {
    optimizations += '- **Cache Research Results**: Reuse analysis for multiple node variants\n';
  }
  
  if (agentList.includes('testing') && agentList.includes('documentation')) {
    optimizations += '- **Parallel Execution**: Run testing and documentation simultaneously\n';
  }
  
  if (agentList.length > 3) {
    optimizations += '- **Selective Execution**: Create focused workflows for specific needs\n';
  }
  
  return optimizations || '- **Sequential Optimization**: Optimize agent ordering for your pipeline\n';
}

function getGeneralOptimizations(): string {
  return `- **Agent Selection**: Choose only necessary agents for your use case
- **Quality Levels**: Use 'development' for iteration, 'production' for final
- **Parallel Processing**: Enable for independent agents
- **Caching**: Reuse research results across multiple generations
- **Focus Areas**: Specify to reduce processing scope`;
}

function analyzeCurrentPipeline(agents: string): string {
  const agentList = agents.split(',').map(a => a.trim());
  return `Pipeline: ${agentList.join(' → ')}
Estimated time: ${estimateTime(agentList)}
Parallelization potential: ${getParallelizationPotential(agentList)}`;
}

function getOptimizationRecommendations(agents: string, issues?: string): string {
  let recommendations = '';
  
  if (issues?.includes('slow')) {
    recommendations += '- Enable parallel execution for testing + documentation\n';
    recommendations += '- Use "development" quality level for faster iteration\n';
  }
  
  if (issues?.includes('memory')) {
    recommendations += '- Reduce concurrent agent execution\n';
    recommendations += '- Enable artifact cleanup after each agent\n';
  }
  
  return recommendations || '- Optimize agent ordering for better dependency flow\n- Enable result caching for repeated sources';
}

function estimateTime(agents: string[]): string {
  const timeMap: Record<string, number> = {
    research: 5,
    coding: 10,
    testing: 5,
    documentation: 3,
    devops: 2,
  };
  
  const totalMinutes = agents.reduce((sum, agent) => sum + (timeMap[agent] || 5), 0);
  return `${totalMinutes} minutes`;
}

function getParallelizationPotential(agents: string[]): string {
  if (agents.includes('testing') && agents.includes('documentation')) {
    return 'High (testing + documentation can run in parallel)';
  }
  return 'Medium (some agents can be parallelized)';
}

function getBeginnerTestingGuidance(): string {
  return `#### **Start Simple:**
- **Basic Functionality**: Does the node work with valid inputs?
- **Input Validation**: Test with invalid/missing inputs
- **Edge Cases**: Empty images, extreme parameter values
- **Integration**: Works in ComfyUI workflow

#### **Use Framework Tools:**
- **testing_agent_generate**: Creates comprehensive test suites
- **testing_agent_execute**: Runs tests and provides coverage
- **testing_agent_validate**: Checks test quality`;
}

function getAdvancedTestingGuidance(): string {
  return `#### **Comprehensive Testing:**
- **Unit Tests**: Individual method testing
- **Integration Tests**: Full workflow testing  
- **Performance Tests**: Memory and speed benchmarks
- **Regression Tests**: Prevent breaking changes
- **User Acceptance**: Real-world usage scenarios`;
}

function getNodeTypeSpecificGuidance(nodeType: string): string {
  const guidance: Record<string, string> = {
    'image-processing': `
### Image Processing Specific Guidelines:
- **Tensor Formats**: Ensure BHWC format consistency
- **Color Spaces**: Handle RGB/BGR conversions properly
- **Memory Management**: Use torch.no_grad() for inference
- **Batch Processing**: Support variable batch sizes`,
    
    'model-inference': `
### Model Inference Specific Guidelines:
- **Model Loading**: Lazy loading for memory efficiency
- **Device Management**: Proper GPU/CPU handling
- **Preprocessing**: Consistent input normalization
- **Postprocessing**: Proper output formatting`,
  };
  
  return guidance[nodeType] || '';
}

// Export prompt definitions for MCP server
export const prompts: Prompt[] = [
  {
    name: 'generate-node-from-paper',
    description: 'Step-by-step guidance for generating ComfyUI nodes from research papers',
    arguments: [
      {
        name: 'paper_url',
        description: 'URL to the research paper',
        required: true,
      },
      {
        name: 'focus_area',
        description: 'Specific area to focus on (optional)',
        required: false,
      },
    ],
  },
  {
    name: 'debug-node-generation',
    description: 'Help debug failed node generation attempts',
    arguments: [
      {
        name: 'execution_id',
        description: 'ID of failed execution (optional)',
        required: false,
      },
      {
        name: 'error_description',
        description: 'Description of the error (optional)',
        required: false,
      },
    ],
  },
  {
    name: 'optimize-workflow',
    description: 'Optimize agent pipeline configuration for better performance',
    arguments: [
      {
        name: 'current_agents',
        description: 'Current agent configuration (optional)',
        required: false,
      },
      {
        name: 'performance_issues',
        description: 'Description of performance issues (optional)',
        required: false,
      },
    ],
  },
  {
    name: 'best-practices-guidance',
    description: 'Comprehensive guidance for professional ComfyUI node development',
    arguments: [
      {
        name: 'node_type',
        description: 'Type of node being developed (optional)',
        required: false,
      },
      {
        name: 'experience',
        description: 'Developer experience level (optional)',
        required: false,
      },
    ],
  },
];

// Prompt handler function
export function handlePromptRequest(name: string, args: Record<string, string>): PromptMessage[] {
  switch (name) {
    case 'generate-node-from-paper':
      return generateNodeFromPaper(args.paper_url, args.focus_area);
    case 'debug-node-generation':
      return debugNodeGeneration(args.execution_id, args.error_description);
    case 'optimize-workflow':
      return optimizeWorkflow(args.current_agents, args.performance_issues);
    case 'best-practices-guidance':
      return bestPracticesGuidance(args.node_type, args.experience);
    default:
      throw new Error(`Unknown prompt: ${name}`);
  }
}
