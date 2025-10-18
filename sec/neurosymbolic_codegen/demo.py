"""
Interactive Demo for Neuro-Symbolic Code Generation AI

Demonstrates core capabilities of the system.
"""

import sys
from pathlib import Path


def print_banner() -> None:
    """Print demo banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║        Neuro-Symbolic Code Generation AI - Interactive Demo      ║
║                                                                   ║
║  A production-grade system combining neural networks with         ║
║  symbolic reasoning for advanced code generation                  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def demo_ast_encoder() -> None:
    """Demonstrate AST Encoder functionality."""
    print("\n" + "="*70)
    print("1. AST Encoder Demo")
    print("="*70)
    
    print("\nAST Encoder Module:")
    print("  - Converts source code into Abstract Syntax Trees")
    print("  - Uses tree-sitter for language-agnostic parsing")
    print("  - Generates structural embeddings with PyTorch")
    
    sample_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
    
    print(f"\nSample Code:\n{sample_code}")
    print("\n✓ Would parse AST and extract:")
    print("  - Nodes: function definitions, conditionals, returns")
    print("  - Edges: control flow, data dependencies")
    print("  - Structural patterns: recursion detected")


def demo_smt_connector() -> None:
    """Demonstrate SMT Connector functionality."""
    print("\n" + "="*70)
    print("2. SMT Connector Demo")
    print("="*70)
    
    print("\nSMT Connector Module:")
    print("  - Bridges neural and symbolic reasoning")
    print("  - Uses Z3 and PySMT for constraint solving")
    print("  - Validates logical correctness")
    
    constraints = ["x > 0", "x < 10", "y == x * 2"]
    variables = {"x": "int", "y": "int"}
    
    print(f"\nExample Constraints: {constraints}")
    print(f"Variables: {variables}")
    print("\n✓ Would verify constraints and check:")
    print("  - Satisfiability: SATISFIABLE")
    print("  - Example solution: x=3, y=6")
    
    function_spec = {
        "name": "divide",
        "preconditions": ["divisor != 0"],
        "postconditions": ["result * divisor == dividend"],
    }
    
    print(f"\n✓ Would validate function specifications:")
    print(f"  - Preconditions: {function_spec['preconditions']}")
    print(f"  - Postconditions: {function_spec['postconditions']}")


def demo_reward_model() -> None:
    """Demonstrate Reward Model functionality."""
    print("\n" + "="*70)
    print("3. Multi-dimensional Reward Model Demo")
    print("="*70)
    
    print("\nReward Model Module:")
    print("  Formula: R = 0.6*TestPass + 0.15*(1-SecRisk) + 0.15*Quality + 0.1*License")
    
    test_results = {"total": 10, "passed": 8, "coverage": 0.75}
    security_analysis = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    quality_metrics = {"maintainability": 65, "complexity": 8, "duplication": 5}
    license_info = {"compliant": 18, "total": 20}
    
    print(f"\nExample Inputs:")
    print(f"  - Tests: {test_results['passed']}/{test_results['total']} passed, {test_results['coverage']*100}% coverage")
    print(f"  - Security: {security_analysis['high']} high, {security_analysis['medium']} medium vulnerabilities")
    print(f"  - Quality: Maintainability {quality_metrics['maintainability']}/100, Complexity {quality_metrics['complexity']}")
    print(f"  - License: {license_info['compliant']}/{license_info['total']} compliant")
    
    test_score = 0.7 * (test_results['passed']/test_results['total']) + 0.3 * test_results['coverage']
    total_reward = 0.6 * test_score + 0.15 * 0.7 + 0.15 * 0.65 + 0.1 * 0.9
    
    print(f"\n✓ Calculated Reward: {total_reward:.3f}")
    print(f"  - Test Score: {test_score:.3f}")
    print(f"  - Security Score: 0.700")
    print(f"  - Quality Score: 0.650")
    print(f"  - License Score: 0.900")


def demo_orchestrator() -> None:
    """Demonstrate Neuro-Symbolic Orchestrator."""
    print("\n" + "="*70)
    print("4. Neuro-Symbolic Orchestrator Demo")
    print("="*70)
    
    print("\nNeuro-Symbolic Orchestrator:")
    print("  - Combines neural creativity with symbolic reasoning")
    print("  - Adaptive pathway selection based on task")
    print("  - Meta-learning for strategy optimization")
    
    prompt = "Write a function to calculate prime numbers"
    print(f"\nPrompt: {prompt}")
    
    modes = ["NEURAL", "SYMBOLIC", "HYBRID"]
    confidences = [0.85, 0.95, 0.90]
    
    for mode, conf in zip(modes, confidences):
        print(f"\n✓ {mode} Mode:")
        print(f"  - Confidence: {conf:.2f}")
        print(f"  - Strategy: {'Creative generation' if mode == 'NEURAL' else 'Formal verification' if mode == 'SYMBOLIC' else 'Combined approach'}")


def demo_self_repair() -> None:
    """Demonstrate Self-Repair Mechanism."""
    print("\n" + "="*70)
    print("5. Self-Repair Mechanism Demo")
    print("="*70)
    
    print("\nSelf-Repair Mechanism:")
    print("  - Detects compile/runtime errors")
    print("  - Iterative correction loops")
    print("  - Learning from repair history")
    
    buggy_code = '''
def divide(a, b):
    return a / b
'''
    
    print(f"\nAnalyzing buggy code:\n{buggy_code}")
    print("\n✓ Would detect defects:")
    print("  - Missing zero-division check")
    print("  - No type validation")
    print("  - No error handling")
    
    print(f"\n✓ Would apply repairs:")
    print("  - Add: if b == 0: raise ValueError")
    print("  - Add: type hints")
    print("  - Add: try/except block")


def demo_security_analyzer() -> None:
    """Demonstrate Security Analyzer."""
    print("\n" + "="*70)
    print("6. Security Analysis Engine Demo")
    print("="*70)
    
    print("\nSecurity Analysis Engine:")
    print("  - Static analysis with Bandit")
    print("  - Semantic analysis with Semgrep")
    print("  - License compliance with SPDX")
    
    code = '''
import pickle
import os

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def execute_command(cmd):
    os.system(cmd)
'''
    
    print(f"\nAnalyzing potentially unsafe code:\n{code}")
    
    print(f"\n✓ Would detect vulnerabilities:")
    print(f"  - HIGH: Unsafe deserialization (pickle)")
    print(f"  - CRITICAL: Command injection (os.system)")
    print(f"  - MEDIUM: Path traversal risk")
    
    print(f"\n✓ Recommendations:")
    print(f"  - Use json instead of pickle")
    print(f"  - Use subprocess with shell=False")
    print(f"  - Validate and sanitize file paths")


def run_full_demo() -> None:
    """Run full demonstration of all components."""
    print_banner()
    
    print("\nInitializing Neuro-Symbolic Code Generation AI System...")
    print("This demo showcases core capabilities of the system.\n")
    
    try:
        demo_ast_encoder()
        demo_smt_connector()
        demo_reward_model()
        demo_orchestrator()
        demo_self_repair()
        demo_security_analyzer()
        
        print("\n" + "="*70)
        print("Demo Complete!")
        print("="*70)
        print("\nAll core components demonstrated successfully.")
        print("\nNext steps:")
        print("  - Install Taskfile: brew install go-task (macOS) or see taskfile.dev")
        print("  - Run 'task setup' to install all dependencies")
        print("  - Run 'task download-models' to download pre-trained models")
        print("  - Run 'task prepare-data' to prepare training data")
        print("  - Run 'task train' to start training pipeline")
        print("  - Run 'task evaluate' to evaluate on benchmarks")
        print("\nFor more information, see README.md and docs/")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point for demo."""
    run_full_demo()


if __name__ == "__main__":
    main()
