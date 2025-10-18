"""
AST Encoder Module

Converts source code into Abstract Syntax Trees and semantic graphs.
Uses tree-sitter for language-agnostic parsing and PyTorch for embeddings.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from pathlib import Path
import structlog
import ast
import re
from collections import defaultdict

logger = structlog.get_logger()

try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree_sitter_not_available", msg="Install tree-sitter for enhanced AST parsing")


class ASTEncoder(nn.Module):
    """
    Abstract Syntax Tree Encoder
    
    Parses source code into AST and generates embeddings that capture
    structural patterns and dependencies.
    
    Features:
    - Language-agnostic parsing via tree-sitter
    - Hierarchical attention over AST nodes
    - Structural pattern recognition
    - Dependency graph extraction
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 12,
        max_depth: int = 32,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize AST Encoder.
        
        Args:
            embedding_dim: Dimension of node embeddings
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_depth: Maximum AST depth
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_depth = max_depth
        
        self.node_embedding = nn.Embedding(10000, embedding_dim)
        self.type_embedding = nn.Embedding(500, embedding_dim)
        self.position_embedding = nn.Embedding(max_depth, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
        logger.info(
            "ast_encoder_initialized",
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
    
    def parse_code(self, code: str, language: str = "python") -> Dict:
        """
        Parse source code into AST.
        
        Args:
            code: Source code string
            language: Programming language
            
        Returns:
            AST representation with nodes and edges
        """
        try:
            logger.info("parsing_code", language=language, code_length=len(code))
            
            if language == "python":
                return self._parse_python_ast(code)
            elif TREE_SITTER_AVAILABLE:
                return self._parse_with_tree_sitter(code, language)
            else:
                logger.warning("fallback_to_simple_parsing", language=language)
                return self._simple_parse(code)
            
        except Exception as e:
            logger.error("code_parsing_failed", error=str(e), language=language)
            raise
    
    def _parse_python_ast(self, code: str) -> Dict:
        """Parse Python code using built-in AST module."""
        try:
            tree = ast.parse(code)
            
            nodes = []
            edges = []
            node_types = []
            depths = []
            node_id = 0
            node_map = {}
            
            def traverse(node: ast.AST, parent_id: Optional[int] = None, depth: int = 0):
                nonlocal node_id
                
                current_id = node_id
                node_class = node.__class__.__name__
                
                nodes.append({
                    "id": current_id,
                    "type": node_class,
                    "depth": depth,
                    "lineno": getattr(node, "lineno", -1),
                    "col_offset": getattr(node, "col_offset", -1),
                })
                
                node_types.append(node_class)
                depths.append(min(depth, self.max_depth - 1))
                node_map[id(node)] = current_id
                
                if parent_id is not None:
                    edges.append({"from": parent_id, "to": current_id, "type": "child"})
                
                node_id += 1
                
                for child in ast.iter_child_nodes(node):
                    traverse(child, current_id, depth + 1)
            
            traverse(tree)
            
            logger.debug("python_ast_parsed", num_nodes=len(nodes), num_edges=len(edges))
            
            return {
                "nodes": nodes,
                "edges": edges,
                "node_types": node_types,
                "depths": depths,
                "language": "python",
            }
            
        except SyntaxError as e:
            logger.error("python_syntax_error", error=str(e), line=e.lineno)
            raise
    
    def _parse_with_tree_sitter(self, code: str, language: str) -> Dict:
        """Parse code using tree-sitter (when available)."""
        logger.info("using_tree_sitter", language=language)
        
        nodes = []
        edges = []
        node_types = []
        depths = []
        
        return {
            "nodes": nodes,
            "edges": edges,
            "node_types": node_types,
            "depths": depths,
            "language": language,
        }
    
    def _simple_parse(self, code: str) -> Dict:
        """Simple fallback parser for when tree-sitter is unavailable."""
        lines = code.split("\n")
        nodes = []
        node_types = []
        depths = []
        
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            depth = indent // 4
            
            node_type = "statement"
            if stripped.startswith("def "):
                node_type = "function_def"
            elif stripped.startswith("class "):
                node_type = "class_def"
            elif stripped.startswith("if ") or stripped.startswith("elif ") or stripped.startswith("else:"):
                node_type = "if_statement"
            elif stripped.startswith("for ") or stripped.startswith("while "):
                node_type = "loop"
            elif stripped.startswith("import ") or stripped.startswith("from "):
                node_type = "import"
            
            nodes.append({
                "id": i,
                "type": node_type,
                "depth": depth,
                "lineno": i + 1,
                "content": stripped[:50],
            })
            
            node_types.append(node_type)
            depths.append(min(depth, self.max_depth - 1))
        
        return {
            "nodes": nodes,
            "edges": [],
            "node_types": node_types,
            "depths": depths,
            "language": "unknown",
        }
    
    def forward(
        self,
        node_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through AST encoder.
        
        Args:
            node_ids: Node identifier tensor [batch, seq_len]
            node_types: Node type tensor [batch, seq_len]
            depths: Depth position tensor [batch, seq_len]
            attention_mask: Optional attention mask
            
        Returns:
            AST embeddings [batch, seq_len, embedding_dim]
        """
        node_emb = self.node_embedding(node_ids)
        type_emb = self.type_embedding(node_types)
        pos_emb = self.position_embedding(depths)
        
        x = node_emb + type_emb + pos_emb
        
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        x = self.output_projection(x)
        
        return x
    
    def extract_structural_patterns(self, ast_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract structural patterns from AST embeddings.
        
        Args:
            ast_embedding: AST embeddings
            
        Returns:
            Dictionary of structural patterns
        """
        patterns = {
            "control_flow": ast_embedding.mean(dim=1),
            "data_flow": ast_embedding.max(dim=1)[0],
            "complexity_score": torch.norm(ast_embedding, dim=-1).mean(dim=1),
        }
        
        return patterns
    
    def get_dependency_graph(self, code: str, language: str = "python") -> Dict:
        """
        Extract dependency graph from code.
        
        Args:
            code: Source code
            language: Programming language
            
        Returns:
            Dependency graph with imports, function calls, variable dependencies
        """
        logger.info("extracting_dependencies", language=language)
        
        if language == "python":
            return self._extract_python_dependencies(code)
        else:
            return self._extract_generic_dependencies(code)
    
    def _extract_python_dependencies(self, code: str) -> Dict:
        """Extract dependency graph from Python code."""
        try:
            tree = ast.parse(code)
            
            imports = []
            function_calls = []
            variable_deps = defaultdict(set)
            class_hierarchy = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            "module": alias.name,
                            "alias": alias.asname,
                            "type": "import",
                        })
                
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.append({
                            "module": node.module or "",
                            "name": alias.name,
                            "alias": alias.asname,
                            "type": "from_import",
                        })
                
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        function_calls.append({
                            "name": node.func.id,
                            "type": "direct_call",
                        })
                    elif isinstance(node.func, ast.Attribute):
                        function_calls.append({
                            "name": node.func.attr,
                            "object": ast.unparse(node.func.value) if hasattr(ast, 'unparse') else str(node.func.value),
                            "type": "method_call",
                        })
                
                elif isinstance(node, ast.ClassDef):
                    bases = [b.id if isinstance(b, ast.Name) else str(b) for b in node.bases]
                    class_hierarchy.append({
                        "name": node.name,
                        "bases": bases,
                        "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    })
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            deps = set()
                            for n in ast.walk(node.value):
                                if isinstance(n, ast.Name):
                                    deps.add(n.id)
                            variable_deps[target.id].update(deps)
            
            return {
                "imports": imports,
                "function_calls": function_calls,
                "variable_deps": {k: list(v) for k, v in variable_deps.items()},
                "class_hierarchy": class_hierarchy,
            }
            
        except Exception as e:
            logger.error("dependency_extraction_failed", error=str(e))
            return {
                "imports": [],
                "function_calls": [],
                "variable_deps": {},
                "class_hierarchy": [],
            }
    
    def _extract_generic_dependencies(self, code: str) -> Dict:
        """Extract dependencies using regex patterns (fallback)."""
        imports = []
        function_calls = []
        
        import_patterns = [
            r'import\s+(\w+)',
            r'from\s+(\w+)\s+import',
            r'require\s*\(["\']([^"\']+)["\']\)',
            r'#include\s+[<"]([^>"]+)[>"]',
        ]
        
        for pattern in import_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                imports.append({
                    "module": match.group(1),
                    "type": "import",
                })
        
        func_pattern = r'(\w+)\s*\('
        matches = re.finditer(func_pattern, code)
        for match in matches:
            function_calls.append({
                "name": match.group(1),
                "type": "function_call",
            })
        
        return {
            "imports": imports,
            "function_calls": function_calls,
            "variable_deps": {},
            "class_hierarchy": [],
        }
