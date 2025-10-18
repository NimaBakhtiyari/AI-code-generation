"""Unit tests for AST Encoder module."""

import pytest
import torch
from neurosymbolic_codegen.core.ast_encoder import ASTEncoder


class TestASTEncoder:
    """Test suite for AST Encoder."""
    
    @pytest.fixture
    def encoder(self):
        """Create AST encoder instance."""
        return ASTEncoder(embedding_dim=256, num_layers=2, num_heads=4)
    
    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.embedding_dim == 256
        assert encoder.num_layers == 2
        assert encoder.num_heads == 4
    
    def test_parse_code(self, encoder):
        """Test code parsing."""
        code = "def test(): return 42"
        ast_data = encoder.parse_code(code, language="python")
        
        assert "nodes" in ast_data
        assert "edges" in ast_data
        assert "node_types" in ast_data
        assert "depths" in ast_data
    
    def test_forward_pass(self, encoder):
        """Test forward pass through encoder."""
        batch_size = 2
        seq_len = 10
        
        node_ids = torch.randint(0, 1000, (batch_size, seq_len))
        node_types = torch.randint(0, 50, (batch_size, seq_len))
        depths = torch.randint(0, 10, (batch_size, seq_len))
        
        output = encoder(node_ids, node_types, depths)
        
        assert output.shape == (batch_size, seq_len, 256)
    
    def test_extract_structural_patterns(self, encoder):
        """Test structural pattern extraction."""
        batch_size = 2
        seq_len = 10
        
        ast_embedding = torch.randn(batch_size, seq_len, 256)
        patterns = encoder.extract_structural_patterns(ast_embedding)
        
        assert "control_flow" in patterns
        assert "data_flow" in patterns
        assert "complexity_score" in patterns
    
    def test_get_dependency_graph(self, encoder):
        """Test dependency graph extraction."""
        code = "import os\ndef test(): os.path.join('a', 'b')"
        deps = encoder.get_dependency_graph(code)
        
        assert "imports" in deps
        assert "function_calls" in deps
        assert "variable_deps" in deps
        assert "class_hierarchy" in deps
