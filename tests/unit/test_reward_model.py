"""Unit tests for Reward Model."""

import pytest
from neurosymbolic_codegen.core.reward_model import RewardModel, RewardComponents


class TestRewardModel:
    """Test suite for Reward Model."""
    
    @pytest.fixture
    def reward_model(self):
        """Create reward model instance."""
        return RewardModel()
    
    def test_initialization(self, reward_model):
        """Test reward model initialization."""
        assert reward_model.test_weight == 0.6
        assert reward_model.security_weight == 0.15
        assert reward_model.quality_weight == 0.15
        assert reward_model.license_weight == 0.1
    
    def test_calculate_reward(self, reward_model):
        """Test reward calculation."""
        test_results = {"total": 10, "passed": 8, "coverage": 0.75}
        security_analysis = {"vulnerabilities": {"critical": 0, "high": 1, "medium": 2, "low": 3}}
        quality_metrics = {"maintainability_index": 65, "cyclomatic_complexity": 8, "duplication_percentage": 5}
        license_info = {"compliant_dependencies": 18, "total_dependencies": 20, "has_license_file": True}
        
        components = reward_model.calculate_reward(
            test_results, security_analysis, quality_metrics, license_info
        )
        
        assert isinstance(components, RewardComponents)
        assert 0 <= components.total_reward <= 1
        assert 0 <= components.test_pass_rate <= 1
        assert 0 <= components.security_risk <= 1
        assert 0 <= components.static_quality <= 1
        assert 0 <= components.license_compliance <= 1
    
    def test_perfect_score(self, reward_model):
        """Test perfect reward score."""
        test_results = {"total": 10, "passed": 10, "coverage": 1.0}
        security_analysis = {"vulnerabilities": {"critical": 0, "high": 0, "medium": 0, "low": 0}}
        quality_metrics = {"maintainability_index": 100, "cyclomatic_complexity": 1, "duplication_percentage": 0}
        license_info = {"compliant_dependencies": 20, "total_dependencies": 20, "has_license_file": True}
        
        components = reward_model.calculate_reward(
            test_results, security_analysis, quality_metrics, license_info
        )
        
        assert components.total_reward > 0.9
    
    def test_reward_breakdown(self, reward_model):
        """Test reward breakdown."""
        test_results = {"total": 10, "passed": 8, "coverage": 0.75}
        security_analysis = {"vulnerabilities": {"critical": 0, "high": 0, "medium": 0, "low": 0}}
        quality_metrics = {"maintainability_index": 70, "cyclomatic_complexity": 5, "duplication_percentage": 3}
        license_info = {"compliant_dependencies": 20, "total_dependencies": 20, "has_license_file": True}
        
        components = reward_model.calculate_reward(
            test_results, security_analysis, quality_metrics, license_info
        )
        
        breakdown = reward_model.get_reward_breakdown(components)
        
        assert "test_contribution" in breakdown
        assert "security_contribution" in breakdown
        assert "quality_contribution" in breakdown
        assert "license_contribution" in breakdown
        assert "total_reward" in breakdown
