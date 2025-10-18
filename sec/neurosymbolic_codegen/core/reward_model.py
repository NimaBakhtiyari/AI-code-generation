"""
Reward Model Module

Multi-dimensional reward calculation for RLHF training.
Formula: R = 0.6 * TestPassRate + 0.15 * (1 - SecurityRisk) + 0.15 * StaticQuality + 0.1 * LicenseCompliance
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class RewardComponents:
    """Individual components of the reward signal."""
    test_pass_rate: float
    security_risk: float
    static_quality: float
    license_compliance: float
    total_reward: float


class RewardModel:
    """
    Multi-dimensional Reward Model
    
    Calculates reward signals for reinforcement learning from multiple dimensions:
    - Test Pass Rate (60%): Unit and integration test success
    - Security Risk (15%): Vulnerability and security analysis
    - Static Quality (15%): Code maintainability and complexity
    - License Compliance (10%): Dependency license validation
    
    Features:
    - Continuous feedback integration
    - Multi-objective optimization
    - Adaptive weight adjustment
    - Explainable reward breakdown
    """
    
    def __init__(
        self,
        test_weight: float = 0.6,
        security_weight: float = 0.15,
        quality_weight: float = 0.15,
        license_weight: float = 0.1,
        normalize: bool = True,
    ) -> None:
        """
        Initialize Reward Model.
        
        Args:
            test_weight: Weight for test pass rate (default 0.6)
            security_weight: Weight for security risk (default 0.15)
            quality_weight: Weight for static quality (default 0.15)
            license_weight: Weight for license compliance (default 0.1)
            normalize: Whether to normalize rewards to [0, 1]
        """
        self.test_weight = test_weight
        self.security_weight = security_weight
        self.quality_weight = quality_weight
        self.license_weight = license_weight
        self.normalize = normalize
        
        total_weight = test_weight + security_weight + quality_weight + license_weight
        assert abs(total_weight - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total_weight}"
        
        logger.info(
            "reward_model_initialized",
            test_weight=test_weight,
            security_weight=security_weight,
            quality_weight=quality_weight,
            license_weight=license_weight,
        )
    
    def calculate_reward(
        self,
        test_results: Dict[str, Any],
        security_analysis: Dict[str, Any],
        quality_metrics: Dict[str, Any],
        license_info: Dict[str, Any],
    ) -> RewardComponents:
        """
        Calculate total reward from all components.
        
        Args:
            test_results: Test execution results
            security_analysis: Security scan results
            quality_metrics: Static code quality metrics
            license_info: License compliance information
            
        Returns:
            RewardComponents with individual and total rewards
        """
        test_score = self._calculate_test_score(test_results)
        security_score = self._calculate_security_score(security_analysis)
        quality_score = self._calculate_quality_score(quality_metrics)
        license_score = self._calculate_license_score(license_info)
        
        total_reward = (
            self.test_weight * test_score +
            self.security_weight * (1.0 - security_score) +
            self.quality_weight * quality_score +
            self.license_weight * license_score
        )
        
        if self.normalize:
            total_reward = max(0.0, min(1.0, total_reward))
        
        components = RewardComponents(
            test_pass_rate=test_score,
            security_risk=security_score,
            static_quality=quality_score,
            license_compliance=license_score,
            total_reward=total_reward,
        )
        
        logger.info(
            "reward_calculated",
            total_reward=total_reward,
            test_score=test_score,
            security_score=security_score,
            quality_score=quality_score,
            license_score=license_score,
        )
        
        return components
    
    def _calculate_test_score(self, test_results: Dict[str, Any]) -> float:
        """
        Calculate test pass rate score.
        
        Args:
            test_results: Test execution results
            
        Returns:
            Test pass rate [0, 1]
        """
        total_tests = test_results.get("total", 0)
        passed_tests = test_results.get("passed", 0)
        
        if total_tests == 0:
            return 0.0
        
        pass_rate = passed_tests / total_tests
        
        coverage = test_results.get("coverage", 0.0)
        
        score = 0.7 * pass_rate + 0.3 * coverage
        
        return score
    
    def _calculate_security_score(self, security_analysis: Dict[str, Any]) -> float:
        """
        Calculate security risk score (higher = more risk).
        
        Args:
            security_analysis: Security scan results
            
        Returns:
            Security risk score [0, 1]
        """
        vulnerabilities = security_analysis.get("vulnerabilities", {})
        
        critical = vulnerabilities.get("critical", 0)
        high = vulnerabilities.get("high", 0)
        medium = vulnerabilities.get("medium", 0)
        low = vulnerabilities.get("low", 0)
        
        weighted_risk = (
            1.0 * critical +
            0.7 * high +
            0.4 * medium +
            0.1 * low
        )
        
        risk_score = min(1.0, weighted_risk / 10.0)
        
        return risk_score
    
    def _calculate_quality_score(self, quality_metrics: Dict[str, Any]) -> float:
        """
        Calculate static code quality score.
        
        Args:
            quality_metrics: Code quality metrics
            
        Returns:
            Quality score [0, 1]
        """
        maintainability = quality_metrics.get("maintainability_index", 0.0) / 100.0
        
        complexity = quality_metrics.get("cyclomatic_complexity", 10)
        complexity_score = max(0.0, 1.0 - (complexity - 1) / 20.0)
        
        duplication = quality_metrics.get("duplication_percentage", 0.0) / 100.0
        duplication_score = 1.0 - duplication
        
        quality_score = (
            0.4 * maintainability +
            0.4 * complexity_score +
            0.2 * duplication_score
        )
        
        return quality_score
    
    def _calculate_license_score(self, license_info: Dict[str, Any]) -> float:
        """
        Calculate license compliance score.
        
        Args:
            license_info: License compliance information
            
        Returns:
            Compliance score [0, 1]
        """
        compliant_deps = license_info.get("compliant_dependencies", 0)
        total_deps = license_info.get("total_dependencies", 0)
        
        if total_deps == 0:
            return 1.0
        
        compliance_rate = compliant_deps / total_deps
        
        has_license = license_info.get("has_license_file", False)
        license_bonus = 0.2 if has_license else 0.0
        
        score = min(1.0, 0.8 * compliance_rate + license_bonus)
        
        return score
    
    def get_reward_breakdown(self, components: RewardComponents) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components.
        
        Args:
            components: Reward components
            
        Returns:
            Dictionary with weighted contributions
        """
        return {
            "test_contribution": self.test_weight * components.test_pass_rate,
            "security_contribution": self.security_weight * (1.0 - components.security_risk),
            "quality_contribution": self.quality_weight * components.static_quality,
            "license_contribution": self.license_weight * components.license_compliance,
            "total_reward": components.total_reward,
        }
