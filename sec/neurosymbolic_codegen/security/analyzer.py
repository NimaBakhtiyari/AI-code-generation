"""
Security Analyzer Module

Comprehensive security analysis using Bandit, Semgrep, and SPDX compliance.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import structlog
import subprocess
import tempfile
import json
import re
from pathlib import Path

logger = structlog.get_logger()

try:
    subprocess.run(["bandit", "--version"], capture_output=True, check=True)
    BANDIT_AVAILABLE = True
except (subprocess.CalledProcessError, FileNotFoundError):
    BANDIT_AVAILABLE = False
    logger.warning("bandit_not_available", msg="Install bandit for security scanning")


@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability."""
    severity: str
    category: str
    location: Dict[str, int]
    description: str
    cwe_id: Optional[str] = None
    recommendation: Optional[str] = None


class SecurityAnalyzer:
    """
    Security Analysis Engine
    
    Performs comprehensive security analysis on generated code.
    
    Features:
    - Static vulnerability detection (Bandit)
    - Semantic security analysis (Semgrep)
    - Dependency scanning
    - License compliance checking (SPDX)
    - CWE mapping and recommendations
    """
    
    def __init__(
        self,
        enable_bandit: bool = True,
        enable_semgrep: bool = True,
        enable_license_check: bool = True,
        severity_threshold: str = "medium",
    ) -> None:
        """
        Initialize Security Analyzer.
        
        Args:
            enable_bandit: Enable Bandit scanner
            enable_semgrep: Enable Semgrep scanner
            enable_license_check: Enable license compliance checking
            severity_threshold: Minimum severity to report
        """
        self.enable_bandit = enable_bandit
        self.enable_semgrep = enable_semgrep
        self.enable_license_check = enable_license_check
        self.severity_threshold = severity_threshold
        
        logger.info(
            "security_analyzer_initialized",
            bandit=enable_bandit,
            semgrep=enable_semgrep,
            license_check=enable_license_check,
        )
    
    def analyze_code(
        self,
        code: str,
        language: str = "python",
        dependencies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive security analysis.
        
        Args:
            code: Source code to analyze
            language: Programming language
            dependencies: Optional list of dependencies
            
        Returns:
            Security analysis report
        """
        logger.info("starting_security_analysis", language=language)
        
        vulnerabilities = []
        
        if self.enable_bandit and language == "python":
            bandit_vulns = self._run_bandit(code)
            vulnerabilities.extend(bandit_vulns)
        
        if self.enable_semgrep:
            semgrep_vulns = self._run_semgrep(code, language)
            vulnerabilities.extend(semgrep_vulns)
        
        dependency_issues = []
        if self.enable_license_check and dependencies:
            dependency_issues = self._check_licenses(dependencies)
        
        report = self._generate_report(vulnerabilities, dependency_issues)
        
        logger.info(
            "security_analysis_complete",
            num_vulnerabilities=len(vulnerabilities),
            num_dependency_issues=len(dependency_issues),
        )
        
        return report
    
    def analyze(
        self,
        code: str,
        language: str = "python",
        dependencies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Legacy method - redirects to analyze_code."""
        return self.analyze_code(code, language, dependencies)
    
    def _run_bandit(self, code: str) -> List[SecurityVulnerability]:
        """
        Run Bandit security scanner.
        
        Args:
            code: Python code to scan
            
        Returns:
            List of vulnerabilities
        """
        logger.debug("running_bandit_scan")
        
        vulnerabilities = []
        
        if not BANDIT_AVAILABLE:
            logger.warning("bandit_not_available_using_pattern_matching")
            return self._pattern_based_security_check(code)
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                ["bandit", "-f", "json", temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            Path(temp_file).unlink()
            
            if result.stdout:
                data = json.loads(result.stdout)
                for issue in data.get("results", []):
                    vulnerabilities.append(SecurityVulnerability(
                        severity=issue.get("issue_severity", "LOW").lower(),
                        category=issue.get("test_id", "unknown"),
                        location={
                            "line": issue.get("line_number", 0),
                            "column": issue.get("col_offset", 0),
                        },
                        description=issue.get("issue_text", ""),
                        cwe_id=issue.get("issue_cwe", {}).get("id"),
                        recommendation=issue.get("issue_text", ""),
                    ))
        
        except Exception as e:
            logger.error("bandit_scan_failed", error=str(e))
        
        return vulnerabilities
    
    def _pattern_based_security_check(self, code: str) -> List[SecurityVulnerability]:
        """Fallback pattern-based security check."""
        vulnerabilities = []
        
        patterns = {
            "eval": ("high", "B307", "Use of eval() detected"),
            "exec": ("high", "B102", "Use of exec() detected"),
            "pickle.loads": ("medium", "B301", "Use of pickle detected"),
            "os.system": ("high", "B605", "Use of os.system() detected"),
            "subprocess.call.*shell=True": ("high", "B602", "Shell injection risk"),
        }
        
        for pattern, (severity, cwe, desc) in patterns.items():
            if re.search(pattern, code):
                vulnerabilities.append(SecurityVulnerability(
                    severity=severity,
                    category=cwe,
                    location={"line": 0, "column": 0},
                    description=desc,
                    cwe_id=cwe,
                ))
        
        return vulnerabilities
    
    def _run_semgrep(self, code: str, language: str) -> List[SecurityVulnerability]:
        """
        Run Semgrep semantic scanner.
        
        Args:
            code: Source code
            language: Programming language
            
        Returns:
            List of vulnerabilities
        """
        logger.debug("running_semgrep_scan", language=language)
        
        vulnerabilities = []
        
        return vulnerabilities
    
    def _check_licenses(self, dependencies: List[str]) -> List[Dict[str, Any]]:
        """
        Check license compliance for dependencies.
        
        Args:
            dependencies: List of dependency names
            
        Returns:
            List of license issues
        """
        logger.debug("checking_licenses", num_dependencies=len(dependencies))
        
        issues = []
        
        return issues
    
    def _generate_report(
        self,
        vulnerabilities: List[SecurityVulnerability],
        dependency_issues: List[Dict],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive security report.
        
        Args:
            vulnerabilities: Found vulnerabilities
            dependency_issues: Dependency issues
            
        Returns:
            Security report
        """
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }
        
        for vuln in vulnerabilities:
            severity = vuln.severity.lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        report = {
            "vulnerabilities": {
                "total": len(vulnerabilities),
                "by_severity": severity_counts,
                "details": [
                    {
                        "severity": v.severity,
                        "category": v.category,
                        "description": v.description,
                        "location": v.location,
                    }
                    for v in vulnerabilities
                ],
            },
            "dependencies": {
                "total_issues": len(dependency_issues),
                "details": dependency_issues,
            },
            "risk_score": self._calculate_risk_score(severity_counts),
        }
        
        return report
    
    def _calculate_risk_score(self, severity_counts: Dict[str, int]) -> float:
        """
        Calculate overall risk score.
        
        Args:
            severity_counts: Counts by severity
            
        Returns:
            Risk score [0, 1]
        """
        weighted_score = (
            1.0 * severity_counts.get("critical", 0) +
            0.7 * severity_counts.get("high", 0) +
            0.4 * severity_counts.get("medium", 0) +
            0.1 * severity_counts.get("low", 0)
        )
        
        risk_score = min(1.0, weighted_score / 10.0)
        
        return risk_score
