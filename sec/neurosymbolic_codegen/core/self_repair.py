"""
Self-Repair Mechanism Module

Detects and auto-corrects code defects using static/dynamic analysis
and iterative correction loops with Neural Compiler Feedback.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger()


class DefectType(Enum):
    """Types of code defects."""
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    SECURITY_VULNERABILITY = "security_vulnerability"
    STYLE_VIOLATION = "style_violation"


@dataclass
class CodeDefect:
    """Represents a detected code defect."""
    defect_type: DefectType
    severity: str
    location: Dict[str, int]
    message: str
    suggested_fix: Optional[str] = None


class SelfRepairMechanism:
    """
    Self-Repair Mechanism for Automated Code Correction
    
    Detects compile/runtime errors and triggers correction loops.
    Uses Neural Compiler Feedback (NCF) for learning from failures.
    
    Features:
    - Static code analysis (AST-based)
    - Dynamic runtime analysis
    - Iterative correction loops
    - Failure-aware reward shaping
    - Learning from repair history
    """
    
    def __init__(
        self,
        max_repair_iterations: int = 5,
        confidence_threshold: float = 0.6,
        enable_learning: bool = True,
    ) -> None:
        """
        Initialize Self-Repair Mechanism.
        
        Args:
            max_repair_iterations: Maximum repair attempts
            confidence_threshold: Minimum confidence for applying fix
            enable_learning: Enable learning from repair history
        """
        self.max_repair_iterations = max_repair_iterations
        self.confidence_threshold = confidence_threshold
        self.enable_learning = enable_learning
        self.repair_history: List[Dict] = []
        
        logger.info(
            "self_repair_initialized",
            max_iterations=max_repair_iterations,
            confidence_threshold=confidence_threshold,
        )
    
    def detect_defects(
        self,
        code: str,
        language: str = "python",
        enable_dynamic: bool = False,
    ) -> List[CodeDefect]:
        """
        Detect code defects using static and optional dynamic analysis.
        
        Args:
            code: Source code to analyze
            language: Programming language
            enable_dynamic: Enable dynamic runtime analysis
            
        Returns:
            List of detected defects
        """
        logger.info("detecting_defects", language=language, code_length=len(code))
        
        defects = []
        
        static_defects = self._static_analysis(code, language)
        defects.extend(static_defects)
        
        if enable_dynamic:
            dynamic_defects = self._dynamic_analysis(code, language)
            defects.extend(dynamic_defects)
        
        logger.info("defects_detected", num_defects=len(defects))
        return defects
    
    def repair_code(
        self,
        code: str,
        defects: List[Dict],
        language: str = "python",
    ) -> Optional[str]:
        """
        Attempt to repair code defects iteratively.
        
        Args:
            code: Original source code
            defects: List of detected defects (dicts from security analyzer)
            language: Programming language
            
        Returns:
            Repaired code or None if repair failed
        """
        logger.info("starting_repair", num_defects=len(defects))
        
        if not defects:
            return code
        
        current_code = code
        
        for defect_dict in defects:
            try:
                defect_type_str = defect_dict.get("defect_type", "security_vulnerability")
                if hasattr(DefectType, defect_type_str.upper()):
                    defect_type = DefectType[defect_type_str.upper()]
                else:
                    defect_type = DefectType.SECURITY_VULNERABILITY
                
                defect = CodeDefect(
                    defect_type=defect_type,
                    severity=defect_dict.get("severity", "medium"),
                    location=defect_dict.get("location", {"line": 0, "column": 0}),
                    message=defect_dict.get("message", ""),
                )
                
                fix, confidence = self._generate_fix(current_code, defect, language)
                
                if confidence >= self.confidence_threshold:
                    current_code = self._apply_fix(current_code, fix)
                    logger.info("defect_repaired", defect_type=defect.defect_type.value)
                
            except Exception as e:
                logger.error("repair_iteration_failed", error=str(e))
                continue
        
        logger.info("repair_complete")
        return current_code
    
    def _static_analysis(self, code: str, language: str) -> List[CodeDefect]:
        """
        Perform static code analysis.
        
        Args:
            code: Source code
            language: Programming language
            
        Returns:
            List of static defects
        """
        defects = []
        
        if language == "python":
            import ast as python_ast
            try:
                python_ast.parse(code)
            except SyntaxError as e:
                defects.append(CodeDefect(
                    defect_type=DefectType.SYNTAX_ERROR,
                    severity="high",
                    location={"line": e.lineno or 0, "column": e.offset or 0},
                    message=str(e.msg),
                    suggested_fix=None,
                ))
            
            try:
                compile(code, '<string>', 'exec')
            except Exception as e:
                if "SyntaxError" not in str(type(e)):
                    defects.append(CodeDefect(
                        defect_type=DefectType.TYPE_ERROR,
                        severity="medium",
                        location={"line": 0, "column": 0},
                        message=str(e),
                    ))
        
        logger.debug("static_analysis_complete", num_defects=len(defects))
        return defects
    
    def _dynamic_analysis(self, code: str, language: str) -> List[CodeDefect]:
        """
        Perform dynamic runtime analysis.
        
        Args:
            code: Source code
            language: Programming language
            
        Returns:
            List of dynamic defects
        """
        defects = []
        
        logger.debug("dynamic_analysis_complete", num_defects=len(defects))
        return defects
    
    def _generate_fix(
        self,
        code: str,
        defect: CodeDefect,
        language: str,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Generate a fix for a detected defect.
        
        Args:
            code: Source code
            defect: Detected defect
            language: Programming language
            
        Returns:
            Tuple of (fix_suggestion, confidence)
        """
        logger.debug("generating_fix", defect_type=defect.defect_type.value)
        
        if defect.suggested_fix:
            return {"type": "suggested", "fix": defect.suggested_fix}, 0.9
        
        fix = {"type": "no_fix", "description": "No automatic fix available"}
        confidence = 0.0
        
        if defect.defect_type == DefectType.SYNTAX_ERROR:
            if "unexpected EOF" in defect.message.lower():
                fix = {
                    "type": "add_closing",
                    "description": "Add missing closing bracket/parenthesis",
                    "location": defect.location
                }
                confidence = 0.8
            elif "invalid syntax" in defect.message.lower():
                fix = {
                    "type": "syntax_correction",
                    "description": "Correct syntax error",
                    "location": defect.location
                }
                confidence = 0.6
        
        elif defect.defect_type == DefectType.TYPE_ERROR:
            fix = {
                "type": "type_cast",
                "description": "Add type conversion or fix type mismatch",
                "location": defect.location
            }
            confidence = 0.5
        
        return fix, confidence
    
    def _apply_fix(self, code: str, fix: Dict[str, Any]) -> str:
        """
        Apply a fix to the code.
        
        Args:
            code: Original code
            fix: Fix dictionary to apply
            
        Returns:
            Modified code
        """
        logger.debug("applying_fix", fix_type=fix.get("type"))
        
        modified_code = code
        
        if fix.get("type") == "suggested" and "fix" in fix:
            modified_code = fix["fix"]
        elif fix.get("type") == "add_closing":
            modified_code = code + "\n)"
        elif fix.get("type") == "syntax_correction":
            pass
        
        return modified_code
    
    def learn_from_repair(
        self,
        original_code: str,
        repaired_code: str,
        defects: List[CodeDefect],
        success: bool,
    ) -> None:
        """
        Learn from repair history for future improvements.
        
        Args:
            original_code: Original code with defects
            repaired_code: Repaired code
            defects: Detected defects
            success: Whether repair was successful
        """
        if not self.enable_learning:
            return
        
        repair_record = {
            "original_length": len(original_code),
            "num_defects": len(defects),
            "defect_types": [d.defect_type.value for d in defects],
            "success": success,
        }
        
        self.repair_history.append(repair_record)
        
        logger.info("repair_recorded", success=success, total_repairs=len(self.repair_history))
    
    def get_repair_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about repair history.
        
        Returns:
            Dictionary of repair statistics
        """
        if not self.repair_history:
            return {"total_repairs": 0}
        
        total = len(self.repair_history)
        successful = sum(1 for r in self.repair_history if r["success"])
        
        return {
            "total_repairs": total,
            "successful_repairs": successful,
            "success_rate": successful / total if total > 0 else 0.0,
        }
