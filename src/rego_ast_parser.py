"""Parse Rego files using OPA's AST parser.

Per architecture spec:
- Extract helpers with an AST parser (not regex)
- Get accurate source spans
- Handle all Rego syntax correctly
"""

import base64
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RegoRule:
    """A parsed Rego rule/function from AST."""
    name: str
    args: List[str]
    start_line: int
    end_line: int
    is_default: bool
    doc_comment: Optional[str]
    source_file: str
    
    @property
    def signature(self) -> str:
        """Generate signature string."""
        if self.args:
            return f"{self.name}({', '.join(self.args)})"
        return self.name
    
    @property
    def is_private(self) -> bool:
        """Check if rule is private (starts with _)."""
        return self.name.startswith("_")


@dataclass
class RegoModule:
    """A parsed Rego module from AST."""
    package: str
    imports: List[str]
    rules: List[RegoRule]
    source_file: str
    comments: Dict[int, str]  # line -> comment text


class RegoASTParser:
    """Parse Rego files using OPA's AST.
    
    Uses `opa parse --format json` to get accurate AST.
    """
    
    def __init__(self, opa_path: str = "opa"):
        """Initialize parser.
        
        Args:
            opa_path: Path to opa binary
        """
        self.opa_path = opa_path
        self._verify_opa()
    
    def _verify_opa(self):
        """Verify OPA is available."""
        try:
            result = subprocess.run(
                [self.opa_path, "version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("OPA not available")
        except FileNotFoundError:
            raise RuntimeError(f"OPA binary not found at: {self.opa_path}")
    
    def parse_file(self, file_path: Path) -> Optional[RegoModule]:
        """Parse a Rego file and return structured module.
        
        Args:
            file_path: Path to .rego file
            
        Returns:
            RegoModule or None if parsing fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None
        
        # Run OPA parse with locations and comments
        try:
            result = subprocess.run(
                [self.opa_path, "parse", "--format", "json", 
                 "--json-include", "locations,comments", str(file_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
        except subprocess.TimeoutExpired:
            print(f"Warning: OPA parse timed out for {file_path}")
            return None
        
        if result.returncode != 0:
            # Try to parse error
            try:
                error_data = json.loads(result.stdout)
                if "errors" in error_data:
                    print(f"Warning: OPA parse error for {file_path}: {error_data['errors']}")
            except:
                pass
            return None
        
        try:
            ast = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        
        return self._parse_ast(ast, str(file_path))
    
    def _parse_ast(self, ast: Dict, source_file: str) -> RegoModule:
        """Parse OPA AST JSON into RegoModule.
        
        Args:
            ast: Parsed JSON from OPA
            source_file: Source file path
            
        Returns:
            RegoModule
        """
        # Extract package
        package = self._extract_package(ast.get("package", {}))
        
        # Extract imports
        imports = self._extract_imports(ast.get("imports", []))
        
        # Extract comments
        comments = self._extract_comments(ast.get("comments", []))
        
        # Extract rules
        rules = self._extract_rules(ast.get("rules", []), source_file, comments)
        
        return RegoModule(
            package=package,
            imports=imports,
            rules=rules,
            source_file=source_file,
            comments=comments,
        )
    
    def _extract_package(self, pkg_ast: Dict) -> str:
        """Extract package name from AST."""
        if not pkg_ast:
            return "unknown"
        
        path = pkg_ast.get("path", [])
        parts = []
        
        for p in path:
            if p.get("type") == "string":
                parts.append(p.get("value", ""))
            elif p.get("type") == "var" and p.get("value") != "data":
                parts.append(p.get("value", ""))
        
        return ".".join(parts) if parts else "unknown"
    
    def _extract_imports(self, imports_ast: List) -> List[str]:
        """Extract import statements from AST."""
        imports = []
        
        for imp in imports_ast:
            path = imp.get("path", {})
            if path.get("type") == "ref":
                parts = []
                for v in path.get("value", []):
                    if v.get("type") == "string":
                        parts.append(v.get("value", ""))
                    elif v.get("type") == "var":
                        parts.append(v.get("value", ""))
                if parts:
                    imports.append(".".join(parts))
        
        return imports
    
    def _extract_comments(self, comments_ast: List) -> Dict[int, str]:
        """Extract comments indexed by line number."""
        comments = {}
        
        for comment in comments_ast:
            loc = comment.get("location", {})
            line = loc.get("row", 0)
            
            # Decode base64 text
            text_b64 = comment.get("text") or loc.get("text", "")
            try:
                text = base64.b64decode(text_b64).decode("utf-8").strip()
                # Remove # prefix if present
                if text.startswith("#"):
                    text = text[1:].strip()
                comments[line] = text
            except:
                pass
        
        return comments
    
    def _extract_rules(
        self, 
        rules_ast: List, 
        source_file: str,
        comments: Dict[int, str]
    ) -> List[RegoRule]:
        """Extract rules from AST."""
        rules = []
        seen_names = set()  # Track to avoid duplicates
        
        for rule_ast in rules_ast:
            head = rule_ast.get("head", {})
            name = head.get("name", "")
            
            if not name:
                # Try ref for rules like `deny contains result`
                ref = head.get("ref", [])
                if ref:
                    for r in ref:
                        if r.get("type") == "var":
                            name = r.get("value", "")
                            break
            
            if not name:
                continue
            
            # Skip if we've already seen this name (first definition only)
            if name in seen_names:
                continue
            seen_names.add(name)
            
            # Extract args
            args = []
            for arg in head.get("args", []):
                if arg.get("type") == "var":
                    args.append(arg.get("value", ""))
            
            # Get location
            loc = rule_ast.get("location", head.get("location", {}))
            start_line = loc.get("row", 0)
            
            # Estimate end line from body
            end_line = self._estimate_end_line(rule_ast, start_line)
            
            # Check if default rule
            is_default = rule_ast.get("default", False)
            
            # Get doc comment (comments immediately before the rule)
            doc_comment = self._get_doc_comment(start_line, comments)
            
            rules.append(RegoRule(
                name=name,
                args=args,
                start_line=start_line,
                end_line=end_line,
                is_default=is_default,
                doc_comment=doc_comment,
                source_file=source_file,
            ))
        
        return rules
    
    def _estimate_end_line(self, rule_ast: Dict, start_line: int) -> int:
        """Estimate the end line of a rule by finding max line in AST."""
        max_line = start_line
        
        def find_max_line(node):
            nonlocal max_line
            if isinstance(node, dict):
                loc = node.get("location", {})
                row = loc.get("row", 0)
                if row > max_line:
                    max_line = row
                for v in node.values():
                    find_max_line(v)
            elif isinstance(node, list):
                for item in node:
                    find_max_line(item)
        
        find_max_line(rule_ast)
        return max_line
    
    def _get_doc_comment(self, rule_line: int, comments: Dict[int, str]) -> Optional[str]:
        """Get documentation comment for a rule.
        
        Looks for comments immediately before the rule.
        """
        doc_lines = []
        
        # Look for comments in the 5 lines before the rule
        for line in range(rule_line - 1, max(0, rule_line - 6), -1):
            if line in comments:
                doc_lines.insert(0, comments[line])
            else:
                # Stop at first non-comment line
                break
        
        return " ".join(doc_lines) if doc_lines else None


def extract_function_body(source: str, start_line: int, end_line: int) -> str:
    """Extract function body from source code.
    
    Args:
        source: Full source code
        start_line: Starting line (1-indexed)
        end_line: Ending line (1-indexed)
        
    Returns:
        Function body text
    """
    lines = source.split('\n')
    
    # Adjust for 1-indexed lines
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    
    return '\n'.join(lines[start_idx:end_idx])

