"""Tests for rego_ast_parser module.

Tests follow the architecture spec requirements:
- Extract helpers with AST parser (not regex)
- Get accurate source spans
- Handle all Rego syntax correctly
"""

import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestRegoASTParser:
    """Tests for RegoASTParser class."""
    
    @pytest.fixture
    def sample_rego(self):
        """Sample Rego code for testing."""
        return '''package lib.test

import rego.v1

# Converts an array to a set
to_set(arr) := {member | some member in arr}

# Checks if needle is in haystack
# Returns true if found
included_in(needle, haystack) if {
    some item in haystack
    item == needle
}

# A rule with contains keyword
deny contains result if {
    some x in input.items
    x.bad
    result := {"msg": "bad item found"}
}

# Private helper (should be skipped)
_private_helper(x) := x * 2

# A default rule
default allow := false

# Simple assignment
simple_value := 42
'''
    
    @pytest.fixture
    def sample_rego_file(self, sample_rego):
        """Create a temporary Rego file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rego', delete=False) as f:
            f.write(sample_rego)
            f.flush()
            yield Path(f.name)
    
    def test_import_available(self):
        """Test that module can be imported."""
        from rego_ast_parser import RegoASTParser, RegoModule, RegoRule
        assert RegoASTParser is not None
    
    def test_parse_file_extracts_package(self, sample_rego_file):
        """Test that package is correctly extracted."""
        from rego_ast_parser import RegoASTParser
        
        try:
            parser = RegoASTParser()
        except RuntimeError:
            pytest.skip("OPA not available")
        
        module = parser.parse_file(sample_rego_file)
        
        assert module is not None
        assert module.package == "lib.test"
    
    def test_parse_file_extracts_imports(self, sample_rego_file):
        """Test that imports are correctly extracted."""
        from rego_ast_parser import RegoASTParser
        
        try:
            parser = RegoASTParser()
        except RuntimeError:
            pytest.skip("OPA not available")
        
        module = parser.parse_file(sample_rego_file)
        
        assert module is not None
        assert "rego.v1" in module.imports
    
    def test_parse_file_extracts_rules(self, sample_rego_file):
        """Test that rules are correctly extracted."""
        from rego_ast_parser import RegoASTParser
        
        try:
            parser = RegoASTParser()
        except RuntimeError:
            pytest.skip("OPA not available")
        
        module = parser.parse_file(sample_rego_file)
        
        assert module is not None
        
        rule_names = [r.name for r in module.rules]
        
        # Should have public rules
        assert "to_set" in rule_names
        assert "included_in" in rule_names
        assert "deny" in rule_names
        
        # Private rules ARE in the AST (filtering happens at build stage)
        # The is_private property can be used to filter them
        private_rules = [r for r in module.rules if r.is_private]
        public_rules = [r for r in module.rules if not r.is_private]
        assert len(public_rules) > 0
    
    def test_rule_has_correct_args(self, sample_rego_file):
        """Test that function arguments are correctly extracted."""
        from rego_ast_parser import RegoASTParser
        
        try:
            parser = RegoASTParser()
        except RuntimeError:
            pytest.skip("OPA not available")
        
        module = parser.parse_file(sample_rego_file)
        
        # Find to_set rule
        to_set = next((r for r in module.rules if r.name == "to_set"), None)
        assert to_set is not None
        assert to_set.args == ["arr"]
        
        # Find included_in rule
        included_in = next((r for r in module.rules if r.name == "included_in"), None)
        assert included_in is not None
        assert included_in.args == ["needle", "haystack"]
    
    def test_rule_signature(self, sample_rego_file):
        """Test that signatures are correctly generated."""
        from rego_ast_parser import RegoASTParser
        
        try:
            parser = RegoASTParser()
        except RuntimeError:
            pytest.skip("OPA not available")
        
        module = parser.parse_file(sample_rego_file)
        
        to_set = next((r for r in module.rules if r.name == "to_set"), None)
        assert to_set is not None
        assert to_set.signature == "to_set(arr)"
        
        included_in = next((r for r in module.rules if r.name == "included_in"), None)
        assert included_in is not None
        assert included_in.signature == "included_in(needle, haystack)"
    
    def test_rule_has_line_numbers(self, sample_rego_file):
        """Test that line numbers are extracted."""
        from rego_ast_parser import RegoASTParser
        
        try:
            parser = RegoASTParser()
        except RuntimeError:
            pytest.skip("OPA not available")
        
        module = parser.parse_file(sample_rego_file)
        
        to_set = next((r for r in module.rules if r.name == "to_set"), None)
        assert to_set is not None
        assert to_set.start_line > 0
        assert to_set.end_line >= to_set.start_line
    
    def test_rule_is_private(self, sample_rego_file):
        """Test that private rule detection works."""
        from rego_ast_parser import RegoRule
        
        public_rule = RegoRule(
            name="public_helper",
            args=[],
            start_line=1,
            end_line=1,
            is_default=False,
            doc_comment=None,
            source_file="test.rego"
        )
        assert not public_rule.is_private
        
        private_rule = RegoRule(
            name="_private_helper",
            args=[],
            start_line=1,
            end_line=1,
            is_default=False,
            doc_comment=None,
            source_file="test.rego"
        )
        assert private_rule.is_private


class TestExtractFunctionBody:
    """Tests for extract_function_body helper."""
    
    def test_extract_single_line_function(self):
        """Test extracting a single-line function."""
        from rego_ast_parser import extract_function_body
        
        source = '''package test

to_set(arr) := {x | some x in arr}

other_func := true
'''
        body = extract_function_body(source, 3, 3)
        assert "to_set(arr)" in body
        assert "{x | some x in arr}" in body
    
    def test_extract_multiline_function(self):
        """Test extracting a multi-line function."""
        from rego_ast_parser import extract_function_body
        
        source = '''package test

my_func(x) if {
    x > 0
    x < 100
}

other := true
'''
        body = extract_function_body(source, 3, 6)
        assert "my_func(x)" in body
        assert "x > 0" in body
        assert "x < 100" in body


class TestRealPolicyFiles:
    """Tests with real policy files if available."""
    
    @pytest.fixture
    def real_lib_dir(self):
        """Get real policy/lib directory if it exists."""
        repo_root = Path(__file__).parent.parent
        lib_dir = repo_root / "policy" / "lib"
        
        if lib_dir.exists() and any(lib_dir.glob("*.rego")):
            return lib_dir
        
        pytest.skip("No real policy files available")
    
    def test_parse_real_files(self, real_lib_dir):
        """Test parsing real policy files."""
        from rego_ast_parser import RegoASTParser
        
        try:
            parser = RegoASTParser()
        except RuntimeError:
            pytest.skip("OPA not available")
        
        # Parse first non-test file
        rego_files = [f for f in real_lib_dir.glob("*.rego") if "_test" not in f.name]
        
        if not rego_files:
            pytest.skip("No non-test Rego files found")
        
        module = parser.parse_file(rego_files[0])
        
        assert module is not None
        assert module.package != ""
        print(f"\nParsed {rego_files[0].name}: {len(module.rules)} rules")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

