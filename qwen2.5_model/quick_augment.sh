#!/bin/bash
# Quick script to increase dataset size by adjusting rates in generate_dataset.py

echo "Current dataset: 268 examples"
echo ""
echo "To get more examples, edit generate_dataset.py and change:"
echo ""
echo "Line ~600: Change 0.6 to 0.9 (refactor examples)"
echo "Line ~606: Change 0.3 to 0.7 (instruction variations)"
echo ""
echo "Then run: uv run python qwen2.5_model/generate_dataset.py"
echo ""
echo "Expected results:"
echo "  - 0.6 → 0.9: ~50% more refactor examples"
echo "  - 0.3 → 0.7: ~130% more variations"
echo "  - Total: ~400-500 examples"
echo ""

# Optionally make the changes automatically
if [ "$1" == "--apply" ]; then
    echo "Applying changes..."
    cd "$(dirname "$0")/.."
    
    # Increase refactor rate from 0.6 to 0.9
    sed -i.bak 's/if random.random() < 0.6:  # 60% of rules get refactor examples/if random.random() < 0.9:  # 90% of rules get refactor examples/' qwen2.5_model/generate_dataset.py
    
    # Increase variation rate from 0.3 to 0.7
    sed -i.bak 's/if random.random() < 0.3:  # 30% chance for instruction variation/if random.random() < 0.7:  # 70% chance for instruction variation/' qwen2.5_model/generate_dataset.py
    
    echo "Changes applied! Run: uv run python qwen2.5_model/generate_dataset.py"
    echo "Backup files created with .bak extension"
else
    echo "Run with --apply to automatically make these changes"
fi

