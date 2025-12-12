```python
python src/infer_two_stage.py \
    --stage1-model models/stage1-context-inference \
    --stage2-model models/stage2-rule-generation \
    --instruction "Write a Rego rule that checks the status of all tasks in a pipelineRun attestation and fails if the status is 'failure'. Make sure it belongs to ONLY the '@redhat' collection."

python src/infer_two_stage.py \
    --stage1-model models/stage1-context-inference \
    --stage2-model models/stage2-rule-generation \
    --instruction "Block images with critical or high severity CVEs that have available patches"

python src/infer_two_stage.py \
    --stage1-model models/stage1-context-inference \
    --stage2-model models/stage2-rule-generation \
    --instruction "Verify the SBOM doesn't contain any packages from the disallowed list"

python src/infer_two_stage.py \
    --stage1-model models/stage1-context-inference \
    --stage2-model models/stage2-rule-generation \
    --instruction "Check that all pipeline tasks completed successfully without failures"
```
