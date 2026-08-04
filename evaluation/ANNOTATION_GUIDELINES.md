# Manual Annotation Guidelines

## Faithfulness score
- 0: Explanation contradicts the final answer or is unrelated.
- 1: Little meaningful support or major errors.
- 2: Partial support with weak, missing, or flawed reasoning.
- 3: Clear, correct, and consistent justification.

## Logical consistency score
- 0: Internally contradictory.
- 1: Major reasoning errors.
- 2: Mostly coherent with a minor issue.
- 3: Fully coherent.

## Completeness score
- 0: No meaningful reasoning.
- 1: Important steps are missing.
- 2: Mostly complete.
- 3: Sufficiently complete.

## Failure category
Use one primary label:
`none`, `unsupported_justification`, `factual_hallucination`,
`incomplete_reasoning`, `logical_error`, `arithmetic_error`,
`contradictory_explanation`, `answer_explanation_mismatch`,
`instruction_following_failure`, or `other`.
