# Natural PDF Evaluation Process Improvements

## Overview
Based on the analysis of LLM evaluation results, we've identified key areas where the evaluation process needed updating to reflect Natural PDF's modern capabilities and best practices.

## Key Issues Identified in Current Evaluations

1. **Over-reliance on TATR**: Nearly every table extraction defaulted to `analyze_layout('tatr')` 
2. **Generic placeholder code**: Using "AnchorText" instead of actual text from PDFs
3. **Missing modern features**: No use of Guides API despite its advantages
4. **Outdated difficulty assessments**: Flagging tiny fonts and RTL as difficult when they're now well-supported

## Improvements Made

### 1. Updated Cheatsheet (`LLM_NaturalPDF_CheatSheet.md`)
- **Prioritized Guides API** as the preferred method for table extraction
- Added section on why Guides is better than TATR (faster, more control, robust)
- Emphasized `.extract_table()` (singular) over `.extract_tables()`
- Added notes about recent improvements (tiny text support, RTL handling)
- **Added text layer removal options** for handling corrupted PDFs with `text_layer=False` or `remove_text_layer()`

### 2. Enhanced Workflows (`LLM_NaturalPDF_Workflows.md`)
- Added 4 new Guides-based examples at the top
- Updated existing examples to use modern patterns
- Showed practical patterns like `snap_to_whitespace()`
- Demonstrated fallback strategies when TATR isn't needed
- **Added workflow for handling corrupted text layers** (e.g., "(cid:xxx)" issues)
- **Added nuanced examples of `until=` parameter** - showing when it's helpful vs unnecessary

### 3. New Decision Tree (`extraction_decision_tree.md`)
- Clear guidance on when to use each extraction approach
- Specific criteria for choosing Guides vs direct extraction vs TATR
- Quality checklist to ensure practical, working code
- **Guidance on when to discard corrupted text layers**
- **Clarified when `until=` is beneficial** (stopping before sections) vs optional (going to page edge)

### 4. Updated Prompts (`llm_enrich.py`)
- Emphasized modern features and patterns
- Required use of actual text from PDFs, not placeholders
- Added guidance on handling corrupted text layers
- **Clarified `until=` is only needed when stopping before page edge**
- Stressed consistency between document and page-level code

### 5. Quality Evaluation (`evaluate_quality.py`)
- Automated scoring system (0-12 points)
- Tracks usage of modern features (Guides, snap_to_whitespace)
- Penalizes placeholder text
- Rewards practical patterns (parent navigation, real anchors)
- **Made `until=` usage a bonus, not a requirement**

### 6. Retry Mechanism (`llm_enrich_with_retry.py`)
- Automatically retries low-scoring suggestions
- Provides specific feedback on what to improve
- Can set custom quality thresholds
- Limits retries to prevent infinite loops

## Results After Improvements

Initial evaluation showed:
- 0% Guides API usage
- 56.7% TATR reliance
- Average score: 3.4/12

After improvements:
- 42.2% Guides API usage ✅
- 20.0% TATR usage ✅
- Average score: 5.6/12 ✅
- More realistic difficulty assessments

## Next Steps

1. Run full evaluation suite with updated materials
2. Consider higher retry thresholds for better quality
3. Add more examples of edge cases to workflows
4. Continue refining based on LLM feedback patterns

## Expected Outcomes

When you run the evaluation again with these improvements:

1. **More Guides API usage**: Should see 60-80% of table extractions using Guides
2. **Less TATR reliance**: TATR only for genuinely complex multi-table scenarios
3. **Real text anchors**: No more "AnchorText" placeholders
4. **Better difficulty assessment**: Fewer false positives for "difficult" elements
5. **Higher quality scores**: Average score should increase from ~3/12 to ~8/12

## How to Use

1. **Before running evaluation**: Review the updated cheatsheet and workflows
2. **During evaluation**: The LLM will now prefer modern approaches automatically
3. **After evaluation**: Run `evaluate_quality.py` to assess improvement

```bash
# Run the evaluation
python -m tools.bad_pdf_eval.eval_suite

# Enrich with LLM
python -m tools.bad_pdf_eval.llm_enrich --model gpt-4o

# Collate results
python -m tools.bad_pdf_eval.collate_summaries

# Analyze quality
python -m tools.bad_pdf_eval.evaluate_quality
```

## Future Enhancements

1. **Add more Guides examples** for specific industries (invoices, government forms, scientific papers)
2. **Create automated tests** from the LLM's test case suggestions
3. **Track performance metrics** (extraction speed, accuracy) not just code quality
4. **Build a feedback loop** where successful extractions become new workflow examples

## Key Takeaway

The main insight is that Natural PDF has evolved significantly, and the evaluation process needs to showcase these improvements. By prioritizing Guides API and modern patterns, we can demonstrate that many "difficult" PDFs are actually quite manageable with the right approach. 