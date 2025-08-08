"""Built-in evaluation functions for the AutoTuning system."""

import json
import re
from typing import Dict, Any
from difflib import SequenceMatcher
import logging

from ..dataset import DatasetItem
from ..evaluator import ItemEvaluation, ItemResult

logger = logging.getLogger(__name__)


def similarity_evaluation(
    item_result: ItemResult,
    dataset_item: DatasetItem,
    context: Dict[str, Any]
) -> ItemEvaluation:
    """Evaluate using text similarity metrics."""
    if not dataset_item.expected_output:
        return ItemEvaluation(
            item_id=dataset_item.id,
            score=0.0,
            error="No expected output for similarity evaluation"
        )
    
    # Calculate similarity ratio
    similarity = SequenceMatcher(
        None, 
        item_result.agent_output.strip(), 
        dataset_item.expected_output.strip()
    ).ratio()
    
    # Additional metrics
    metric_scores = {
        "similarity": similarity,
        "length_ratio": min(len(item_result.agent_output), len(dataset_item.expected_output)) / 
                       max(len(item_result.agent_output), len(dataset_item.expected_output), 1)
    }
    
    # Check for exact match bonus
    if item_result.agent_output.strip().lower() == dataset_item.expected_output.strip().lower():
        metric_scores["exact_match"] = 1.0
        similarity = max(similarity, 0.95)  # Boost similarity for exact matches
    else:
        metric_scores["exact_match"] = 0.0
    
    return ItemEvaluation(
        item_id=dataset_item.id,
        score=similarity,
        metric_scores=metric_scores,
        feedback=f"Similarity: {similarity:.3f}, Length ratio: {metric_scores['length_ratio']:.3f}"
    )


def llm_evaluation(
    item_result: ItemResult,
    dataset_item: DatasetItem,
    context: Dict[str, Any]
) -> ItemEvaluation:
    """Evaluate using LLM judge."""
    llm_config = context.get('llm_evaluator_config', {})
    
    if not llm_config:
        return ItemEvaluation(
            item_id=dataset_item.id,
            score=0.0,
            error="No LLM evaluator configuration provided"
        )
    
    # Construct evaluation prompt
    prompt = f"""
    Evaluate the following agent response for the given task:
    
    Task: {dataset_item.input_data}
    Expected: {dataset_item.expected_output or "No expected output provided"}
    Agent Response: {item_result.agent_output}
    
    Rate from 0.0 to 1.0 on:
    - Correctness: How accurate is the response?
    - Helpfulness: How useful is the response?
    - Clarity: How clear and understandable is the response?
    
    Additional evaluation criteria:
    {json.dumps(dataset_item.evaluation_criteria, indent=2) if dataset_item.evaluation_criteria else "None"}
    
    Return JSON in this exact format:
    {{"correctness": 0.0-1.0, "helpfulness": 0.0-1.0, "clarity": 0.0-1.0, "overall": 0.0-1.0, "feedback": "brief explanation"}}
    """
    
    try:
        # Call LLM (this would need to be implemented with actual LLM integration)
        response = _call_llm(prompt, llm_config)
        scores = _parse_llm_scores(response)
        
        return ItemEvaluation(
            item_id=dataset_item.id,
            score=scores["overall"],
            metric_scores={k: v for k, v in scores.items() if k not in ["overall", "feedback"]},
            feedback=scores.get("feedback", "")
        )
        
    except Exception as e:
        logger.error(f"LLM evaluation failed for item {dataset_item.id}: {e}")
        return ItemEvaluation(
            item_id=dataset_item.id,
            score=0.0,
            error=f"LLM evaluation failed: {str(e)}"
        )


def rule_based_evaluation(
    item_result: ItemResult,
    dataset_item: DatasetItem,
    context: Dict[str, Any]
) -> ItemEvaluation:
    """Evaluate using predefined rules."""
    agent_output = item_result.agent_output
    criteria = dataset_item.evaluation_criteria or {}
    
    scores = {}
    feedback_parts = []
    
    # Length constraints
    if 'min_length' in criteria:
        min_len = criteria['min_length']
        meets_min = len(agent_output) >= min_len
        scores['length_adequate'] = 1.0 if meets_min else 0.0
        feedback_parts.append(f"Min length ({min_len}): {'✓' if meets_min else '✗'}")
    
    if 'max_length' in criteria:
        max_len = criteria['max_length']
        meets_max = len(agent_output) <= max_len
        scores['length_within_max'] = 1.0 if meets_max else 0.0
        feedback_parts.append(f"Max length ({max_len}): {'✓' if meets_max else '✗'}")
    
    # Required keywords
    if 'required_keywords' in criteria:
        keywords = criteria['required_keywords']
        keywords_found = sum(1 for kw in keywords if kw.lower() in agent_output.lower())
        scores['keywords_coverage'] = keywords_found / len(keywords) if keywords else 0.0
        feedback_parts.append(f"Keywords ({keywords_found}/{len(keywords)})")
    
    # Forbidden content
    if 'forbidden_content' in criteria:
        forbidden = criteria['forbidden_content']
        has_forbidden = any(forbidden_item.lower() in agent_output.lower() for forbidden_item in forbidden)
        scores['content_compliance'] = 0.0 if has_forbidden else 1.0
        feedback_parts.append(f"Content compliance: {'✗' if has_forbidden else '✓'}")
    
    # Format validation
    if 'expected_format' in criteria:
        format_type = criteria['expected_format']
        format_valid = _validate_format(agent_output, format_type)
        scores['format_compliance'] = 1.0 if format_valid else 0.0
        feedback_parts.append(f"Format ({format_type}): {'✓' if format_valid else '✗'}")
    
    # Pattern matching
    if 'required_patterns' in criteria:
        patterns = criteria['required_patterns']
        patterns_found = 0
        for pattern in patterns:
            if re.search(pattern, agent_output, re.IGNORECASE):
                patterns_found += 1
        scores['pattern_coverage'] = patterns_found / len(patterns) if patterns else 0.0
        feedback_parts.append(f"Patterns ({patterns_found}/{len(patterns)})")
    
    # Boolean criteria
    for key, expected_value in criteria.items():
        if key.startswith('should_') and isinstance(expected_value, bool):
            if key == 'should_acknowledge_order':
                has_acknowledgment = bool(re.search(r'#\d+', agent_output))
                scores[key] = 1.0 if has_acknowledgment == expected_value else 0.0
                feedback_parts.append(f"Order acknowledgment: {'✓' if has_acknowledgment else '✗'}")
            
            elif key == 'should_offer_help':
                help_phrases = ['help', 'assist', 'support', 'resolve']
                has_help_offer = any(phrase in agent_output.lower() for phrase in help_phrases)
                scores[key] = 1.0 if has_help_offer == expected_value else 0.0
                feedback_parts.append(f"Help offer: {'✓' if has_help_offer else '✗'}")
    
    # Calculate overall score
    overall = sum(scores.values()) / len(scores) if scores else 0.0
    
    return ItemEvaluation(
        item_id=dataset_item.id,
        score=overall,
        metric_scores=scores,
        feedback=" | ".join(feedback_parts) if feedback_parts else f"Overall: {overall:.3f}"
    )


def _call_llm(prompt: str, llm_config: Dict[str, Any]) -> str:
    """Call LLM API (placeholder implementation)."""
    # This would integrate with actual LLM APIs
    # For now, return a mock response
    logger.warning("Using mock LLM response - implement actual LLM integration")
    
    return '''
    {
        "correctness": 0.8,
        "helpfulness": 0.7,
        "clarity": 0.9,
        "overall": 0.8,
        "feedback": "Response is accurate and clear, could be more helpful"
    }
    '''


def _parse_llm_scores(response: str) -> Dict[str, Any]:
    """Parse LLM evaluation response."""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            scores = json.loads(json_str)
            
            # Validate scores are in valid range
            for key, value in scores.items():
                if key != "feedback" and isinstance(value, (int, float)):
                    scores[key] = max(0.0, min(1.0, float(value)))
            
            return scores
        else:
            raise ValueError("No JSON found in LLM response")
            
    except Exception as e:
        logger.error(f"Failed to parse LLM scores: {e}")
        return {
            "correctness": 0.5,
            "helpfulness": 0.5,
            "clarity": 0.5,
            "overall": 0.5,
            "feedback": f"Failed to parse LLM response: {str(e)}"
        }


def _validate_format(text: str, format_type: str) -> bool:
    """Validate text format."""
    text = text.strip()
    
    if format_type == "json":
        try:
            json.loads(text)
            return True
        except:
            return False
    
    elif format_type == "yaml":
        try:
            import yaml
            yaml.safe_load(text)
            return True
        except:
            return False
    
    elif format_type == "xml":
        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(text)
            return True
        except:
            return False
    
    elif format_type == "markdown":
        # Basic markdown validation - check for common markdown elements
        markdown_indicators = ['#', '*', '**', '[', '](', '`', '```']
        return any(indicator in text for indicator in markdown_indicators)
    
    elif format_type == "code":
        # Basic code validation - check for common code patterns
        code_indicators = ['{', '}', '(', ')', 'def ', 'function ', 'class ', 'import ', 'from ']
        return any(indicator in text for indicator in code_indicators)
    
    else:
        logger.warning(f"Unknown format type: {format_type}")
        return True  # Default to valid for unknown formats