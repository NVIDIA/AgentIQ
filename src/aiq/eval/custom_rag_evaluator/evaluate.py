import asyncio
import logging
from tqdm import tqdm

from langchain_core.language_models import BaseChatModel

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema import SystemMessage, HumanMessage

from aiq.eval.evaluator.evaluator_model import EvalInput
from aiq.eval.evaluator.evaluator_model import EvalOutput
from aiq.eval.evaluator.evaluator_model import EvalOutputItem
from aiq.eval.evaluator.evaluator_model import EvalInputItem
from aiq.eval.utils.tqdm_position_registry import TqdmPositionRegistry

logger = logging.getLogger(__name__)


def evaluation_prompt(judge_llm_prompt: str, question: str, answer_description: str, generated_answer: str, format_instructions: str, default_scoring: bool):
    """
    This function generates a prompt for the judge LLM to evaluate the generated answer.
    """

    DEFAULT_SCORING_INSTRUCTIONS = """
    The coverage score is a measure of how well the generated answer covers the critical aspects mentioned in the expected answer. A low coverage score indicates that the generated answer misses critical aspects of the expected answer. A middle coverage score indicates that the generated answer covers some of the must-haves of the expected answer but lacks other details. A high coverage score indicates that all of the expected aspects are present in the generated answer.
    The correctness score is a measure of how well the generated answer matches the expected answer. A low correctness score indicates that the generated answer is incorrect or does not match the expected answer. A middle correctness score indicates that the generated answer is correct but lacks some details. A high correctness score indicates that the generated answer is exactly the same as the expected answer.
    The relevance score is a measure of how well the generated answer is relevant to the question. A low relevance score indicates that the generated answer is not relevant to the question. A middle relevance score indicates that the generated answer is somewhat relevant to the question. A high relevance score indicates that the generated answer is exactly relevant to the question.
    The reasoning is a 1-2 sentence explanation for the scoring.
    """

    DEFAULT_EVAL_PROMPT = (
        f"You are an intelligent assistant that responds strictly in JSON format."
        f"Judge based on the following scoring rubric: {DEFAULT_SCORING_INSTRUCTIONS}"
        f"{judge_llm_prompt}\n"
        f"{format_instructions}\n"
        f"Here is the user's query: {question}"
        f"Here is the description of the expected answer: {answer_description}"
        f"Here is the generated answer: {generated_answer}"
    )

    EVAL_PROMPT = (
        f"You are an intelligent assistant that responds strictly in JSON format. {judge_llm_prompt}\n"
        f"{format_instructions}\n"
        f"Here is the user's query: {question}"
        f"Here is the description of the expected answer: {answer_description}"
        f"Here is the generated answer: {generated_answer}"
    )

    return EVAL_PROMPT if not default_scoring else DEFAULT_EVAL_PROMPT

class TunableRagEvaluator:
    '''Customizable RAG evaluator class with customizable LLM prompt for scoring.'''

    def __init__(self, llm: BaseChatModel, judge_llm_prompt: str, max_concurrency: int, default_scoring: bool, default_score_weights: dict):
        self.llm = llm
        self.max_concurrency = max_concurrency
        self.judge_llm_prompt = judge_llm_prompt
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.default_scoring = default_scoring
        # Set equal weights for each score
        self.default_score_weights = {
            "coverage": 1/3,
            "correctness": 1/3,
            "relevance": 1/3
        }

    async def evaluate(self, eval_input: EvalInput) -> EvalOutput:
        '''Evaluate function'''

        async def process_item(item):
            """Compute RAG evaluation for an individual item"""
            question = item.input_obj
            answer_description = item.expected_output_obj
            generated_answer = item.output_obj

            # Call judge LLM to generate score
            score = 0.0

            default_evaluation_schema = [
                ResponseSchema(name="coverage_score", description="Score for the coverage of all critical aspects mentioned in the expected answer. Ex. 0.5", type="float"),
                ResponseSchema(name="correctness_score", description="Score for the accuracy of the generated answer compared to the expected answer. Ex. 0.5", type="float"),
                ResponseSchema(name="relevance_score", description="Score for the relevance of the generated answer to the question. Ex. 0.5", type="float"),
                ResponseSchema(name="reasoning", description="1-2 summarized sentences of reasoning for the scores. Ex. 'The generated answer covers all critical aspects mentioned in the expected answer, is correct, and is relevant to the question.'", type="string"),
            ]

            custom_evaluation_schema = [
                ResponseSchema(name="score", description="Score for the generated answer. Ex. 0.5", type="float"),
                ResponseSchema(name="reasoning", description="1-2 sentence reasoning for the score. Ex. 'The generated answer is exactly the same as the description of the expected answer.'", type="string"),
            ]

            if self.default_scoring:
                evaluation_schema = default_evaluation_schema
            else:
                evaluation_schema = custom_evaluation_schema

            llm_input_response_parser = StructuredOutputParser.from_response_schemas(evaluation_schema)
            format_instructions = llm_input_response_parser.get_format_instructions()

            eval_prompt = evaluation_prompt(
                judge_llm_prompt=self.judge_llm_prompt,
                question = question,
                answer_description = answer_description,
                generated_answer = generated_answer,
                format_instructions=format_instructions,
                default_scoring=self.default_scoring
            )

            messages = [SystemMessage(content="You must respond only in JSON format."), HumanMessage(content=eval_prompt)]

            response = await self.llm.ainvoke(messages)
            try:
                parsed_response = llm_input_response_parser.parse(response.content)
                if self.default_scoring:
                    try:
                        coverage_score = parsed_response["coverage_score"]
                        correctness_score = parsed_response["correctness_score"]
                        relevance_score = parsed_response["relevance_score"]
                        reasoning = parsed_response["reasoning"]
                    except KeyError as e:
                        logger.error(f"Missing required keys in default scoring response: {', '.join(str(arg) for arg in e.args)}")
                        reasoning = f"Error in evaluator from parsing judge LLM response. Missing required key(s): {', '.join(str(arg) for arg in e.args)}"
                        raise

                    # Calculate score
                    coverage_weight = self.default_score_weights.get("coverage", 0)
                    correctness_weight = self.default_score_weights.get("correctness", 0) 
                    relevance_weight = self.default_score_weights.get("relevance", 0)
                    
                    if round(coverage_weight + correctness_weight + relevance_weight, 2) != 1:
                        logger.warning("The sum of the default score weights is not 1. The weights will be normalized.")
                        coverage_weight = coverage_weight / (coverage_weight + correctness_weight + relevance_weight)
                        correctness_weight = correctness_weight / (coverage_weight + correctness_weight + relevance_weight)
                        relevance_weight = relevance_weight / (coverage_weight + correctness_weight + relevance_weight)

                    score = (coverage_weight * coverage_score + correctness_weight * correctness_score + relevance_weight * relevance_score)

                else:
                    try:
                        score = parsed_response["score"]
                        reasoning = parsed_response["reasoning"]
                    except KeyError as e:
                        logger.error(f"Missing required keys in custom scoring response: {', '.join(str(arg) for arg in e.args)}")
                        reasoning = f"Error in evaluator from parsing judge LLM response. Missing required key(s): {', '.join(str(arg) for arg in e.args)}"
                        raise
            except (KeyError, ValueError) as e:
                logger.error(f"Error parsing judge LLM response: {e}")
                score = 0.0
                reasoning = "Error in evaluator from parsing judge LLM response."

            if self.default_scoring:
                reasoning = {
                    "question": question,
                    "answer_description": answer_description,
                    "generated_answer": generated_answer,
                    "score_breakdown": {
                        "coverage_score": coverage_score,
                        "correctness_score": correctness_score,
                        "relevance_score": relevance_score,
                    },
                    "reasoning": reasoning,
                }
            else:
                reasoning = {
                    "question": question,
                    "answer_description": answer_description,
                    "generated_answer": generated_answer,
                    "reasoning": reasoning
                }
            
            return score, reasoning

        async def wrapped_process(item: EvalInputItem) -> tuple[float, dict]:
            """
            Process an item asynchronously and update the progress bar.
            Use the semaphore to limit the number of concurrent items.
            """
            async with self.semaphore:
              result = await process_item(item)
              # Update the progress bar
              pbar.update(1)
              return result

        try:
            # Claim a tqdm position to display the progress bar
            tqdm_position = TqdmPositionRegistry.claim()
            # Create a progress bar
            pbar = tqdm(total=len(eval_input.eval_input_items), desc="Evaluating RAG", position=tqdm_position)
            # Process items concurrently with a limit on concurrency
            results = await asyncio.gather(*[wrapped_process(item) for item in eval_input.eval_input_items])
        finally:
            pbar.close()
            TqdmPositionRegistry.release(tqdm_position)

        # Extract scores and reasonings
        sample_scores, sample_reasonings = zip(*results) if results else ([], [])

        # Compute average score
        avg_score = round(sum(sample_scores) / len(sample_scores), 2) if sample_scores else 0.0

        # Construct EvalOutputItems
        eval_output_items = [
            EvalOutputItem(id=item.id, score=score, reasoning=reasoning)
            for item, score, reasoning in zip(eval_input.eval_input_items, sample_scores, sample_reasonings)
        ]

        return EvalOutput(average_score=avg_score, eval_output_items=eval_output_items)