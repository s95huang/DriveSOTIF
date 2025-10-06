import os
import json
import random
import logging
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import base64
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from openai import OpenAI
from pydantic import BaseModel
from enum import Enum

# Enhanced Pydantic models for structured responses
class CaptionGenerationResponse(BaseModel):
    caption: str
    confidence_score: int  # 1-5 scale
    sotif_relevance: str
    key_elements: List[str]

class CaptionValidationResponse(BaseModel):
    is_valid: bool
    explanation: str
    accuracy_score: Optional[int] = None  # 1-5 scale
    issues_found: Optional[List[str]] = None
    suggested_improvements: Optional[str] = None

class QuestionType(str, Enum):
    CLOSE_ENDED = "close_ended"
    OPEN_ENDED = "open_ended"

class SOTIFRiskType(str, Enum):
    PERCEPTION = "perception"
    PREDICTION = "prediction"
    PLANNING = "planning"
    CONTROL = "control"
    ENVIRONMENTAL = "environmental"
    INFRASTRUCTURE = "infrastructure"
    MULTI_AGENT = "multi_agent"

class ClassifiedQuestion(BaseModel):
    question: str
    type: QuestionType
    expected_answer_type: str  # "yes_no", "count", "multiple_choice", "analysis", "explanation", "recommendation"
    difficulty: int  # 1-5 scale
    sotif_relevance: str
    risk_categories: Optional[List[SOTIFRiskType]] = None

class QuestionGenerationResponse(BaseModel):
    q1: str
    q1_difficulty: str  # "easy", "medium", "hard"
    q1_type: str  # "close_ended", "open_ended"
    q1_expected_answer_type: str  # "yes_no", "count", "identification", "analysis", "recommendation"
    q1_sotif_relevance: str
    
    q2: str
    q2_difficulty: str
    q2_type: str
    q2_expected_answer_type: str
    q2_sotif_relevance: str
    
    q3: str
    q3_difficulty: str
    q3_type: str
    q3_expected_answer_type: str
    q3_sotif_relevance: str
    
    q4: str
    q4_difficulty: str
    q4_type: str
    q4_expected_answer_type: str
    q4_sotif_relevance: str
    
    q5: str
    q5_difficulty: str
    q5_type: str
    q5_expected_answer_type: str
    q5_sotif_relevance: str

class QuestionValidationResponse(BaseModel):
    is_valid: bool
    explanation: str
    answerable_from_image: bool
    sotif_relevance_score: int  # 1-5 scale
    type_classification_correct: bool
    issues_found: Optional[List[str]] = None

class RiskAnalysis(BaseModel):
    risk_type: SOTIFRiskType
    description: str
    severity_level: int  # 1-5 scale
    likelihood: int  # 1-5 scale

class AnswerSetGenerationResponse(BaseModel):
    q1_answer: str
    q2_answer: str
    q3_answer: str
    q4_answer: str
    q5_answer: str

class AnswerGenerationResponse(BaseModel):
    answer: str
    confidence_score: int  # 1-5 scale
    answer_type: str  # "factual", "analytical", "recommendation"
    reasoning: str
    risk_analysis: Optional[List[RiskAnalysis]] = None  # For SOTIF risk questions

class AnswerValidationResponse(BaseModel):
    is_valid: bool
    explanation: str
    factually_correct: bool
    answers_question: bool
    completeness_score: int  # 1-5 scale
    sotif_depth_score: int  # 1-5 scale for SOTIF-related answers
    suggested_answer: Optional[str] = None
    issues_found: Optional[List[str]] = None

class FinalValidationResponse(BaseModel):
    overall_valid: bool
    caption_score: int
    question_score: int
    answer_score: int
    sotif_alignment_score: int
    final_recommendation: str
    improvement_suggestions: Optional[List[str]] = None

class SingleQuestionResponse(BaseModel):
    question: str

class SingleQuestionWithMetadataResponse(BaseModel):
    question: str
    difficulty: str  # "easy", "medium", "hard"
    type: str  # "close_ended", "open_ended"
    expected_answer_type: str  # "yes_no", "count", "identification", "analysis", "recommendation"
    sotif_relevance: str

class SingleAnswerResponse(BaseModel):
    answer: str

def encode_image_to_base64(image_path: str) -> str:
    """Encode a local image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise Exception(f"Error encoding image: {str(e)}")

def query_openai_structured(
    image_path: str,
    prompt: str,
    response_model: BaseModel,
    model: str = "gpt-5",
    api_key: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    generation_stats: Optional[Dict] = None
) -> Tuple[bool, Any, Optional[str]]:
    """Query OpenAI with structured response using Pydantic models"""
    
    # Track API call
    if generation_stats:
        generation_stats['total_api_calls'] += 1
        generation_stats['calls_by_type'][response_model.__name__] += 1
    
    try:
        client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
        # Encode the image
        base64_image = encode_image_to_base64(image_path)
        
        # Determine image format
        image_format = image_path.lower().split('.')[-1]
        if image_format in ['jpg', 'jpeg']:
            mime_type = 'image/jpeg'
        elif image_format == 'png':
            mime_type = 'image/png'
        elif image_format == 'gif':
            mime_type = 'image/gif'
        elif image_format == 'webp':
            mime_type = 'image/webp'
        else:
            mime_type = 'image/jpeg'
        
        # Create structured response
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format=response_model
        )
        
        if generation_stats:
            generation_stats['successful_api_calls'] += 1
        
        if logger:
            logger.info(f"API Success - {response_model.__name__}")
        
        return True, response.choices[0].message.parsed, None
        
    except Exception as e:
        if generation_stats:
            generation_stats['failed_api_calls'] += 1
        
        error_msg = str(e)
        if logger:
            logger.error(f"API Failed - {response_model.__name__}: {error_msg}")
        
        return False, None, error_msg

def setup_logging(log_dir="generation_logs"):
    """Setup comprehensive logging for dataset generation"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"generation_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Dataset generation session started - Log file: {log_file}")
    return logger

class SOTIFDatasetGenerator:
    
    def __init__(self, api_key=None, model="gpt-5", random_seed=42, log_dir="generation_logs", max_regen_attempts=2):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.max_regen_attempts = max_regen_attempts
        self.logger = setup_logging(log_dir)
        
        # Enhanced statistics with detailed attempt tracking
        self.generation_stats = {
            'total_api_calls': 0,
            'successful_api_calls': 0,
            'failed_api_calls': 0,
            'calls_by_type': {
                'CaptionGenerationResponse': 0,
                'CaptionValidationResponse': 0,
                'QuestionGenerationResponse': 0,
                'QuestionValidationResponse': 0,
                'AnswerSetGenerationResponse': 0,
                'AnswerValidationResponse': 0,
                'FinalValidationResponse': 0,
                'SingleQuestionResponse': 0,
                'SingleQuestionWithMetadataResponse': 0,
                'SingleAnswerResponse': 0
            },
            'pipeline_stats': {
                'total_images_processed': 0,
                'successful_generations': 0,
                'failed_at_caption': 0,
                'failed_at_question': 0,
                'failed_at_answer': 0,
                'failed_at_final_validation': 0
            },
            'regeneration_stats': {
                'caption_regenerations': 0,
                'question_regenerations': 0,
                'answer_regenerations': 0,
                'caption_regen_successes': 0,
                'question_regen_successes': 0,
                'answer_regen_successes': 0
            },
            'failure_rates': {
                'initial_caption_failures': 0,
                'initial_question_failures': 0,
                'initial_answer_failures': 0,
                'final_caption_failures': 0,
                'final_question_failures': 0,
                'final_answer_failures': 0
            },
            # NEW: Detailed attempt tracking
            'attempt_success_tracking': {
                'caption_success_by_attempt': {1: 0, 2: 0, 3: 0 , 4: 0, 5: 0},
                'question_success_by_attempt': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                'answer_success_by_attempt': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                'total_caption_attempts': 0,
                'total_question_attempts': 0,
                'total_answer_attempts': 0
            }
        }
        
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
            self.logger.info(f"Random seed set to: {random_seed}")
        
        self.logger.info(f"Generator initialized with model: {model}")
        self.logger.info(f"Max regeneration attempts: {max_regen_attempts}")

    def step1_generate_caption(self, image_path: str, attempt: int = 1) -> Tuple[bool, Optional[str], Optional[str]]:
        """Step 1: Generate SOTIF-relevant caption for the image with regeneration support"""
        
        attempt_text = f" (Attempt {attempt})" if attempt > 1 else ""
        # self.logger.info(f"Step 1: Generating caption for {os.path.basename(image_path)}{attempt_text}")
        
        prompt = """
        We are studying Perception problems of Safety of the Intended Functionality (SOTIF) Problems in Long-tail Traffic Scenarios.
        Generate high quality caption for the image uploaded. Be objective, natural, and human-like.
        
        The caption should focus on SOTIF-relevant elements including:
        - Traffic scenario description
        - Environmental conditions (weather, lighting, visibility)
        - Road users and their behaviors
        - Infrastructure elements
        - Any unusual or edge-case elements
        
        Provide:
        - A comprehensive caption (2-4 sentences)
        - Confidence score (1-5, where 5 is most confident)
        - Brief explanation of SOTIF relevance
        - Key elements list (important objects/conditions identified)
        """
        
        success, response, error = query_openai_structured(
            image_path=image_path,
            prompt=prompt,
            response_model=CaptionGenerationResponse,
            model=self.model,
            api_key=self.api_key,
            logger=self.logger,
            generation_stats=self.generation_stats
        )
        
        if success and response:
            self.logger.info(f"  ✅ Caption generated{attempt_text}: {response.caption[:50]}...")
            return True, response.caption, None
        else:
            self.logger.error(f"  ❌ Caption generation failed{attempt_text}: {error}")
            return False, None, error

    def step2_validate_caption(self, image_path: str, caption: str, attempt: int = 1) -> Tuple[bool, Optional[str]]:
        """Step 2: Validate caption accuracy and SOTIF relevance with regeneration support"""
        
        attempt_text = f" (Attempt {attempt})" if attempt > 1 else ""
        # self.logger.info(f"Step 2: Validating caption{attempt_text}")
        
        prompt = f"""
        We are building a dataset for Safety of the Intended Functionality (SOTIF) research in autonomous vehicles.
        
        Analyze this caption for the given traffic scene image:
        Caption: "{caption}"
        
        Evaluate:
        1. Factual accuracy - Does the caption correctly describe what you see?
        2. Completeness - Are important safety-relevant elements mentioned?
        3. SOTIF relevance - Does it capture elements relevant for AV safety analysis?
        4. Clarity and precision of description
        
        Rate accuracy (1-5 scale) and identify any issues.
        If issues exist, suggest specific improvements.
        """
        
        success, response, error = query_openai_structured(
            image_path=image_path,
            prompt=prompt,
            response_model=CaptionValidationResponse,
            model=self.model,
            api_key=self.api_key,
            logger=self.logger,
            generation_stats=self.generation_stats
        )
        
        if success and response:
            if response.is_valid:
                self.logger.info(f"  ✅ Caption validated successfully{attempt_text}")
                return True, None
            else:
                self.logger.warning(f"  ⚠️ Caption validation failed{attempt_text}: {response.explanation}")
                return False, response.explanation
        else:
            self.logger.error(f"  ❌ Caption validation error{attempt_text}: {error}")
            return False, error

    def step3_generate_questions(self, image_path: str, caption: str, attempt: int = 1) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Step 3: Generate questions with LLM-generated metadata using the user's proven prompt format"""
        
        attempt_text = f" (Attempt {attempt})" if attempt > 1 else ""
        # self.logger.info(f"Step 3: Generating questions with metadata{attempt_text}")
        
        prompt = f"""
        We are studying Perception problems of Safety of the Intended Functionality (SOTIF) Problems in Long-tail Traffic Scenarios. 
        Generate 5 SOTIF related questions that can be concisely and easily answered based on the image uploaded. 
        
        For each question (q1-q5), provide:
        1. The question text
        2. Difficulty level: "easy", "medium", or "hard" (generate 2 easy, 2 medium, 1 hard)
        3. Question type: "close_ended" or "open_ended" 
        4. Expected answer type: "yes_no", "count", "identification", "analysis", "recommendation"
        5. SOTIF relevance: Brief explanation of how this relates to SOTIF research
        
        Be objective, natural, and human-like. Don't generate lengthy questions. 
        
        Half of the questions should be closed-ended questions like:
        - Existence: Does this object exist? 
        - Type: What is the object at the bottom left? 
        - Count: How many cars?
        - Key object: Is this object a key object?
        
        The rest should be open-ended questions focusing on:
        - SOTIF risk identification
        - Perception challenges for autonomous vehicles
        - Safety recommendations
        
        Image Caption: "{caption}"
        
        The image categories include:
        - Environment subset: scenarios that degrade perception ability (rain, snow, particulate, illumination)
        - Object subset: scenarios that degrade cognition ability (common road users with unusual appearances, temporary obstacles, disturbing objects)
        
        Provide all question details including metadata for proper classification and analysis.
        """
        
        success, response, error = query_openai_structured(
            image_path=image_path,
            prompt=prompt,
            response_model=QuestionGenerationResponse,
            model=self.model,
            api_key=self.api_key,
            logger=self.logger,
            generation_stats=self.generation_stats
        )
        
        if success and response:
            # self.logger.info(f"  ✅ Generated 5 questions with LLM metadata{attempt_text}")
            
            # Convert to dictionary format for easier handling
            questions_dict = {
                "q1": response.q1,
                "q2": response.q2,
                "q3": response.q3,
                "q4": response.q4,
                "q5": response.q5
            }
            
            # Extract LLM-generated metadata
            metadata = {
                "q1": {
                    "difficulty": response.q1_difficulty,
                    "type": response.q1_type,
                    "expected_answer_type": response.q1_expected_answer_type,
                    "sotif_relevance": response.q1_sotif_relevance
                },
                "q2": {
                    "difficulty": response.q2_difficulty,
                    "type": response.q2_type,
                    "expected_answer_type": response.q2_expected_answer_type,
                    "sotif_relevance": response.q2_sotif_relevance
                },
                "q3": {
                    "difficulty": response.q3_difficulty,
                    "type": response.q3_type,
                    "expected_answer_type": response.q3_expected_answer_type,
                    "sotif_relevance": response.q3_sotif_relevance
                },
                "q4": {
                    "difficulty": response.q4_difficulty,
                    "type": response.q4_type,
                    "expected_answer_type": response.q4_expected_answer_type,
                    "sotif_relevance": response.q4_sotif_relevance
                },
                "q5": {
                    "difficulty": response.q5_difficulty,
                    "type": response.q5_type,
                    "expected_answer_type": response.q5_expected_answer_type,
                    "sotif_relevance": response.q5_sotif_relevance
                }
            }
            
            return True, {"questions": questions_dict, "metadata": metadata}, None
        else:
            self.logger.error(f"  ❌ Question generation failed{attempt_text}: {error}")
            return False, None, error

    def step4_validate_questions(self, image_path: str, caption: str, questions_data: Dict, attempt: int = 1) -> Tuple[bool, Optional[str], Optional[List[str]]]:
        """Step 4: Validate generated questions individually and return failed question keys"""
        
        attempt_text = f" (Attempt {attempt})" if attempt > 1 else ""
        # self.logger.info(f"Step 4: Validating questions individually{attempt_text}")
        
        questions = questions_data["questions"]
        metadata = questions_data["metadata"]
        failed_questions = []
        
        # Validate ALL questions individually
        for q_key, question in questions.items():
            q_metadata = metadata.get(q_key, {})
            
            prompt = f"""
            Evaluate this classified question for a SOTIF traffic scenario dataset:
            
            Image Caption: "{caption}"
            Question: "{question}"
            Claimed Type: {q_metadata.get('type', 'unknown')}
            Expected Answer Type: {q_metadata.get('expected_answer_type', 'unknown')}
            SOTIF Relevance: {q_metadata.get('sotif_relevance', 'Not provided')}
            
            Check:
            1. Can this question be answered from the image? 
            2. Is it relevant to SOTIF research objectives?
            3. Is the type classification (close_ended/open_ended) correct?
            4. Is it clearly worded and unambiguous?
            5. Does it contribute meaningful value to autonomous vehicle safety research?
            
            Rate SOTIF relevance (1-5 scale) and verify type classification.
            """
            
            success, response, error = query_openai_structured(
                image_path=image_path,
                prompt=prompt,
                response_model=QuestionValidationResponse,
                model=self.model,
                api_key=self.api_key,
                logger=self.logger,
                generation_stats=self.generation_stats
            )
            
            if success and response:
                if not response.is_valid or not response.type_classification_correct:
                    self.logger.warning(f"  ⚠️ Question '{q_key}' validation failed{attempt_text}: {response.explanation}")
                    failed_questions.append(q_key)
                else:
                    self.logger.debug(f"  ✅ Question '{q_key}' validated successfully{attempt_text}")
            else:
                self.logger.error(f"  ❌ Question '{q_key}' validation error{attempt_text}: {error}")
                failed_questions.append(q_key)
        
        if failed_questions:
            failed_keys_str = ", ".join([f"'{key}'" for key in failed_questions])
            self.logger.warning(f"  ⚠️ Questions failed validation{attempt_text}: {failed_questions}")
            return False, f"Questions [{failed_keys_str}] validation failed", failed_questions
        else:
            self.logger.info(f"  ✅ All questions validated successfully{attempt_text}")
            return True, None, None

    def regenerate_individual_questions(self, image_path: str, caption: str, questions_data: Dict, failed_question_keys: List[str], attempt: int = 1) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Regenerate only the failed questions with LLM-generated metadata while keeping successful ones"""
        
        attempt_text = f" (Attempt {attempt})" if attempt > 1 else ""
        self.logger.info(f"Regenerating individual questions with metadata{attempt_text}: {failed_question_keys}")
        
        questions = questions_data["questions"].copy()
        metadata = questions_data["metadata"].copy()
        
        for q_key in failed_question_keys:
            q_metadata = metadata.get(q_key, {})
            
            prompt = f"""
            We are studying Perception problems of Safety of the Intended Functionality (SOTIF) Problems in Long-tail Traffic Scenarios.
            Generate ONE specific SOTIF-related question that can be concisely and easily answered based on the image uploaded.
            
            Image Caption: "{caption}"
            Question Key: {q_key}
            Target Difficulty: {q_metadata.get('difficulty', 'medium')} (but adjust if needed based on image content)
            Target Type: {q_metadata.get('type', 'unknown')} (but adjust if needed)
            
            Provide:
            1. The question text (be objective, natural, and human-like)
            2. Actual difficulty level: "easy", "medium", or "hard"
            3. Question type: "close_ended" or "open_ended"
            4. Expected answer type: "yes_no", "count", "identification", "analysis", "recommendation"
            5. SOTIF relevance: Brief explanation of how this relates to SOTIF research
            
            Don't generate lengthy questions. Ensure it's answerable from the image.
            
            For closed-ended questions use formats like:
            - Existence: Does this object exist? 
            - Type: What is the object at the bottom left? 
            - Count: How many cars?
            
            For open-ended questions focus on:
            - SOTIF risk identification
            - Perception challenges for autonomous vehicles
            - Safety recommendations
            
            Context: The image categories include environment subset (scenarios that degrade perception ability like rain, snow, particulate, illumination) and object subset (scenarios that degrade cognition ability with common road users in unusual appearances, temporary obstacles, disturbing objects).
            """
            
            # Use the new SingleQuestionWithMetadataResponse model
            success, response, error = query_openai_structured(
                image_path=image_path,
                prompt=prompt,
                response_model=SingleQuestionWithMetadataResponse,
                model=self.model,
                api_key=self.api_key,
                logger=self.logger,
                generation_stats=self.generation_stats
            )
            
            if success and response:
                # Update the failed question and its metadata
                questions[q_key] = response.question
                metadata[q_key] = {
                    "difficulty": response.difficulty,
                    "type": response.type,
                    "expected_answer_type": response.expected_answer_type,
                    "sotif_relevance": response.sotif_relevance
                }
                # self.logger.info(f"  ✅ Successfully regenerated question with metadata for {q_key}{attempt_text}")
            else:
                self.logger.error(f"  ❌ Failed to regenerate question for {q_key}{attempt_text}: {error}")
                return False, None, f"Failed to regenerate question for {q_key}: {error}"
        
        return True, {"questions": questions, "metadata": metadata}, None

    def regenerate_questions_with_attempts(self, image_path: str, caption: str) -> Tuple[bool, Optional[Dict], Optional[str], int]:
        """Generate and validate questions with individual question regeneration and attempt tracking"""
        
        for overall_attempt in range(1, self.max_regen_attempts + 2):
            self.generation_stats['attempt_success_tracking']['total_question_attempts'] += 1
            attempt_text = f" (Attempt {overall_attempt})" if overall_attempt > 1 else ""
            
            # Step 3: Generate initial questions (or regenerate all if first attempt)
            if overall_attempt == 1:
                success, questions_data, error = self.step3_generate_questions(image_path, caption, overall_attempt)
                if not success:
                    if overall_attempt == 1:
                        self.generation_stats['failure_rates']['initial_question_failures'] += 1
                    continue
            
            # Step 4: Validate all questions
            success, error_msg, failed_question_keys = self.step4_validate_questions(image_path, caption, questions_data, overall_attempt)
            
            if success:
                # All questions validated successfully
                if overall_attempt > 1:
                    self.generation_stats['regeneration_stats']['question_regen_successes'] += 1
                    self.logger.info(f"  ✅  Question regeneration succeeded on attempt {overall_attempt}")
                
                # Track success by attempt number
                if overall_attempt <= 3:
                    self.generation_stats['attempt_success_tracking']['question_success_by_attempt'][overall_attempt] += 1
                
                return True, questions_data, None, overall_attempt
            
            # Some questions failed validation
            if failed_question_keys:
                if overall_attempt <= self.max_regen_attempts + 1:
                    self.generation_stats['regeneration_stats']['question_regenerations'] += 1
                    self.logger.warning(f"  🔄 Regenerating individual questions{attempt_text}: {failed_question_keys}")
                    
                    # Regenerate only the failed questions
                    success, questions_data, error = self.regenerate_individual_questions(
                        image_path, caption, questions_data, failed_question_keys, overall_attempt
                    )
                    if not success:
                        if overall_attempt > self.max_regen_attempts + 1:
                            self.generation_stats['failure_rates']['final_question_failures'] += 1
                        continue
                else:
                    self.generation_stats['failure_rates']['final_question_failures'] += 1
                    break
            else:
                # Other validation error, regenerate all
                if overall_attempt <= self.max_regen_attempts + 1:
                    self.generation_stats['regeneration_stats']['question_regenerations'] += 1
                    self.logger.warning(f"  🔄 Regenerating all questions{attempt_text} (validation error)")
                    success, questions_data, error = self.step3_generate_questions(image_path, caption, overall_attempt + 1)
                    if not success:
                        continue
                else:
                    self.generation_stats['failure_rates']['final_question_failures'] += 1
                    break
        
        return False, None, f"Question validation failed after {self.max_regen_attempts} regeneration attempts", self.max_regen_attempts + 1

    def step5_generate_answers(self, image_path: str, caption: str, questions_data: Dict, attempt: int = 1) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Step 5: Generate answers using the user's proven answer prompt"""
        
        attempt_text = f" (Attempt {attempt})" if attempt > 1 else ""
        # self.logger.info(f"Step 5: Generating answers{attempt_text}")
        
        questions = questions_data["questions"]
        metadata = questions_data["metadata"]
        
        # Create a single prompt for all questions to maintain consistency
        questions_text = "\n".join([f"{key}: {question}" for key, question in questions.items()])
        
        prompt = f"""
        We are studying Perception problems of Safety of the Intended Functionality (SOTIF) Problems in Long-tail Traffic Scenarios.
        Answer the questions concisely and accurately based on the image uploaded.
        For yes/no questions, just answer yes or no. For multiple-choice questions, answer with the most likely answer.
        For each question, generate objective, natural, and human-like answer.
        Ensure a balance of answering the questions at entry-level, intermediate, and expert level.
        
        Generate questions and answer pairs focusing on identifying perception-related SOTIF risk and provide reasoning and recommended actions. Generate natural, objective questions and provide the answer directly, concisely and accurately based on the image uploaded.
        
        Image Caption: "{caption}"
        
        Questions to answer:
        {questions_text}
        
        Context: The image categories include environment subset (scenarios that degrade perception ability like rain, snow, particulate, illumination) and object subset (scenarios that degrade cognition ability with common road users in unusual appearances, temporary obstacles, disturbing objects).
        
        Provide answers for all 5 questions (q1-q5).
        """
        
        success, response, error = query_openai_structured(
            image_path=image_path,
            prompt=prompt,
            response_model=AnswerSetGenerationResponse,
            model=self.model,
            api_key=self.api_key,
            logger=self.logger,
            generation_stats=self.generation_stats
        )
        
        if success and response:
            # Convert response to the expected format
            answers = {
                "q1": {
                    "answer": response.q1_answer,
                    "confidence_score": 4,  # Default values since we're generating as a set
                    "answer_type": "factual",
                    "reasoning": "Generated as part of answer set"
                },
                "q2": {
                    "answer": response.q2_answer,
                    "confidence_score": 4,
                    "answer_type": "factual", 
                    "reasoning": "Generated as part of answer set"
                },
                "q3": {
                    "answer": response.q3_answer,
                    "confidence_score": 4,
                    "answer_type": "analytical",
                    "reasoning": "Generated as part of answer set"
                },
                "q4": {
                    "answer": response.q4_answer,
                    "confidence_score": 4,
                    "answer_type": "analytical",
                    "reasoning": "Generated as part of answer set"
                },
                "q5": {
                    "answer": response.q5_answer,
                    "confidence_score": 4,
                    "answer_type": "recommendation",
                    "reasoning": "Generated as part of answer set"
                }
            }
            
            # self.logger.info(f"  ✅ Generated 5 answers{attempt_text}")
            return True, answers, None
        else:
            self.logger.error(f"  ❌ Answer generation failed{attempt_text}: {error}")
            return False, None, f"Answer generation failed: {error}"

    def step6_validate_answers(self, image_path: str, caption: str, questions_data: Dict, answers: Dict, attempt: int = 1) -> Tuple[bool, Optional[str], Optional[List[str]]]:
        """Step 6: Validate ALL answers individually and return failed answer keys"""
        
        attempt_text = f" (Attempt {attempt})" if attempt > 1 else ""
        # self.logger.info(f"Step 6: Validating all 5 answers individually{attempt_text}")
        
        questions = questions_data["questions"]
        metadata = questions_data["metadata"]
        failed_answers = []
        
        # Validate ALL Q&A pairs individually
        for q_key in questions.keys():
            question = questions[q_key]
            answer_data = answers.get(q_key, {})
            answer = answer_data.get("answer", "")
            q_type = metadata.get(q_key, {}).get('type', 'unknown')
            
            prompt = f"""
            Evaluate this Q&A pair for a SOTIF traffic scenario dataset:
            
            Image Caption: "{caption}"
            Question: "{question}"
            Question Type: {q_type}
            Answer: "{answer}"
            
            Evaluate:
            1. Factual correctness based on the image
            2. Does the answer directly address the question?
            3. Completeness and detail level
            4. Accuracy for SOTIF research purposes
            5. For SOTIF-related answers, depth of safety analysis
            
            Rate completeness (1-5 scale) and SOTIF depth (1-5 scale).
            Provide specific feedback if inadequate.
            If the answer is poor, suggest a better answer.
            """
            
            success, response, error = query_openai_structured(
                image_path=image_path,
                prompt=prompt,
                response_model=AnswerValidationResponse,
                model=self.model,
                api_key=self.api_key,
                logger=self.logger,
                generation_stats=self.generation_stats
            )
            
            if success and response:
                if not response.is_valid:
                    self.logger.warning(f"  ⚠️ Answer '{q_key}' validation failed{attempt_text}: {response.explanation}")
                    failed_answers.append(q_key)
                else:
                    self.logger.debug(f"  ✅ Answer '{q_key}' validated successfully{attempt_text}")
            else:
                self.logger.error(f"  ❌ Answer '{q_key}' validation error{attempt_text}: {error}")
                failed_answers.append(q_key)
        
        if failed_answers:
            failed_keys_str = ", ".join([f"'{key}'" for key in failed_answers])
            self.logger.warning(f"  ⚠️ Answers failed validation{attempt_text}: {failed_answers}")
            return False, f"Answers [{failed_keys_str}] validation failed", failed_answers
        else:
            # self.logger.info(f"  ✅ All answers validated successfully{attempt_text}")
            return True, None, None

    def regenerate_individual_answers(self, image_path: str, caption: str, questions_data: Dict, current_answers: Dict, failed_answer_keys: List[str], attempt: int = 1) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Regenerate only the failed answers while keeping successful ones"""
        
        attempt_text = f" (Attempt {attempt})" if attempt > 1 else ""
        self.logger.info(f"Regenerating individual answers{attempt_text}: {failed_answer_keys}")
        
        questions = questions_data["questions"]
        metadata = questions_data["metadata"]
        answers = current_answers.copy()
        
        for q_key in failed_answer_keys:
            question = questions[q_key]
            q_metadata = metadata.get(q_key, {})
            
            prompt = f"""
            We are studying Perception problems of Safety of the Intended Functionality (SOTIF) Problems in Long-tail Traffic Scenarios.
            Answer this specific question concisely and accurately based on the image uploaded.
            
            Image Caption: "{caption}"
            Question: "{question}"
            Question Type: {q_metadata.get('type', 'unknown')}
            Question Difficulty: {q_metadata.get('difficulty', 'medium')}
            
            Instructions:
            - For yes/no questions, just answer yes or no
            - For counting questions, provide the specific count
            - For identification questions, provide clear classification
            - For analysis questions, provide detailed SOTIF risk analysis
            - For recommendation questions, provide specific safety recommendations
            
            Generate an objective, natural, and human-like answer that directly addresses this question.
            """
            
            # Use the global SingleAnswerResponse model
            
            success, response, error = query_openai_structured(
                image_path=image_path,
                prompt=prompt,
                response_model=SingleAnswerResponse,
                model=self.model,
                api_key=self.api_key,
                logger=self.logger,
                generation_stats=self.generation_stats
            )
            
            if success and response:
                # Update the failed answer
                answers[q_key] = {
                    "answer": response.answer,
                    "confidence_score": 4,
                    "answer_type": q_metadata.get('expected_answer_type', 'factual'),
                    "reasoning": f"Regenerated answer for {q_key}"
                }
                self.logger.info(f"  ✅ Successfully regenerated answer for {q_key}{attempt_text}")
            else:
                self.logger.error(f"  ❌ Failed to regenerate answer for {q_key}{attempt_text}: {error}")
                return False, None, f"Failed to regenerate answer for {q_key}: {error}"
        
        return True, answers, None

    def regenerate_answers_with_attempts(self, image_path: str, caption: str, questions_data: Dict) -> Tuple[bool, Optional[Dict], Optional[str], int]:
        """Generate and validate answers with individual answer regeneration and attempt tracking"""
        
        for overall_attempt in range(1, self.max_regen_attempts + 2):
            self.generation_stats['attempt_success_tracking']['total_answer_attempts'] += 1
            attempt_text = f" (Attempt {overall_attempt})" if overall_attempt > 1 else ""
            
            # Step 5: Generate initial answers (or regenerate all if first attempt)
            if overall_attempt == 1:
                success, answers, error = self.step5_generate_answers(image_path, caption, questions_data, overall_attempt)
                if not success:
                    if overall_attempt == 1:
                        self.generation_stats['failure_rates']['initial_answer_failures'] += 1
                    continue
            
            # Step 6: Validate all answers
            success, error_msg, failed_answer_keys = self.step6_validate_answers(image_path, caption, questions_data, answers, overall_attempt)
            
            if success:
                # All answers validated successfully
                if overall_attempt > 1:
                    self.generation_stats['regeneration_stats']['answer_regen_successes'] += 1
                    self.logger.info(f"  ✅  Answer regeneration succeeded on attempt {overall_attempt}")
                
                # Track success by attempt number
                if overall_attempt <= 3:
                    self.generation_stats['attempt_success_tracking']['answer_success_by_attempt'][overall_attempt] += 1
                
                return True, answers, None, overall_attempt
            
            # Some answers failed validation
            if failed_answer_keys:
                if overall_attempt <= self.max_regen_attempts + 1:
                    self.generation_stats['regeneration_stats']['answer_regenerations'] += 1
                    self.logger.warning(f"  🔄 Regenerating individual answers{attempt_text}: {failed_answer_keys}")
                    
                    # Regenerate only the failed answers
                    success, answers, error = self.regenerate_individual_answers(
                        image_path, caption, questions_data, answers, failed_answer_keys, overall_attempt
                    )
                    if not success:
                        if overall_attempt > self.max_regen_attempts + 1:
                            self.generation_stats['failure_rates']['final_answer_failures'] += 1
                        continue
                else:
                    self.generation_stats['failure_rates']['final_answer_failures'] += 1
                    break
            else:
                # Other validation error, regenerate all
                if overall_attempt <= self.max_regen_attempts + 1:
                    self.generation_stats['regeneration_stats']['answer_regenerations'] += 1
                    self.logger.warning(f"  🔄 Regenerating all answers{attempt_text} (validation error)")
                    success, answers, error = self.step5_generate_answers(image_path, caption, questions_data, overall_attempt + 1)
                    if not success:
                        continue
                else:
                    self.generation_stats['failure_rates']['final_answer_failures'] += 1
                    break
        
        return False, None, f"Answer validation failed after {self.max_regen_attempts} regeneration attempts", self.max_regen_attempts + 1

    def step7_final_validation(self, image_path: str, caption: str, questions_data: Dict, answers: Dict) -> Tuple[bool, Optional[str]]:
        """Step 7: Final comprehensive validation of the complete data item"""
        
        # self.logger.info(f"Step 7: Final validation")
        
        questions = questions_data["questions"]
        # Sample one Q&A pair for final validation
        sample_key = random.choice(['q1', 'q2', 'q3', 'q4', 'q5'])
        sample_question = questions[sample_key]
        sample_answer = answers[sample_key]["answer"]
        
        prompt = f"""
        Perform final quality assessment of this complete SOTIF dataset item:
        
        Image Caption: "{caption}"
        Sample Question: "{sample_question}"
        Sample Answer: "{sample_answer}"
        Total Questions: 5 (q1-q5 format)
        
        Evaluate overall quality:
        1. Caption score (1-5): Accuracy and SOTIF relevance
        2. Question score (1-5): Quality, variety, difficulty balance (2 easy, 2 medium, 1 hard)
        3. Answer score (1-5): Accuracy, conciseness, and SOTIF depth
        4. SOTIF alignment score (1-5): Relevance to perception problems in long-tail traffic scenarios
        
        Consider the format requirements:
        - Questions should be concise and easily answerable
        - Mix of closed-ended (existence, type, count) and open-ended questions
        - Answers should be objective, natural, and human-like
        - Focus on perception-related SOTIF risks
        
        Provide final recommendation: ACCEPT or REJECT
        If issues exist, suggest specific improvements.
        """
        
        success, response, error = query_openai_structured(
            image_path=image_path,
            prompt=prompt,
            response_model=FinalValidationResponse,
            model=self.model,
            api_key=self.api_key,
            logger=self.logger,
            generation_stats=self.generation_stats
        )
        
        if success and response:
            if response.overall_valid:
                self.logger.info(f"  ✅ Final validation passed - {response.final_recommendation}")
                return True, None
            else:
                self.logger.warning(f"  ⚠️ Final validation failed: {response.final_recommendation}")
                self.generation_stats['pipeline_stats']['failed_at_final_validation'] += 1
                self.logger.info("Suggestions for improvement:" + response.improvement_suggestions)
                return False, response.final_recommendation
        else:
            self.logger.error(f"  ❌ Final validation error: {error}")
            return False, error

    def regenerate_caption_with_attempts(self, image_path: str) -> Tuple[bool, Optional[str], Optional[str], int]:
        """Generate and validate caption with attempt tracking"""
        
        for attempt in range(1, self.max_regen_attempts + 2):
            self.generation_stats['attempt_success_tracking']['total_caption_attempts'] += 1
            
            # Generate
            success, caption, error = self.step1_generate_caption(image_path, attempt=attempt)
            if not success:
                if attempt == 1:
                    self.generation_stats['failure_rates']['initial_caption_failures'] += 1
                if attempt > self.max_regen_attempts + 1:
                    self.generation_stats['failure_rates']['final_caption_failures'] += 1
                    return False, None, f"Caption generation failed after {self.max_regen_attempts} regeneration attempts: {error}", attempt
                continue
            
            # Validate
            success, error = self.step2_validate_caption(image_path, caption, attempt=attempt)
            if success:
                if attempt > 1:
                    self.generation_stats['regeneration_stats']['caption_regen_successes'] += 1
                    self.logger.info(f"  ✅ Caption regeneration succeeded on attempt {attempt}")
                
                # Track success by attempt number
                if attempt <= 3:
                    self.generation_stats['attempt_success_tracking']['caption_success_by_attempt'][attempt] += 1
                
                return True, caption, None, attempt
            else:
                if attempt == 1:
                    self.generation_stats['failure_rates']['initial_caption_failures'] += 1
                if attempt <= self.max_regen_attempts + 1:
                    self.generation_stats['regeneration_stats']['caption_regenerations'] += 1
                    self.logger.warning(f"  🔄 Regenerating caption (attempt {attempt + 1}/{self.max_regen_attempts + 1})")
                else:
                    self.generation_stats['failure_rates']['final_caption_failures'] += 1
        
        return False, None, f"Caption validation failed after {self.max_regen_attempts} regeneration attempts", self.max_regen_attempts + 1

    def process_single_image(self, image_path: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Process a single image through the complete pipeline with regeneration and attempt tracking"""
        
        image_name = os.path.basename(image_path)

        
        self.generation_stats['pipeline_stats']['total_images_processed'] += 1
        
        try:
            # Step 1 & 2: Generate and Validate Caption with attempt tracking
            success, caption, error, caption_attempts = self.regenerate_caption_with_attempts(image_path)
            if not success:
                self.generation_stats['pipeline_stats']['failed_at_caption'] += 1
                return False, None, error
            
            # Step 3 & 4: Generate and Validate Questions with individual regeneration and attempt tracking
            success, questions_data, error, question_attempts = self.regenerate_questions_with_attempts(image_path, caption)
            if not success:
                self.generation_stats['pipeline_stats']['failed_at_question'] += 1
                return False, None, error
            
            # Step 5 & 6: Generate and Validate Answers with individual regeneration and attempt tracking
            success, answers, error, answer_attempts = self.regenerate_answers_with_attempts(image_path, caption, questions_data)
            if not success:
                self.generation_stats['pipeline_stats']['failed_at_answer'] += 1
                return False, None, error
            
            # Step 7: Final Validation (no regeneration, as it's comprehensive)
            success, error = self.step7_final_validation(image_path, caption, questions_data, answers)
            if not success:
                # self.generation_stats['pipeline_stats']['failed_at_final_validation'] += 1
                return False, None, f"Final validation failed: {error}"
            
            # Success - compile results with attempt tracking
            # Convert answers back to simple format for compatibility
            simple_answers = {}
            for key, answer_data in answers.items():
                simple_answers[key] = answer_data["answer"]
            
            result = {
                'image_path': image_path,
                'image_name': image_name,
                'caption': caption,
                'questions': questions_data["questions"],
                'answers': simple_answers,
                'question_metadata': questions_data["metadata"],
                'answer_metadata': answers,
                'generation_timestamp': datetime.now().isoformat(),
                'model_used': self.model,
                'attempt_tracking': {
                    'caption_attempts': caption_attempts,
                    'question_attempts': question_attempts,
                    'answer_attempts': answer_attempts
                }
            }
            
            self.generation_stats['pipeline_stats']['successful_generations'] += 1
            self.logger.info(f"✅ Successfully generated complete dataset item for {image_name}")
            self.logger.info(f"   📊 Attempt Summary: Caption={caption_attempts}, Questions={question_attempts}, Answers={answer_attempts}")
            
            return True, result, None
            
        except Exception as e:
            error_msg = f"Unexpected error processing {image_name}: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg

    def save_dataset_item(self, result: Dict, output_dir: str):
        """Save individual dataset item to organized folder structure"""
        
        # Create directory structure
        os.makedirs(os.path.join(output_dir, 'captions'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'questions'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'answers'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
        
        base_name = os.path.splitext(result['image_name'])[0]
        
        # Save caption
        caption_data = {"caption": result['caption']}
        with open(os.path.join(output_dir, 'captions', f'{base_name}.json'), 'w') as f:
            json.dump(caption_data, f, indent=2)
        
        # Save questions
        with open(os.path.join(output_dir, 'questions', f'{base_name}.json'), 'w') as f:
            json.dump(result['questions'], f, indent=2)
        
        # Save answers
        with open(os.path.join(output_dir, 'answers', f'{base_name}.json'), 'w') as f:
            json.dump(result['answers'], f, indent=2)
        
        # Save metadata (question classifications, answer details, attempt tracking, etc.)
        metadata = {
            'question_metadata': result.get('question_metadata', {}),
            'answer_metadata': result.get('answer_metadata', {}),
            'attempt_tracking': result.get('attempt_tracking', {}),
            'generation_timestamp': result.get('generation_timestamp'),
            'model_used': result.get('model_used')
        }
        with open(os.path.join(output_dir, 'metadata', f'{base_name}.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    def calculate_failure_rates(self) -> Dict:
        """Calculate failure rates before and after regeneration"""
        
        total_processed = self.generation_stats['pipeline_stats']['total_images_processed']
        
        if total_processed == 0:
            return {
                'initial_rates': {},
                'final_rates': {},
                'improvement': {}
            }
        
        # Initial failure rates (before regeneration)
        initial_rates = {}
        final_rates = {}
        improvement = {}
        
        for component in ['caption', 'question', 'answer']:
            initial_failures = self.generation_stats['failure_rates'][f'initial_{component}_failures']
            final_failures = self.generation_stats['failure_rates'][f'final_{component}_failures']
            
            initial_rate = (initial_failures / total_processed) * 100
            final_rate = (final_failures / total_processed) * 100
            improvement_rate = initial_rate - final_rate
            
            initial_rates[component] = initial_rate
            final_rates[component] = final_rate
            improvement[component] = improvement_rate
        
        return {
            'initial_rates': initial_rates,
            'final_rates': final_rates,
            'improvement': improvement
        }

    def calculate_attempt_statistics(self) -> Dict:
        """Calculate detailed attempt statistics"""
        
        attempt_stats = self.generation_stats['attempt_success_tracking']
        
        statistics = {
            'caption_attempt_distribution': {},
            'question_attempt_distribution': {},
            'answer_attempt_distribution': {},
            'average_attempts': {},
            'success_rates_by_attempt': {}
        }
        
        for component in ['caption', 'question', 'answer']:
            total_attempts = attempt_stats[f'total_{component}_attempts']
            success_by_attempt = attempt_stats[f'{component}_success_by_attempt']
            
            if total_attempts > 0:
                # Calculate distribution percentages
                distribution = {}
                total_successes = sum(success_by_attempt.values())
                
                for attempt_num in [1, 2, 3]:
                    successes = success_by_attempt.get(attempt_num, 0)
                    percentage = (successes / total_successes * 100) if total_successes > 0 else 0
                    distribution[f'attempt_{attempt_num}'] = {
                        'count': successes,
                        'percentage': percentage
                    }
                
                statistics[f'{component}_attempt_distribution'] = distribution
                
                # Calculate average attempts (weighted average)
                weighted_sum = sum(attempt_num * count for attempt_num, count in success_by_attempt.items())
                statistics['average_attempts'][component] = weighted_sum / total_successes if total_successes > 0 else 0
                
                # Success rate by attempt
                statistics['success_rates_by_attempt'][component] = {
                    f'attempt_{attempt_num}': (count / total_attempts * 100) if total_attempts > 0 else 0
                    for attempt_num, count in success_by_attempt.items()
                }
            else:
                statistics[f'{component}_attempt_distribution'] = {}
                statistics['average_attempts'][component] = 0
                statistics['success_rates_by_attempt'][component] = {}
        
        return statistics

    def generate_dataset(self, input_image_dir: str, output_dir: str) -> List[Dict]:
        """Generate complete dataset from input images with regeneration support and attempt tracking"""
        
        self.logger.info(f"Starting dataset generation with enhanced regeneration and attempt tracking")
        self.logger.info(f"Input directory: {input_image_dir}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Max regeneration attempts: {self.max_regen_attempts}")
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp']:
            image_files.extend([f for f in os.listdir(input_image_dir) 
                              if f.lower().endswith(ext.replace('*', ''))])
        
        if not image_files:
            self.logger.error(f"No image files found in {input_image_dir}")
            return []
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        all_results = []
        successful_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(input_image_dir, image_file)

            # check to see if the caption, questions, and answers already exist
            caption_path = os.path.join(output_dir, 'captions', f"{os.path.splitext(image_file)[0]}.json")
            questions_path = os.path.join(output_dir, 'questions', f"{os.path.splitext(image_file)[0]}.json")
            answers_path = os.path.join(output_dir, 'answers', f"{os.path.splitext(image_file)[0]}.json")

            # check each file exists,  i think it is better to check all three files exist
            if os.path.exists(caption_path) and os.path.exists(questions_path) and os.path.exists(answers_path):
                # self.logger.info(f"Skipping {image_file} - already processed")
                continue


            
            self.logger.info(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
            
            # Process the image
            success, result, error = self.process_single_image(image_path)
            
            if success and result:
                # Copy image to output directory
                import shutil
                shutil.copy2(image_path, os.path.join(output_dir, 'images', image_file))
                
                # Save dataset components
                self.save_dataset_item(result, output_dir)
                
                all_results.append(result)
                successful_count += 1
                self.logger.info(f"✅ Success - Generated complete dataset item")
                
                # self.logger.info attempt summary for this image
                attempt_tracking = result.get('attempt_tracking', {})
                self.logger.info(f"   📊 Attempts: Caption={attempt_tracking.get('caption_attempts', 'N/A')}, Questions={attempt_tracking.get('question_attempts', 'N/A')}, Answers={attempt_tracking.get('answer_attempts', 'N/A')}")
                
            else:
                self.logger.info(f"❌ Failed - {error}")


        
        # Save comprehensive results with failure rate analysis and attempt statistics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"generation_results_{timestamp}.json")
        
        failure_rates = self.calculate_failure_rates()
        attempt_statistics = self.calculate_attempt_statistics()
        
        comprehensive_data = {
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'model_used': self.model,
                'max_regeneration_attempts': self.max_regen_attempts,
                'input_directory': input_image_dir,
                'output_directory': output_dir,
                'total_images_found': len(image_files),
                'successful_generations': successful_count
            },
            'generation_statistics': self.generation_stats,
            'failure_rate_analysis': failure_rates,
            'attempt_statistics': attempt_statistics,
            'detailed_results': all_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(comprehensive_data, f, indent=2)
        
        # self.logger.info final statistics
        self.log_generation_statistics()
        
        self.logger.info(f"Dataset generation completed!")
        self.logger.info(f"Successful: {successful_count}/{len(image_files)} images")
        self.logger.info(f"Results saved to: {results_file}")
        
        return all_results

    def log_generation_statistics(self):
        """self.logger.info detailed generation statistics including regeneration analysis and attempt tracking"""
        stats = self.generation_stats
        failure_rates = self.calculate_failure_rates()
        attempt_statistics = self.calculate_attempt_statistics()
        
        self.logger.info("\n" + "="*80)
        self.logger.info("ENHANCED DATASET GENERATION STATISTICS WITH ATTEMPT TRACKING")
        self.logger.info("="*80)
        
        # Pipeline statistics
        pipeline = stats['pipeline_stats']
        self.logger.info(f"\nPIPELINE RESULTS:")
        self.logger.info(f"  Total images processed: {pipeline['total_images_processed']}")
        self.logger.info(f"  Successful generations: {pipeline['successful_generations']}")
        self.logger.info(f"  Failed at caption stage: {pipeline['failed_at_caption']}")
        self.logger.info(f"  Failed at question stage: {pipeline['failed_at_question']}")
        self.logger.info(f"  Failed at answer stage: {pipeline['failed_at_answer']}")
        self.logger.info(f"  Failed at final validation: {pipeline['failed_at_final_validation']}")
        
        if pipeline['total_images_processed'] > 0:
            success_rate = (pipeline['successful_generations'] / pipeline['total_images_processed']) * 100
            self.logger.info(f"  SUCCESS RATE: {success_rate:.1f}%")
        
        # NEW: Attempt Statistics
        self.logger.info(f"\nATTEMPT STATISTICS (How many attempts needed to succeed):")
        for component in ['caption', 'question', 'answer']:
            distribution = attempt_statistics[f'{component}_attempt_distribution']
            avg_attempts = attempt_statistics['average_attempts'].get(component, 0)
            
            self.logger.info(f"  {component.title()}:")
            self.logger.info(f"    Average attempts to success: {avg_attempts:.2f}")
            for attempt_key, data in distribution.items():
                attempt_num = attempt_key.split('_')[1]
                count = data['count']
                percentage = data['percentage']
                self.logger.info(f"  ✅   Succeeded on attempt {attempt_num}: {count} images ({percentage:.1f}%)")
        
        # Regeneration statistics
        regen = stats['regeneration_stats']
        self.logger.info(f"\nREGENERATION ANALYSIS:")
        self.logger.info(f"  Caption regenerations attempted: {regen['caption_regenerations']}")
        self.logger.info(f"  Caption regeneration successes: {regen['caption_regen_successes']}")
        self.logger.info(f"  Question regenerations attempted: {regen['question_regenerations']}")
        self.logger.info(f"  Question regeneration successes: {regen['question_regen_successes']}")
        self.logger.info(f"  Answer regenerations attempted: {regen['answer_regenerations']}")
        self.logger.info(f"  Answer regeneration successes: {regen['answer_regen_successes']}")
        
        total_regen_attempts = regen['caption_regenerations'] + regen['question_regenerations'] + regen['answer_regenerations']
        total_regen_successes = regen['caption_regen_successes'] + regen['question_regen_successes'] + regen['answer_regen_successes']
        
        if total_regen_attempts > 0:
            regen_success_rate = (total_regen_successes / total_regen_attempts) * 100
            self.logger.info(f"  Overall regeneration success rate: {regen_success_rate:.1f}%")
        
        # Failure rate comparison
        self.logger.info(f"\nFAILURE RATE ANALYSIS (Before vs After Regeneration):")
        initial_rates = failure_rates['initial_rates']
        final_rates = failure_rates['final_rates']
        improvement = failure_rates['improvement']
        
        for component in ['caption', 'question', 'answer']:
            self.logger.info(f"  {component.title()}:")
            self.logger.info(f"    Initial failure rate: {initial_rates.get(component, 0):.1f}%")
            self.logger.info(f"    Final failure rate: {final_rates.get(component, 0):.1f}%")
            self.logger.info(f"    Improvement: {improvement.get(component, 0):.1f} percentage points")
        
        # API call statistics
        self.logger.info(f"\nAPI CALL STATISTICS:")
        self.logger.info(f"  Total API calls: {stats['total_api_calls']}")
        self.logger.info(f"  Successful calls: {stats['successful_api_calls']}")
        self.logger.info(f"  Failed calls: {stats['failed_api_calls']}")
        
        if stats['total_api_calls'] > 0:
            api_success_rate = (stats['successful_api_calls'] / stats['total_api_calls']) * 100
            self.logger.info(f"  API success rate: {api_success_rate:.1f}%")
        
        self.logger.info(f"\nCalls by type:")
        for call_type, count in stats['calls_by_type'].items():
            self.logger.info(f"    {call_type}: {count}")
        
        self.logger.info("="*80)

def main():
    """Main execution for enhanced dataset generation with LLM-generated metadata, individual regeneration and attempt tracking"""
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: Please set OPENAI_API_KEY environment variable")
        return
   
   
    # input image directory under cwd
    folder_name = "train"
    log_dir = "generation_logs" + f"/{folder_name}"


    # Initialize generator with regeneration support and attempt tracking
    generator = SOTIFDatasetGenerator(
        model="gpt-5",
        random_seed=42,
        log_dir=log_dir,
        max_regen_attempts=5  # Configurable regeneration limit
    )
    
    # Set directories
    # folder_name = "val"
    input_dir = os.getcwd() + f"/{folder_name}"  # Input directory for traffic scene images
    output_dir = "generated_dataset" + f"/{folder_name}"  # Output directory for generated dataset
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        print("Please create the directory and add traffic scene images")
        return
    
    # Generate dataset
    print(f" Starting enhanced generation with LLM metadata & attempt tracking from: {input_dir}")
    print(f" Output will be saved to: {output_dir}")
    
    try:
        results = generator.generate_dataset(input_dir, output_dir)
        
        print(f"\nEnhanced generation with LLM metadata & attempt tracking completed!")
        print(f"Successfully processed: {len(results)} images")
        print(f"Dataset saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Generation failed: {str(e)}")
        generator.logger.error(f"Generation failed: {str(e)}")

if __name__ == "__main__":
    main()