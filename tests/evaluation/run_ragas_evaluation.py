"""
Ragas Evaluation Runner
Chạy đánh giá toàn diện cho RAG system sử dụng Ragas metrics
"""
import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict
from functools import wraps
import asyncio
from datetime import datetime
from datasets import Dataset

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from app.services.rag_service import rag_service
from app.models.vector_models import VectorSearchRequest

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini LLM for Ragas with retry settings
GEMINI_LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0.0,
    # Retry configuration for quota limits
    max_retries=5,
    request_timeout=120,
    # Rate limiting
    max_tokens_per_minute=40000,
    max_requests_per_minute=8  # Below free tier limit of 10
)

# Configure HuggingFace Embeddings for Ragas
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="dangvantuan/vietnamese-document-embedding",
        model_kwargs={'device': 'cuda', 'trust_remote_code': True}, 
        encode_kwargs={'normalize_embeddings': True}
)

# Retry decorator with exponential backoff
def retry_with_backoff(max_retries=5, base_delay=10, max_delay=120):
    """Decorator to retry function with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    # Check for quota exceeded errors
                    if "429" in error_msg or "quota" in error_msg.lower() or "ResourceExhausted" in error_msg:
                        retries += 1
                        if retries >= max_retries:
                            logger.error(f"Max retries ({max_retries}) reached. Giving up.")
                            raise
                        
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (2 ** (retries - 1)), max_delay)
                        
                        # Extract retry delay from error message if available
                        if "Please retry in" in error_msg:
                            try:
                                # Extract seconds from error message
                                import re
                                match = re.search(r'retry in (\d+\.?\d*)s', error_msg)
                                if match:
                                    suggested_delay = float(match.group(1))
                                    delay = max(delay, suggested_delay + 5)  # Add 5s buffer
                            except:
                                pass
                        
                        logger.warning(f"Quota exceeded. Retry {retries}/{max_retries} after {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        # Non-quota error, raise immediately
                        raise
            return None
        return wrapper
    return decorator


class RagasEvaluator:
    """Evaluate RAG system using Ragas metrics"""
    
    def __init__(self, test_data_path: Path):
        self.test_data_path = test_data_path
        self.results = []
        
    def load_test_data(self) -> List[Dict]:
        """Load test questions from JSON"""
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} test questions")
        return data
    
    def run_rag_pipeline(self, question: str, unit_name: str = "default_unit", k: int = 5) -> Dict:
        """Chạy RAG pipeline cho một câu hỏi"""
        try:
            # Call RAG service
            result = rag_service.search_with_llm(
                VectorSearchRequest(
                    query=question,
                    k=k,
                    similarity_threshold=0.3
                ),
                unit_name=unit_name
            )
            
            # Extract response (contexts sẽ lấy từ test data)
            return {
                'answer': result.llm_response,
                'unit_name': result.unit_name
            }
        except Exception as e:
            logger.error(f"Error running RAG pipeline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'answer': "Error",
                'unit_name': unit_name
            }
    
    def prepare_ragas_dataset(self, test_data: List[Dict]) -> Dataset:
        """Prepare dataset cho Ragas evaluation"""
        logger.info("Running RAG pipeline for all test questions...")
        
        evaluation_data = {
            'question': [],
            'answer': [],
            'contexts': [],
            'ground_truth': []
        }
        
        for idx, item in enumerate(test_data):
            question = item['question']
            ground_truth = item['ground_truth']
            expected_contexts = item.get('contexts', [])
            unit_name = item.get('unit_name', 'default_unit')
            
            logger.info(f"Processing question {idx + 1}/{len(test_data)}: {question[:50]}...")
            
            # Run RAG
            rag_result = self.run_rag_pipeline(question, unit_name)
            
            # Store for evaluation
            evaluation_data['question'].append(question)
            evaluation_data['answer'].append(rag_result['answer'])
            
            # Sử dụng contexts từ test data (ground truth)
            evaluation_data['contexts'].append(expected_contexts)
            
            evaluation_data['ground_truth'].append(ground_truth)
            
            # Store detailed result
            self.results.append({
                'question': question,
                'answer': rag_result['answer'],
                'ground_truth': ground_truth,
                'contexts': expected_contexts,
                'unit_name': unit_name,
                'source_filename': item.get('source_filename', 'unknown'),
                'difficulty': item.get('difficulty', 'unknown')
            })
            
            # Thêm delay để tránh vượt quota
            if idx < len(test_data) - 1:  # Không sleep ở câu hỏi cuối
                wait_time = 5  # Tăng lên 5s để đảm bảo không vượt quota
                logger.info(f"Waiting {wait_time} seconds before next question...")
                time.sleep(wait_time)
        
        return Dataset.from_dict(evaluation_data)
    
    @retry_with_backoff(max_retries=5, base_delay=40, max_delay=180)
    def run_evaluation_with_retry(self, dataset):
        """Run Ragas evaluation with retry logic"""
        logger.info("Running Ragas evaluation with Gemini LLM (2 core metrics to avoid quota)...")
        logger.info("WARNING: This may take a while due to API rate limits...")
        
        # Chờ trước khi bắt đầu evaluation để đảm bảo quota đã reset
        logger.info("⏳ Waiting 15 seconds before starting evaluation...")
        time.sleep(15)
        
        # Split dataset into smaller batches to avoid quota issues
        batch_size = 3  # Giảm từ 5 xuống 3 để tránh vượt quota
        all_data = dataset.to_pandas()
        total_samples = len(all_data)
        
        logger.info(f"Processing {total_samples} samples in batches of {batch_size}...")
        
        results = []
        for batch_idx in range(0, total_samples, batch_size):
            batch_end = min(batch_idx + batch_size, total_samples)
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing batch {batch_idx//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
            logger.info(f"Questions {batch_idx + 1} to {batch_end}")
            logger.info(f"{'='*60}")
            
            # Create batch dataset
            batch_data = all_data.iloc[batch_idx:batch_end]
            batch_dataset = Dataset.from_pandas(batch_data)
            
            # Evaluate batch with retry - XỬ LÝ TỪNG CÂU MỘT VỚI DELAY
            try:
                # Process each question in batch individually with delay
                batch_results_list = []
                for q_idx in range(len(batch_dataset)):
                    single_q_data = batch_data.iloc[q_idx:q_idx+1]
                    single_q_dataset = Dataset.from_pandas(single_q_data)
                    
                    logger.info(f"  Evaluating question {batch_idx + q_idx + 1}/{total_samples}...")
                    
                    single_result = evaluate(
                        dataset=single_q_dataset,
                        metrics=[
                            faithfulness,      # Đánh giá độ trung thực
                            answer_relevancy,  # Đánh giá độ liên quan
                        ],
                        llm=GEMINI_LLM,
                        embeddings=EMBEDDINGS
                    )
                    batch_results_list.append(single_result)
                    
                    # Chờ giữa các câu hỏi trong batch
                    if q_idx < len(batch_dataset) - 1:
                        inner_wait = 20  # Tăng lên 20 giây giữa mỗi câu hỏi để tránh quota
                        logger.info(f"  ⏱️  Waiting {inner_wait}s before next question in batch...")
                        time.sleep(inner_wait)
                
                # Combine individual results into batch result
                combined_batch_df = None
                for single_result in batch_results_list:
                    single_df = single_result.to_pandas()
                    if combined_batch_df is None:
                        combined_batch_df = single_df
                    else:
                        combined_batch_df = combined_batch_df._append(single_df, ignore_index=True)
                
                # Create a combined result object
                batch_result = evaluate(
                    dataset=Dataset.from_pandas(combined_batch_df),
                    metrics=[faithfulness, answer_relevancy],
                    llm=GEMINI_LLM,
                    embeddings=EMBEDDINGS
                )
                
                results.append(batch_result)
                logger.info(f"✓ Batch {batch_idx//batch_size + 1} completed successfully")
                
                # Wait between batches to avoid quota - TĂNG LÊN 120s
                if batch_end < total_samples:
                    wait_time = 120  # Tăng từ 90s lên 120s để đảm bảo không vượt quota
                    logger.info(f"⏳ Waiting {wait_time}s before next batch to avoid quota limits...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"Batch {batch_idx//batch_size + 1} failed: {e}")
                raise  # Re-raise to trigger retry decorator
        
        # Combine results from all batches
        if results:
            # Combine all batch results into single result
            combined_df = None
            for result in results:
                batch_df = result.to_pandas()
                if combined_df is None:
                    combined_df = batch_df
                else:
                    combined_df = combined_df._append(batch_df, ignore_index=True)
            
            # Convert back to Dataset and evaluate to get final metrics
            final_dataset = Dataset.from_pandas(combined_df)
            return evaluate(
                dataset=final_dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=GEMINI_LLM,
                embeddings=EMBEDDINGS
            )
        
        return None
    
    def evaluate(self) -> Dict:
        """Run Ragas evaluation"""
        try:
            # Load test data
            test_data = self.load_test_data()
            
            # Prepare dataset cho Ragas
            dataset = self.prepare_ragas_dataset(test_data)
            
            # Run evaluation with retry and batching
            eval_result = self.run_evaluation_with_retry(dataset)
            
            logger.info("\nRagas evaluation completed successfully!")
            return eval_result
            
        except Exception as e:
            logger.error(f"\nEvaluation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def save_results(self, eval_result, output_dir: Path):
        """Save evaluation results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics summary
        summary_file = output_dir / f"ragas_summary_{timestamp}.json"
        if eval_result:
            # Convert EvaluationResult to pandas DataFrame
            df = eval_result.to_pandas()
            
            # Extract metrics safely
            metrics = {}
            if 'faithfulness' in df.columns:
                metrics['faithfulness'] = float(df['faithfulness'].mean())
            if 'answer_relevancy' in df.columns:
                metrics['answer_relevancy'] = float(df['answer_relevancy'].mean())
            if 'context_precision' in df.columns:
                metrics['context_precision'] = float(df['context_precision'].mean())
            if 'context_recall' in df.columns:
                metrics['context_recall'] = float(df['context_recall'].mean())
            
            summary = {
                'timestamp': timestamp,
                'metrics': metrics,
                'total_questions': len(self.results)
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved metrics summary to {summary_file}")
        
        # Save detailed results
        detailed_file = output_dir / f"ragas_detailed_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved detailed results to {detailed_file}")
        
        # Print summary
        self.print_summary(eval_result)
    
    def print_summary(self, eval_result):
        """Print evaluation summary"""
        print("\n" + "=" * 70)
        print("RAGAS EVALUATION SUMMARY")
        print("=" * 70)
        
        if eval_result:
            # Convert to DataFrame for safe access
            df = eval_result.to_pandas()
            
            print(f"\nMetrics:")
            if 'faithfulness' in df.columns:
                print(f"  • Faithfulness:       {df['faithfulness'].mean():.4f}")
            else:
                print("  • Faithfulness:       N/A")
                
            if 'answer_relevancy' in df.columns:
                print(f"  • Answer Relevancy:   {df['answer_relevancy'].mean():.4f}")
            else:
                print("  • Answer Relevancy:   N/A")
                
            if 'context_precision' in df.columns:
                print(f"  • Context Precision:  {df['context_precision'].mean():.4f}")
                
            if 'context_recall' in df.columns:
                print(f"  • Context Recall:     {df['context_recall'].mean():.4f}")
        
        print(f"\nTotal Questions: {len(self.results)}")
        
        # Difficulty breakdown
        difficulty_count = {}
        for r in self.results:
            diff = r.get('difficulty', 'unknown')
            difficulty_count[diff] = difficulty_count.get(diff, 0) + 1
        
        print(f"\nDifficulty Breakdown:")
        for diff, count in difficulty_count.items():
            print(f"  • {diff}: {count}")
        
        print("=" * 70 + "\n")


def main():
    """Main function"""
    # Paths
    TEST_DATA_FILE = Path(__file__).parent.parent / "test_data" / "sample_questions.json"
    OUTPUT_DIR = Path(__file__).parent.parent / "evaluation" / "reports"
    
    if not TEST_DATA_FILE.exists():
        logger.error(f"Test data file not found: {TEST_DATA_FILE}")
        logger.error("Please run generate_test_data.py first!")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("Starting Ragas Evaluation")
    logger.info(f"Test data: {TEST_DATA_FILE}")
    logger.info("=" * 70)
    
    # Run evaluation
    evaluator = RagasEvaluator(TEST_DATA_FILE)
    eval_result = evaluator.evaluate()
    
    # Save results
    evaluator.save_results(eval_result, OUTPUT_DIR)
    
    logger.info("\nEvaluation completed! Check reports in:")
    logger.info(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
