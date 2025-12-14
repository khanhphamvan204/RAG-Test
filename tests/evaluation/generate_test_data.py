"""
Script để tạo synthetic test data cho RAG evaluation
Sử dụng LLM để generate questions từ documents có sẵn trong Redis
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from app.services.embedding_service import get_redis_client
from app.config import Config

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataGenerator:
    """Generate synthetic test questions from existing documents"""
    
    def __init__(self, unit_name: str = "default_unit"):
        self.unit_name = unit_name
        self.redis_client = get_redis_client()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            temperature=0.7,
        )
        
    def get_sample_documents(self, num_samples: int = 10) -> List[Dict]:
        """Lấy sample documents từ Redis"""
        index_name = Config.get_unit_index_name(self.unit_name)
        pattern = f"doc:{index_name}:*"
        
        logger.info(f"Fetching documents from pattern: {pattern}")
        
        sample_docs = []
        for key in self.redis_client.scan_iter(match=pattern, count=num_samples * 2):
            if len(sample_docs) >= num_samples:
                break
                
            try:
                doc_data = self.redis_client.hgetall(key)
                if doc_data and b'content' in doc_data:
                    content = doc_data[b'content'].decode('utf-8')
                    filename = doc_data.get(b'filename', b'unknown').decode('utf-8')
                    
                    # Chỉ lấy documents có content đủ dài
                    if len(content) > 100:
                        sample_docs.append({
                            'content': content,
                            'filename': filename,
                            'key': key.decode('utf-8')
                        })
            except Exception as e:
                logger.error(f"Error reading document {key}: {e}")
                continue
        
        logger.info(f"Retrieved {len(sample_docs)} documents")
        return sample_docs
    
    def generate_questions_from_doc(self, doc: Dict, num_questions: int = 2) -> List[Dict]:
        """Generate questions và ground truth từ một document"""
        content = doc['content']
        filename = doc['filename']
        
        # Truncate content nếu quá dài
        max_content_len = 2000
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        
        prompt_template = PromptTemplate(
            input_variables=["content", "num_questions"],
            template="""
Bạn là một chuyên gia tạo câu hỏi kiểm tra cho hệ thống RAG.

Dựa trên đoạn văn bản sau, hãy tạo {num_questions} câu hỏi test CHẤT LƯỢNG CAO:

ĐOẠN VĂN BẢN:
{content}

YÊU CẦU:
1. Câu hỏi phải CỤ THỂ, có thể trả lời được từ văn bản
2. Câu hỏi phải ĐA DẠNG (factual, procedural, conceptual)
3. Ground truth answer phải CHI TIẾT, DỰA HOÀN TOÀN vào văn bản
4. Bao gồm cả câu hỏi dễ và khó

FORMAT OUTPUT (JSON):
```json
[
  {{
    "question": "Câu hỏi cụ thể?",
    "ground_truth": "Câu trả lời chi tiết dựa trên văn bản",
    "contexts": ["Đoạn văn bản liên quan trực tiếp"],
    "difficulty": "easy/medium/hard"
  }}
]
```

Chỉ trả về JSON array, không giải thích thêm.
"""
        )
        
        try:
            prompt = prompt_template.format(content=content, num_questions=num_questions)
            response = self.llm.invoke(prompt).content
            
            # Extract JSON từ response
            # Loại bỏ ```json và ``` nếu có
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            questions = json.loads(response)
            
            # Add metadata
            for q in questions:
                q['source_filename'] = filename
                q['unit_name'] = self.unit_name
                q['generated_at'] = datetime.now().isoformat()
            
            logger.info(f"Generated {len(questions)} questions from {filename}")
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions from {filename}: {e}")
            return []
    
    def generate_test_dataset(self, 
                            num_docs: int = 5, 
                            questions_per_doc: int = 3) -> List[Dict]:
        """Generate toàn bộ test dataset"""
        logger.info(f"Generating test dataset from {num_docs} documents, " 
                   f"{questions_per_doc} questions per doc")
        
        # Get sample documents
        sample_docs = self.get_sample_documents(num_docs)
        
        if not sample_docs:
            logger.error("No documents found in Redis!")
            return []
        
        # Generate questions
        all_questions = []
        for doc in sample_docs:
            questions = self.generate_questions_from_doc(doc, questions_per_doc)
            all_questions.extend(questions)
        
        logger.info(f"Generated total {len(all_questions)} test questions")
        return all_questions
    
    def save_to_file(self, questions: List[Dict], output_path: Path):
        """Save test data to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(questions)} questions to {output_path}")


def main():
    """Main function"""
    # Configuration
    UNIT_NAME = os.getenv("TEST_UNIT_NAME", "khoa_cong_nghe_thong_tin")  # Changed default
    NUM_DOCS = int(os.getenv("TEST_NUM_DOCS", "5"))
    QUESTIONS_PER_DOC = int(os.getenv("TEST_QUESTIONS_PER_DOC", "4"))
    
    OUTPUT_DIR = Path(__file__).parent.parent / "test_data"
    OUTPUT_FILE = OUTPUT_DIR / "sample_questions.json"
    
    logger.info("=" * 60)
    logger.info("Starting Test Data Generation")
    logger.info(f"Unit: {UNIT_NAME}")
    logger.info(f"Documents to sample: {NUM_DOCS}")
    logger.info(f"Questions per document: {QUESTIONS_PER_DOC}")
    logger.info("=" * 60)
    
    # Generate
    generator = TestDataGenerator(unit_name=UNIT_NAME)
    questions = generator.generate_test_dataset(
        num_docs=NUM_DOCS,
        questions_per_doc=QUESTIONS_PER_DOC
    )
    
    if questions:
        generator.save_to_file(questions, OUTPUT_FILE)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("GENERATION SUMMARY")
        logger.info(f"Total questions: {len(questions)}")
        logger.info(f"Output file: {OUTPUT_FILE}")
        
        # Difficulty breakdown
        difficulty_count = {}
        for q in questions:
            diff = q.get('difficulty', 'unknown')
            difficulty_count[diff] = difficulty_count.get(diff, 0) + 1
        
        logger.info("\nDifficulty breakdown:")
        for diff, count in difficulty_count.items():
            logger.info(f"  {diff}: {count}")
        
        logger.info("=" * 60)
    else:
        logger.error("Failed to generate test questions!")
        sys.exit(1)


if __name__ == "__main__":
    main()
