# RAG Testing với Ragas

Hướng dẫn chạy tests và evaluation cho hệ thống RAG.

## Cài đặt

```bash
# Cài đặt test dependencies
pip install -r requirements-test.txt
```

## Cấu trúc thư mục

```
tests/
├── __init__.py
├── conftest.py                    # Pytest fixtures
├── test_rag_retrieval.py         # Tests cho retrieval quality
├── test_answer_generation.py     # Tests cho answer quality
├── test_langgraph_integration.py # Tests cho LangGraph integration
├── test_data/
│   └── sample_questions.json     # Test dataset (generated)
└── evaluation/
    ├── generate_test_data.py     # Script tạo test data
    ├── run_ragas_evaluation.py   # Script chạy Ragas evaluation
    └── reports/                  # Evaluation reports (generated)
```

## Tạo Test Data

Generate synthetic test data từ documents trong Redis:

```bash
cd tests/evaluation
python generate_test_data.py
```

**Environment variables** (optional):

- `TEST_UNIT_NAME`: Tên đơn vị (default: "default_unit")
- `TEST_NUM_DOCS`: Số documents để sample (default: 5)
- `TEST_QUESTIONS_PER_DOC`: Số questions per document (default: 3)

Output: `tests/test_data/sample_questions.json`

## Chạy Unit Tests

```bash
# Chạy tất cả tests
pytest tests/ -v

# Chạy specific test file
pytest tests/test_rag_retrieval.py -v
pytest tests/test_answer_generation.py -v
pytest tests/test_langgraph_integration.py -v

# Chạy với coverage report
pytest tests/ --cov=app.services --cov-report=html
```

## Chạy Ragas Evaluation

Đánh giá comprehensive với Ragas metrics:

```bash
cd tests/evaluation
python run_ragas_evaluation.py
```

**Metrics được tính:**

- **Faithfulness**: Độ trung thực của answer (không hallucination)
- **Answer Relevancy**: Độ liên quan của answer với question
- **Context Precision**: Độ chính xác của retrieved contexts
- **Context Recall**: Độ đầy đủ của retrieved contexts

**Output**:

- `tests/evaluation/reports/ragas_summary_<timestamp>.json` - Metrics summary
- `tests/evaluation/reports/ragas_detailed_<timestamp>.json` - Chi tiết từng test case

## Interpret Kết Quả

### Ragas Metrics (scale 0-1, cao hơn = tốt hơn)

**Faithfulness** (>0.8 = excellent)

- Đo lường mức độ answer dựa trên contexts
- Score thấp = Có hallucination

**Answer Relevancy** (>0.7 = good)

- Đo lường answer có trả lời đúng question không
- Score thấp = Answer không liên quan hoặc verbose

**Context Precision** (>0.6 = acceptable)

- Đo lường retrieved contexts có relevant không
- Score thấp = Retrieval kém chất lượng

**Context Recall** (>0.7 = good)

- Đo lường có đủ contexts để trả lời không
- Score thấp = Thiếu thông tin

### Example Report

```json
{
  "timestamp": "20250130_155900",
  "metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.78,
    "context_precision": 0.72,
    "context_recall": 0.8
  },
  "total_questions": 15
}
```

## Thêm Test Cases Mới

### Manual test cases

Edit `tests/test_data/sample_questions.json`:

```json
[
  {
    "question": "Câu hỏi mới?",
    "ground_truth": "Câu trả lời chuẩn",
    "contexts": ["Context liên quan"],
    "unit_name": "default_unit",
    "difficulty": "medium"
  }
]
```

### Regenerate với LLM

Chạy lại `generate_test_data.py` với parameters mới.

## Troubleshooting

**"No documents found in Redis"**

- Kiểm tra Redis đang chạy
- Kiểm tra đã upload documents chưa
- Kiểm tra `unit_name` đúng chưa

**"Ragas evaluation failed"**

- Kiểm tra `GOOGLE_API_KEY` trong `.env`
- Kiểm tra test data có ground_truth và contexts

**"Context is empty"**

- RAG service cần enhance để return contexts
- Hiện tại dùng contexts từ test data

## CI/CD Integration (Optional)

Thêm vào GitHub Actions:

````yaml
- name: Run RAG Tests
  run: |
# RAG Testing với Ragas

Hướng dẫn chạy tests và evaluation cho hệ thống RAG.

## Cài đặt

```bash
# Cài đặt test dependencies
pip install -r requirements-test.txt
````

## Cấu trúc thư mục

```
tests/
├── __init__.py
├── conftest.py                    # Pytest fixtures
├── test_rag_retrieval.py         # Tests cho retrieval quality
├── test_answer_generation.py     # Tests cho answer quality
├── test_langgraph_integration.py # Tests cho LangGraph integration
├── test_data/
│   └── sample_questions.json     # Test dataset (generated)
└── evaluation/
    ├── generate_test_data.py     # Script tạo test data
    ├── run_ragas_evaluation.py   # Script chạy Ragas evaluation
    └── reports/                  # Evaluation reports (generated)
```

## Tạo Test Data

Generate synthetic test data từ documents trong Redis:

```bash
cd tests/evaluation
python generate_test_data.py
```

**Environment variables** (optional):

- `TEST_UNIT_NAME`: Tên đơn vị (default: "default_unit")
- `TEST_NUM_DOCS`: Số documents để sample (default: 5)
- `TEST_QUESTIONS_PER_DOC`: Số questions per document (default: 3)

Output: `tests/test_data/sample_questions.json`

## Chạy Unit Tests

```bash
# Chạy tất cả tests
pytest tests/ -v

# Chạy specific test file
pytest tests/test_rag_retrieval.py -v
pytest tests/test_answer_generation.py -v
pytest tests/test_langgraph_integration.py -v

# Chạy với coverage report
pytest tests/ --cov=app.services --cov-report=html
```

## Chạy Ragas Evaluation

Đánh giá comprehensive với Ragas metrics:

```bash
cd tests/evaluation
python run_ragas_evaluation.py
```

**Metrics được tính:**

- **Faithfulness**: Độ trung thực của answer (không hallucination)
- **Answer Relevancy**: Độ liên quan của answer với question
- **Context Precision**: Độ chính xác của retrieved contexts
- **Context Recall**: Độ đầy đủ của retrieved contexts

**Output**:

- `tests/evaluation/reports/ragas_summary_<timestamp>.json` - Metrics summary
- `tests/evaluation/reports/ragas_detailed_<timestamp>.json` - Chi tiết từng test case

## Interpret Kết Quả

### Ragas Metrics (scale 0-1, cao hơn = tốt hơn)

**Faithfulness** (>0.8 = excellent)

- Đo lường mức độ answer dựa trên contexts
- Score thấp = Có hallucination

**Answer Relevancy** (>0.7 = good)

- Đo lường answer có trả lời đúng question không
- Score thấp = Answer không liên quan hoặc verbose

**Context Precision** (>0.6 = acceptable)

- Đo lường retrieved contexts có relevant không
- Score thấp = Retrieval kém chất lượng

**Context Recall** (>0.7 = good)

- Đo lường có đủ contexts để trả lời không
- Score thấp = Thiếu thông tin

### Example Report

```json
{
  "timestamp": "20250130_155900",
  "metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.78,
    "context_precision": 0.72,
    "context_recall": 0.8
  },
  "total_questions": 15
}
```

## Thêm Test Cases Mới

### Manual test cases

Edit `tests/test_data/sample_questions.json`:

```json
[
  {
    "question": "Câu hỏi mới?",
    "ground_truth": "Câu trả lời chuẩn",
    "contexts": ["Context liên quan"],
    "unit_name": "default_unit",
    "difficulty": "medium"
  }
]
```

### Regenerate với LLM

Chạy lại `generate_test_data.py` với parameters mới.

## Troubleshooting

**"No documents found in Redis"**

- Kiểm tra Redis đang chạy
- Kiểm tra đã upload documents chưa
- Kiểm tra `unit_name` đúng chưa

**"Ragas evaluation failed"**

- Kiểm tra `GOOGLE_API_KEY` trong `.env`
- Kiểm tra test data có ground_truth và contexts

**"Context is empty"**

- RAG service cần enhance để return contexts
- Hiện tại dùng contexts từ test data

## CI/CD Integration (Optional)

Thêm vào GitHub Actions:

```yaml
- name: Run RAG Tests
  run: |
    pip install -r requirements-test.txt
    pytest tests/ -v --cov=app.services
```

## Best Practices

1. **Regenerate test data** khi documents trong Redis thay đổi
2. **Run tests thường xuyên** để catch regressions sớm
3. **Monitor metrics trends** qua time để track improvements
4. **Review failed test cases** để identify system weaknesses

---

## API Endpoint Testing

### Test qua HTTP Endpoint

Tests trong `test_api_endpoint.py` kiểm tra RAG system qua API endpoint thực tế:

```bash
# Bật server
uvicorn main:app --reload

# Chạy API tests (terminal khác)
pytest tests/test_api_endpoint.py -v
```

**Test Coverage:**

- ✅ Single query processing
- ✅ Multiple sequential queries
- ✅ Conversation threads với context
- ✅ Response time benchmarking
- ✅ Error handling

### Configuration

Sửa `BASE_URL` trong `test_api_endpoint.py` nếu server chạy ở port khác:

```python
BASE_URL = "http://localhost:8000"
```

---

## Visualization

### Generate Charts

Sau khi chạy Ragas evaluation, tạo visualization:

```bash
python tests/evaluation/visualize_results.py
```

**Output Charts:**

- `metrics_summary_*.png` - Bar chart + Radar chart
- `metrics_comparison_*.png` - Performance vs thresholds
- `difficulty_distribution_*.png` - Test difficulty breakdown

### Custom Thresholds

Edit `visualize_results.py` để thay đổi thresholds:

```python
visualizer.plot_metrics_comparison(
    threshold_excellent=0.85,  # Tăng threshold
    threshold_good=0.70
)
```

### Example Output

Charts được lưu trong `tests/evaluation/reports/`:

```
reports/
├── ragas_summary_20250130_163045.json
├── ragas_detailed_20250130_163045.json
├── metrics_summary_20250130_163145.png
├── metrics_comparison_20250130_163145.png
└── difficulty_distribution_20250130_163145.png
```

---

## Complete Workflow Example

### Full Testing Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt

# 2. Generate test data
python tests/evaluation/generate_test_data.py

# 3. Run unit tests
pytest tests/ -v --cov=app.services

# 4. Run Ragas evaluation
python tests/evaluation/run_ragas_evaluation.py

# 5. Create visualizations
python tests/evaluation/visualize_results.py

# 6. Test API endpoints (server must be running)
pytest tests/test_api_endpoint.py -v
```

---

5. **Run evaluation trước khi deploy** changes lớn
