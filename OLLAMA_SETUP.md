# Hướng dẫn sử dụng LLM Local với Ollama

## Cài đặt Ollama

1. Tải và cài đặt Ollama từ: https://ollama.ai/download
2. Sau khi cài đặt, khởi động Ollama service

## Tải model LLM

Mở terminal và chạy lệnh để tải model:

```bash
# Tải Llama 3.1 8B (khuyên dùng)
ollama pull llama3.1:8b

# Hoặc các model khác:
ollama pull llama3.1:70b
ollama pull mistral:latest
ollama pull gemma:7b
ollama pull codellama:latest
```

## Cấu hình trong file .env

1. Sao chép file `.env.example` thành `.env`:
   ```bash
   cp .env.example .env
   ```

2. Chỉnh sửa file `.env`:
   ```env
   # Thay đổi LLM_PROVIDER từ gemini sang ollama
   LLM_PROVIDER=ollama
   
   # Cấu hình Ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.1:8b
   ```

## Kiểm tra Ollama đang chạy

```bash
# Kiểm tra danh sách models đã tải
ollama list

# Test model
ollama run llama3.1:8b "Sinh viên năm nhất bị cảnh cáo học vụ khi"
```

## API Endpoint của Ollama

Ollama server chạy tại `http://localhost:11434` với các endpoints:

- `POST /api/generate` - Generate text (streaming/non-streaming)
- `POST /api/chat` - Chat conversation
- `GET /api/tags` - Liệt kê models có sẵn

### Ví dụ test trực tiếp với curl:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Sinh viên năm nhất bị cảnh cáo học vụ khi",
  "stream": false
}'
```

## So sánh các LLM Providers

| Provider | Ưu điểm | Nhược điểm | Chi phí |
|----------|---------|-----------|---------|
| **Ollama (Local)** | - Miễn phí<br>- Bảo mật dữ liệu<br>- Không giới hạn requests<br>- Không cần internet | - Cần phần cứng mạnh<br>- Chất lượng phụ thuộc model<br>- Cài đặt phức tạp hơn | Miễn phí |
| **Gemini** | - Chất lượng cao<br>- Nhanh<br>- Dễ setup | - Có giới hạn rate<br>- Cần API key<br>- Dữ liệu gửi lên cloud | Free tier + trả phí |
| **OpenAI** | - Chất lượng tốt nhất<br>- Nhiều model | - Đắt nhất<br>- Rate limits<br>- Cần API key | Trả phí |
| **Groq** | - Cực nhanh<br>- Free tier tốt | - Giới hạn model<br>- Rate limits | Free tier + trả phí |

## Khuyến nghị cấu hình phần cứng cho Ollama

- **8B models** (llama3.1:8b, gemma:7b): RAM >= 8GB
- **70B models** (llama3.1:70b): RAM >= 48GB
- **GPU**: NVIDIA GPU với CUDA giúp tăng tốc đáng kể

## Troubleshooting

### Ollama không chạy
```bash
# Windows: Kiểm tra service
Get-Service Ollama

# Start Ollama nếu chưa chạy
ollama serve
```

### Model chạy quá chậm
- Chuyển sang model nhỏ hơn (8B thay vì 70B)
- Kiểm tra RAM/CPU usage
- Cân nhắc dùng cloud LLM thay thế

### Connection refused
- Kiểm tra OLLAMA_BASE_URL đúng là `http://localhost:11434`
- Đảm bảo Ollama service đang chạy
- Kiểm tra firewall

## Chuyển đổi giữa các providers

Chỉ cần thay đổi trong file `.env`:

```env
# Dùng Ollama local
LLM_PROVIDER=ollama

# Hoặc dùng Gemini
LLM_PROVIDER=gemini

# Hoặc dùng OpenAI
LLM_PROVIDER=openai

# Hoặc dùng Groq
LLM_PROVIDER=groq
```

Không cần thay đổi code, chỉ cần restart server!
