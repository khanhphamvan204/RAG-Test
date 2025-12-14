"""
Debug script để kiểm tra documents trong Redis và pattern
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.embedding_service import get_redis_client
from app.config import Config

def check_redis_documents():
    """Kiểm tra documents trong Redis"""
    redis_client = get_redis_client()
    
    print("=" * 60)
    print("REDIS DOCUMENTS CHECK")
    print("=" * 60)
    
    # Check all doc patterns
    patterns = [
        "doc:*",
        "doc:unit_index_*",
        "*:chunk:*",
        "document:*"
    ]
    
    for pattern in patterns:
        print(f"\nPattern: {pattern}")
        keys = list(redis_client.scan_iter(match=pattern, count=10))
        print(f"Found {len(keys)} keys")
        
        if keys:
            print(f"Sample keys (first 5):")
            for key in keys[:5]:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                print(f"  - {key_str}")
    
    # Check default unit index name
    print("\n" + "=" * 60)
    default_index = Config.get_unit_index_name("default_unit")
    print(f"Default unit index name: {default_index}")
    pattern = f"doc:{default_index}:*"
    print(f"Looking for pattern: {pattern}")
    
    keys = list(redis_client.scan_iter(match=pattern, count=10))
    print(f"Found {len(keys)} documents")
    
    # Try without unit prefix
    print("\n" + "=" * 60)
    print("Trying without unit prefix...")
    pattern = "doc:default_unit:*"
    keys = list(redis_client.scan_iter(match=pattern, count=10))
    print(f"Pattern 'doc:default_unit:*': {len(keys)} keys")
    
    if keys:
        print("Sample:")
        for key in keys[:3]:
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            print(f"  - {key_str}")

if __name__ == "__main__":
    check_redis_documents()
