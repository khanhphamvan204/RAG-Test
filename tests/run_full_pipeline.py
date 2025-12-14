#!/usr/bin/env python
"""
Complete testing pipeline script
Chạy toàn bộ test workflow từ đầu đến cuối
"""
import subprocess
import sys
from pathlib import Path
import time


def run_command(cmd, description, check=True):
    """Run shell command với progress indicator"""
    print("\n" + "=" * 70)
    print(f"Starting: {description}")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            print(f"[SUCCESS] {description}")
        else:
            print(f"[FAILED] {description}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {description}")
        print(f"Error: {e}")
        return False


def main():
    """Main pipeline"""
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║           RAG Testing Pipeline - Complete Workflow                 ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    # Get current Python interpreter (conda env)
    python_exe = sys.executable
    
    # Step 1: Generate test data
    if not run_command(
        f'"{python_exe}" tests/evaluation/generate_test_data.py',
        "Step 1: Generating synthetic test data"
    ):
        print("\n[WARNING] Test data generation failed, but continuing...")
    
    time.sleep(1)
    
    # Step 2: Run unit tests
    if not run_command(
        f'"{python_exe}" -m pytest tests/test_rag_retrieval.py tests/test_answer_generation.py -v',
        "Step 2: Running unit tests",
        check=False  # Don't stop if tests fail
    ):
        print("\n[WARNING] Some unit tests failed, but continuing...")
    
    time.sleep(1)
    
    # Step 3: Run Ragas evaluation
    if not run_command(
        f'"{python_exe}" tests/evaluation/run_ragas_evaluation.py',
        "Step 3: Running Ragas evaluation"
    ):
        print("\n[ERROR] Ragas evaluation failed. Stopping pipeline.")
        sys.exit(1)
    
    time.sleep(1)
    
    # Step 4: Create visualizations
    if not run_command(
        f'"{python_exe}" tests/evaluation/visualize_results.py',
        "Step 4: Creating visualization charts"
    ):
        print("\n[WARNING] Visualization failed, but evaluation completed.")
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Pipeline completed in {elapsed:.2f} seconds")
    print(f"📁 Check results in: tests/evaluation/reports/")
    print("=" * 70)
    
    print("""
    Next steps:
    1. Review Ragas metrics in reports/
    2. Check visualization charts (PNG files)
    3. Analyze failed test cases
    4. Run API tests: pytest tests/test_api_endpoint.py -v
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Pipeline failed with error: {e}")
        sys.exit(1)
