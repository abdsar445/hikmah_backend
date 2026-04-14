import sys
print("starting")
sys.stdout.flush()

try:
    print("importing fastapi...")
    sys.stdout.flush()
    import fastapi
    print("fastapi done")
    sys.stdout.flush()
    
    print("importing pinecone...")
    sys.stdout.flush()
    import pinecone
    print("pinecone done")
    sys.stdout.flush()
    
    print("importing torch...")
    sys.stdout.flush()
    import torch
    print("torch done")
    sys.stdout.flush()
    
    print("importing sentence_transformers...")
    sys.stdout.flush()
    import sentence_transformers
    print("sentence_transformers done")
    sys.stdout.flush()
    
    print("ALL OK")
    sys.stdout.flush()
except Exception as e:
    print(f"Error: {e}")
    sys.stdout.flush()
