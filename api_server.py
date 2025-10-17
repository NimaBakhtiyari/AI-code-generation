#!/usr/bin/env python
"""
Standalone FastAPI Server for Code Generation
No heavy dependencies - pure template-based generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Neuro-Symbolic Code Generation API",
    description="Template-based code generation API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    language: str = Field(default="python")


class GenerateResponse(BaseModel):
    code: str
    language: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


@app.get("/")
async def root():
    return {
        "message": "Neuro-Symbolic Code Generation API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "generate": "/generate (POST)",
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "0.1.0"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    prompt_lower = request.prompt.lower()
    
    if "fibonacci" in prompt_lower:
        code = '''def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

if __name__ == "__main__":
    print([fibonacci(i) for i in range(10)])
'''
    elif "prime" in prompt_lower:
        code = '''def is_prime(n: int) -> bool:
    """Check if number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

if __name__ == "__main__":
    primes = [n for n in range(2, 100) if is_prime(n)]
    print(f"Primes: {primes}")
'''
    elif "factorial" in prompt_lower:
        code = '''def factorial(n: int) -> int:
    """Calculate factorial."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

if __name__ == "__main__":
    print([factorial(i) for i in range(10)])
'''
    else:
        code = f'''def generated_function():
    """
    {request.prompt}
    """
    pass

if __name__ == "__main__":
    generated_function()
'''
    
    return GenerateResponse(
        code=code,
        language=request.language,
        metadata={"method": "template", "prompt_length": len(request.prompt)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
