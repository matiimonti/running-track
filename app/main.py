import time
import uuid
import redis.asyncio as aredis
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from structlog.contextvars import bind_contextvars, clear_contextvars

from app.config import settings
from app.logging_config import setup_logging, logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("startup", environment=settings.environment)
    yield
    logger.info("shutdown")


app = FastAPI(title="Pathfinder", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    clear_contextvars()
    request_id = str(uuid.uuid4())[:8]
    bind_contextvars(
        request_id=request_id, method=request.method, path=request.url.path
    )

    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info("request", status=response.status_code, duration_ms=duration_ms)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
async def health():
    redis_status = "ok"
    try:
        r = aredis.from_url(settings.redis_url)
        await r.ping()
        await r.aclose()
    except Exception as e:
        redis_status = f"error: {str(e)}"

    return {
        "status": "ok" if redis_status == "ok" else "degraded",
        "environment": settings.environment,
        "dependencies": {
            "redis": redis_status,
        },
    }
