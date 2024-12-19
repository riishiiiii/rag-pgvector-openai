import redis.asyncio as redis
import json
from typing import Optional
from functools import lru_cache
from dotenv import load_dotenv
from config import Settings

load_dotenv()

@lru_cache()
def get_settings():
    return Settings()

redis_dns = get_settings().redis_dns


class RedisCacheService:
    def __init__(self, redis_url: str = redis_dns) -> None:
        self.redis = redis.from_url(redis_url)

    async def set(self, key: str, value: dict, expire: int = 3600) -> None:
        """Set a value in the Redis cache."""
        await self.redis.set(key, json.dumps(value), ex=expire)

    async def get(self, key: str) -> Optional[dict]:
        """Get a value from the Redis cache."""
        cached_value = await self.redis.get(key)
        return json.loads(cached_value) if cached_value else None

    async def delete(self, key: str) -> None:
        """Delete a key from the Redis cache."""
        await self.redis.delete(key)

    async def close(self) -> None:
        """Close the Redis connection."""
        await self.redis.close()
