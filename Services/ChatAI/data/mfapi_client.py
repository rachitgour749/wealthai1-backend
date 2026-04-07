import aiohttp
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class MFApiClient:
    BASE_URL = "https://api.mfapi.in/mf"

    async def search_fund(self, query: str) -> List[Dict]:
        """Search for mutual funds by name."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.BASE_URL}/search", params={"q": query}) as response:
                    if response.status == 200:
                        return await response.json()
                    logger.error(f"MFAPI search failed with status {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error searching MFAPI: {e}")
            return []

    async def get_fund_data(self, scheme_code: int) -> Optional[Dict]:
        """Get historical NAV data for a specific scheme code."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.BASE_URL}/{scheme_code}") as response:
                    if response.status == 200:
                        return await response.json()
                    logger.error(f"MFAPI fund data fetch failed for {scheme_code}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching MFAPI data: {e}")
            return None
