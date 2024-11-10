from typing import List
from app.api.models.schema import QAResponse
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class SlackService:
    def __init__(self):
        self.mock_messages = []

    def post_results(self, message: str):
        try:
            # Store message in mock storage instead of posting to Slack
            logger.info(f"Posting message to Slack: {message}")
            self.mock_messages.append(message)
            logger.info(f"Mock: Results would be posted to Slack: {message}")
            return "Total messages posted: " + str(len(self.mock_messages))
        except Exception as e:
            logger.error(f"Error in mock Slack posting: {e}")


    def get_mock_messages(self) -> List[str]:
        return self.mock_messages