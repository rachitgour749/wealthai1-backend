import logging
from Services.reporter_service import DailyReportGenerator

logger = logging.getLogger(__name__)

def run_daily_trade_report():
    """
    Job to generate and send daily trade reports via Excel.
    """
    logger.info("Triggering scheduled Daily Trade Report job...")
    try:
        reporter = DailyReportGenerator()
        reporter.generate_daily_reports()
        logger.info("Daily Trade Report job completed successfully.")
    except Exception as e:
        logger.error(f"Daily Trade Report job failed: {e}")
