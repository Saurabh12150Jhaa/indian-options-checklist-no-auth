"""Facade – delegates to services.bhavcopy for backward compatibility."""
from services.bhavcopy.downloader import (  # noqa: F401
    DownloadResult,
    download_bhavcopies,
    get_available_date_range,
)
from services.bhavcopy.cache import touch_access  # noqa: F401
