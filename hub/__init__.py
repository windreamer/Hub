from .api.dataset import Dataset
from .core.transform.transform import transform  # type: ignore
from .util.bugout_reporter import hub_reporter

__version__ = "2.0a1"

# Reporting
hub_reporter.tags.append(f"version:{__version__}")
hub_reporter.system_report(publish=True)
hub_reporter.setup_excepthook(publish=True)
