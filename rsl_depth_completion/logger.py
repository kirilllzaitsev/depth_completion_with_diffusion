import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d M %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# configure logging at the root level of Lightning
# logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

# configure logging on module level, redirect to file
# logger = logging.getLogger("lightning.pytorch.core")
# logger.addHandler(logging.FileHandler("core.log"))
