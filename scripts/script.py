import attrs
import hydra
import logging
from hydra.core.config_store import ConfigStore

logger = logging.getLogger(__name__)


@attrs.define
class Config:
    pass


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    logger.info("Done!")


if __name__ == "__main__":
    main()
