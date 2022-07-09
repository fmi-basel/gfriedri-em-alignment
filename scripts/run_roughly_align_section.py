from prefect import Flow, Parameter, unmapped
from prefect.executors import LocalDaskExecutor

from sbem.section_align.param_schema import LoadSectionsConfig, AlignSectionsConfig

with Flow("Roughly-align-sections") as flow:
    load_sections_config = Parameter("load_sections_config", required=True)
    align_config = Parameter("align_config", required=True)
    offset_dir = Parameter("offset_dir", required=True)
    logger = prefect.context.get("logger")
    logger.info("INFO level log message.")

    section_pairs = load_section_pairs()

    align_obj = align_section_pair.map(section_pairs,
                                       unmapped(align_config),
                                       unmapped(offset_dir))
    # parallelism


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    load_sections_config = LoadSectionsConfig.from_dict(
        dict(config["LOAD_SECTIONS"]))
    align_config = AlignSectionsConfig.from_dict(dict(config["ALIGN_SECTIONS"]))
    offset_dir = config["OUTPUT"]["offset_dir"]

    flow.executor = LocalDaskExecutor()

    kwargs = dict(load_sections_config=load_sections_config,
                  align_config=align_config,
                  offset_dir=offset_dir)
    flow.run(parameters=kwargs)
