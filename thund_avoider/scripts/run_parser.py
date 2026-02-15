from wakepy.modes import keep

from thund_avoider.services.parser.parser import Parser
from thund_avoider.settings import settings, PARSER_DATA_PATH, PARSER_RESULT_PATH


if __name__ == "__main__":
    parser = Parser(settings.parser_config)

    with keep.presenting():
        parser.get_data(PARSER_DATA_PATH)
        parser.collect_data(
            input_folder=PARSER_DATA_PATH,
            output_path=PARSER_RESULT_PATH,
        )
