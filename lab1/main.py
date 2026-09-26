import argparse
from experiment import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ЛР1: Реалізація та перевірка зворотного поширення похибки")
    parser.add_argument(
        "--intentional-error",
        action="store_true",
        help="Запустити дослід із навмисною помилкою (без ділення на N у градієнті логітів)"
    )
    args = parser.parse_args()

    run_pipeline(intentional_error=args.intentional_error)