""" Скрипт для обучения модели. """
from argparse import ArgumentParser
from wordgenerator import NgramHandler as nHand


parser = ArgumentParser(
    description="Обучение модели по заданным текстам.",
    usage='train.py -in data -m model_dir'
)

parser.add_argument(
    '-in', '--input_dir', action="store",
    type=str, metavar='INPUT FROM', default=[],
    # Добавим возможность использовать папки с пробелами
    # в названии. Вместо строки получим список.
    nargs="+",
    help=('путь к обучающим текстам. '
          'Если не задан - текст вводится из консоли.')
)

parser.add_argument(
    '-m', '--model', action="store", default=[],
    type=str, metavar='PATH', nargs="+",
    help=(
        'путь, по которому будет сохранена модель. '
        'По умолчанию - директория скрипта.'
    )
)

args = parser.parse_args()

if __name__ == '__main__':
    trainer = nHand()
    trainer.fit(
        model_path=' '.join(args.model),
        input_dir=' '.join(args.input_dir)
    )
