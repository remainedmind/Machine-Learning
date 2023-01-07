""" Скрипт для генерации текста на основе обученной модели."""
from argparse import ArgumentParser
from wordgenerator import NgramHandler as nHand


parser = ArgumentParser(
    description='Генерация текста на основе обученной модели.',
    usage='generate.py -m model_dir -l 80 -p Первый раз'
)

parser.add_argument(
    '-m', '--model', action="store", default=[],
    metavar='PATH', type=str,
    # Добавим возможность использовать папки с пробелами в названии.
    nargs="+",  # Вместо строки получим список
    help=(
        'директория, из которой будет загружена модель. '
        'Если не задана - программа попытается найти модель в '
        'директории скрипта. При отсутствии файла возникнет '
        'ошибка!'
    )
)

parser.add_argument(
    '-p', '--prefix', action="store",
    # Добавим возможность парсить несколько аргументов (слов).
    nargs="+", type=str,
    #  Несмотря на заданный тип, при вводе даже одного слова
    #  значением переменной будет список.
    metavar='SEED', default=['random'],
    help=('начальная фраза для генерации - одно, два или три слова.'
          " По умолчанию программа начнёт со случайного слова.")
)

parser.add_argument(
    '-l', '--lenght', action="store",
    type=int, metavar='WORDS AMOUNT', default=50,
    help=('Длина генерируемой последовательности - количество '
          "слов. По умолчанию будет выведено 50 слов. ")
)
args = parser.parse_args()

if __name__ == '__main__':
    instance = nHand()
    instance.generate(
        model_path=' '.join(args.model), seed=args.prefix, length=args.lenght
    )
