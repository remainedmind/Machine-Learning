# -*- coding: utf-8 -*-
"""
Модуль для машинного обучения, включающий один класс.
Методы класса используются в скриптах обучения модели
и генерации текста

"""

import re
import os
import sys
import pickle as pcl
from collections import defaultdict
from numpy import random


class NgramHandler:
    """
    Класс для работы с N-граммной моделью. Основные, публичные
    методы - fit и generate - для обучения и создания текста.
    """

    def __init__(self):
        self.main_path = os.getcwd()  # Текущая директория.

    @staticmethod
    def __get_to_files(input_dir: str) -> list[str]:
        """
        Функция открывает папку с обучающими
        текстами и формирует список подходящих файлов.

        """

        print('Открываем директорию с файлами...')
        try:
            os.chdir(input_dir)
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Ошибка! Папка <{input_dir}> не существует.')

        #   Получим список документов для обучения модели.
        # Нам подходят файлы с кодировкой utf-8 и расширением txt, в
        # названии которых нет точки.
        files_list = [f for f in os.listdir() if f.split('.')[1] == 'txt']
        if not files_list:  # Пустой список
            raise FileNotFoundError(
                'Ошибка! В указанной '
                'директории нет подходящих файлов. '
                )
        return files_list

    def __get_corpus_from_files(self, files: list[str]) -> str:
        """
        Получение из коллекции документов корпуса для обучения
        модели. Собираем все тексты в корпус, после чего
        возвращаемся в рабочую директорию.

        """

        print('Открываем файлы для чтения...')
        corpus = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as text:
                corpus.append(text.read()+'. ')
        os.chdir(self.main_path)  # Назад в исходную директорию.
        return ' '.join(corpus).lower()

    @staticmethod
    def __loot_from_input() -> str:
        """ Функция для получения обучающего корпуса из консоли. """

        data = []
        print(
            'Введите текст, на котором вы хотите обучить модель. '
            'Максимальный объём не ограничен.\nЧтобы закончить'
            ' ввод текста введите Ctrl+D или Ctrl+Z+Enter. ')
        for line in sys.stdin:
            data.append(line)
        return ''.join(data).lower()

    def __prepare_corpus(self, corpus: str) -> None:
        """
        Подготовим корпус для дальнейшего анализа: разделим
        на элементы, токенизируем, уберём всё лишнее.

        """

        print('Обрабатываем текстовые данные...')
        # Текст разобьём на предложения, чтобы
        # обрабатывать первое и последнее слова.
        corpus = corpus.replace('?', '.').replace('!', '.').split('.')
        # Если случится так, что в тексте нет ни одного из
        # знаков [? . !], мы получим одноэлементный список,
        # который программа не сможет обработать.

        if len(corpus) == 1:
            # Самое простое решение - дублировать единственный
            # элемент. Менее осмысленным анализ корпуса из
            # одного предложения от этого не станет.
            corpus.append(corpus[0])

        # Произведём токенизацию. После неё в полученном
        # списке могут оказаться пустые подсписки [].
        # Удалим их позже.
        corpus = [(re.findall(
            r"[а-яё]+-[а-яё]+|[а-яё]+,?", sentence)
        ) for sentence in corpus]
        if corpus == [[], []]:  # Получили пустой корпус
            raise ValueError(
                'Не найдено ни одного русского слова! '
            )

        # Короткие предложения объединим с предыдущими.
        # На случай, если предыдущее предложение тоже слишком
        # короткое, будем использовать рекурсию.
        def append_to_last(
                    things: list[list], index: int, phrase: list) -> None:
            """
            Рекурсивная функция для добавления слов предложения к
            предыдущему предложению
            """

            if index == 0:
                # Если попали на первое предложение,
                # добавим к нему независимо от его длины.
                things[index] = [*things[index], *phrase, *phrase]
                return
            if len(things[index - 1]) > 2:  # Можем добавить к предыдущему
                [things[index - 1].append(word) for word in phrase]
                # Чистим элемент, из которого мы переместили элементы
                things[index] = []
                return
            else:
                # Если предыдущее предложение тоже короткое
                # (или пустое), заглядываем ещё на один
                # элемент "назад"
                append_to_last(things, index - 1, phrase)

        for i, s in enumerate(corpus):
            if len(s) < 3:
                # Метод .pop() здесь не подойдёт, так как
                # из-за него будут меняться индексы элементов и итерация
                # нарушится.
                append_to_last(things=corpus, index=i, phrase=corpus[i])
        # Теперь в списке стало ещё больше "пустышек" [].
        # Удалим их все.
        corpus = [c for c in corpus if c]
        # В конец каждого предложения добавляем слово-маркер.
        # Затем будем заменять его на точку.
        self.corpus = [(*c, '@END@') for c in corpus]

    def __make_trigrams(self) -> defaultdict:
        """
        Функция для составления триграммы. В этой и двух
        аналогичных функциях не будем рассчитывать вероятность
        появления слова: добавим слова в список возможных
        вариантов ровно столько раз, сколько они встречаются
        в тексте - тогда при рандомном выборе слова эта
        вероятность будет учтена сама собой.

        """

        # Создаём словарь с ключом-списком по умолчанию, чтобы
        # не проверять, был ли ранее добавлен выбранный ключ.
        data = defaultdict(list)
        for index, chain in enumerate(self.corpus):
            for i, word in enumerate(chain):
                if i <= 1:  # Первые два слова пропускаем.
                    continue
                elif word == '@END@' and index < len(self.corpus) - 1:
                    #  Если попадаем в конец предложения -  смотрим, на
                    # какое слово начинается следующее.
                    key = (chain[i - 1], word)
                    word = self.corpus[index + 1][0]
                else:
                    key = (chain[i - 2], chain[i - 1])
                data[key].append(word)
        return data

    def __make_bigrams(self) -> defaultdict:
        """ Функция для составления биграммы. """

        data = defaultdict(list)
        for index, chain in enumerate(self.corpus):
            for i, word in enumerate(chain):
                if i == 0 and word != ' - ':
                    data['BEGIN'].append(word)
                if i == 0:
                    if index > 0:
                        key = self.corpus[index - 1][-2]
                    else:
                        continue
                    # Первое слово каждого предложения добавим в
                    # специальную категорию - с этих слов могут
                    # начинаться сгенерированные предложения. Из
                    # книжных диалогов сюда могут пробраться тире - их
                    # включать не будем.
                else:
                    key = (chain[i - 1])
                data[key].append(word)
        return data

    def __make_fourgrams(self) -> defaultdict:
        """ Функция для составления квадрограммы. """

        data = defaultdict(list)
        for index, chain in enumerate(self.corpus):
            for i, word in enumerate(chain):
                if i <= 2:  # Первые три слова пропускаем
                    continue
                elif word == '@END@' and index < len(self.corpus) - 1:
                    #  Если попадаем в конец предложения -  смотрим, на
                    # какое слово начинается следующее, дабы весь текст
                    # имел хоть какой-то смысл.
                    key = (chain[i - 2], chain[i - 1], word)
                    word = self.corpus[index + 1][0]
                else:
                    key = (chain[i - 3], chain[i - 2], chain[i - 1])
                data[key].append(word)
        return data

    def __getstate__(self) -> dict:
        """
        Магический метод для сериализации и записи в файл
        методов экземляра через pickle.

        """
        state = {
            "trigram": self.trigram,
            "quadrogram": self.quadrogram,
            "bigram": self.bigram
        }
        return state

    def __setstate__(self, state: dict):
        """
        Магический метод для восстановления сериализованных
        методов экземляра из файла.

        """
        self.bigram, self.trigram, self.quadrogram = (
            state["bigram"], state["trigram"], state["quadrogram"]
        )

    def fit(
            self, model_path: str | None,
            input_dir: str | None) -> None:
        """
        Обучение и сохранение. Обученная модель - экземпляр
        класса с нужными атрибутами для создания текста.

        """
        if input_dir:
            self.__prepare_corpus(
                self.__get_corpus_from_files(self.__get_to_files(input_dir))
            )
        else:
            self.__prepare_corpus(self.__loot_from_input())

        print('Обучаем модель...')

        # Две n-граммы сохраним как обычные словари:
        # при обращении к ним по несуществующему ключу
        # будем ловить и перехватывать ошибку KeyError.
        self.trigram, self.bigram, self.quadrogram = (
            dict(self.__make_trigrams()),
            dict(self.__make_bigrams()),
            self.__make_fourgrams()
        )

        if model_path:
            try:
                os.chdir(model_path)
            except FileNotFoundError:
                os.makedirs(model_path)
                os.chdir(model_path)

        with open('model.pkl', 'wb') as f:
            pcl.dump(self, f)
        os.chdir(self.main_path)
        print('Модель успешно обучена и сохранена!')

    def __choose_next_word(self, phrase: list[str]) -> str:
        """
        Функция для определения следующего слова. Сначала
        ищем продолжение фразы в квадрограмме - если не
        получилось (KeyError), переходим к триграмме, если
        опять не вышло - к биграмме. Если снова безуспешно,
        выбираем случайное слово по ключу BEGIN.

        """
        selection = self.quadrogram[tuple(phrase)]
        # Если в квадрограмме меньше двух вариантов для
        # следующего слова, то выберем из n-грамм меньшего
        # порядка, чтобы текст был разнообразным, а не
        # копировал один из фрагментов  корпуса.
        if len(selection) > 1:
            # В рамках этой проверки происходит и проверка на
            # наличие ключа в принципе (len > 1 -> len >0)
            return random.choice(selection)

        try:
            word = random.choice(self.trigram[tuple(phrase[1:])])
        except KeyError:
            try:
                word = random.choice(self.bigram[phrase[2]])
            except KeyError:
                word = random.choice(self.bigram['BEGIN'])
        finally:
            return word

    def __create_text(self, opus_length: int, seed: list[str]) -> str:
        """
        Функция для генерации.. Получим список из заданного
        количества слов и объединим их в текст.

        """

        print('Генерируем текст...')
        seed = [s.lower() for s in seed]
        if len(seed) > 3:
            return 'Некорректная начальная фраза! '
        phrase = [*['' for _ in range(3 - len(seed))], *seed]
        opus = []

        # Добавим начальную фразу в текст, если это имеет смысл:
        if 'random' not in seed:
            # Но добавим только непустые строки
            [opus.append(p) for p in phrase if p]

        for i in range(opus_length):
            word = self.__choose_next_word(phrase)
            opus.append(word)
            phrase = [*phrase[1:], word]  # Сдвигаемся на одно слово.

        # Добавим заглавные буквы в предложениях и точки между
        # предложениями.
        opus = '. '.join(
            [s.capitalize() for s in (' '.join(opus).split(" @END@ "))]
        )
        # Добавим точку в самом конце.
        if opus[-1] in (',', '.'):
            opus = opus[:-1]
        opus += '.'
        return opus

    def generate(
            self, model_path: str | None,
            seed: list[str], length: int) -> None:
        """
        Метод, вызываемый скриптом для генерации текста. Переходит
        к файлу с моделью и загружает её, после чего запускает алгоритм
        генерации и выводит результат.

        """
        if model_path:
            print('Открываем нужную директорию...')
            try:
                os.chdir(model_path)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f'Директория "{model_path}" не существует! ')

        print('Загружаем модель...')
        # Создадим переменную, потому что после загрузки модели
        # атрибуты перезапишутся.
        x = self.main_path
        try:
            with open('model.pkl', 'rb') as f:
                self = pcl.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"""{(f'в директори {model_path} ' if model_path else '')}"""
                f"""модель не найдена! Проверьте её наличие в """
                f"""указанном месте. """)

        print('Модель успешно загружена. ')
        os.chdir(x)
        print(self.__create_text(opus_length=length, seed=seed))
