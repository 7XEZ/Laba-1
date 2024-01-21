# Лабораторная работа №2: Использование методов обработки естественного языка для поддержки исследований в области психологии: анализ психологических текстов и интерпретация эмоций
---

## Установка

Для работы с инструментом требуется установить следующие библиотеки:

```bash
pip install nltk textblob
python -m textblob.download_corpora
python -m nltk.downloader vader_lexicon
python -m nltk.downloader punkt
python -m nltk.downloader stopword
```

## Описание
Этот проект представляет собой инструмент анализа эмоций в тексте, использующий методы обработки естественного языка (NLP). Он способен анализировать эмоциональный тон текста, определять ключевые слова и визуализировать частотность слов. Этот инструмент может быть особенно полезен для исследований в области психологии, социологии и любых других областей, где требуется глубокое понимание эмоционального контента текста.

## Функционал
Анализ Сентимента: Использует VADER и TextBlob для определения эмоционального тона текста.
Извлечение Ключевых Слов: Определяет ключевые слова в тексте, исключая стоп-слова.
Визуализация Частотности Слов: Визуализирует наиболее часто используемые слова в тексте.

## Как Использовать

Убедитесь, что у вас установлены Python и необходимые библиотеки (nltk, textblob)

Запустите скрипт и передайте ему текст для анализа.

Получите результаты анализа, включая оценку эмоционального тона, ключевые слова и визуализацию частотности слов.


