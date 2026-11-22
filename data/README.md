# Данные для EDA

Здесь лежат таблицы, которые читают [`notebooks/eda.ipynb`](../notebooks/eda.ipynb) и [`notebooks/models.ipynb`](../notebooks/models.ipynb). Это выжимка из открытых данных SkillsBench. Текст логов в репозиторий не попадает, только посчитанные из него признаки и короткие цитаты, которыми классификатор подтверждает ответ.

Пересобрать таблицы можно так.

```bash
uv sync --group eda
uv run python scripts/fetch_eda_data.py    # около 250 МБ в data/raw, эта папка не в git
uv run python scripts/build_eda_tables.py  # таблицы для EDA в data/processed
uv run python scripts/evaluate_regression.py               # проверки теста «стало хуже»
uv run python scripts/classify_failures.py --split tune    # классификатор, часть для настройки
uv run python scripts/classify_failures.py --split test    # классификатор, отложенная часть
```

Классификатор вызывает Claude Haiku через Claude Code в режиме без интерфейса, нужен вход в Claude Code. Ответы модели лежат в `failure_llm_cache.jsonl`, поэтому повторный запуск ничего не стоит и дает те же цифры.

## Источники

| Что | Откуда, версия | Лицензия |
|---|---|---|
| Запуски, логи Claude Code, аудит провалов | Датасет [benchflow/skillsbench-leaderboard](https://huggingface.co/datasets/benchflow/skillsbench-leaderboard), коммит `f104580` | Apache-2.0 |
| Задачи и навыки | Репозиторий [benchflow-ai/skillsbench](https://github.com/benchflow-ai/skillsbench), релиз v1.1, коммит `2773838` | Apache-2.0 |

## Таблицы в `processed/`

| Файл | Строк | Одна строка это |
|---|---|---|
| `tasks.csv` | 87 | Задача. Категория, сложность, форматы файлов, лимит времени агента, число навыков и флаг `doc_data` для задач на документы и данные |
| `skills.csv` | 232 | Навык задачи. Описание из SKILL.md, длина описания и текста, есть ли скрипты |
| `trials.csv` | 6513 | Запуск любой из 22 конфигураций агента и модели. Режим с навыком или без, исход, награда, время, есть ли токены |
| `claude_code_runs.csv` | 432 | Запуск Claude Code с Opus 4.7, разобранный из лога модулем `skills_testbed.trajectory`. Время по частям, вызовы инструментов, ошибки окружения, открытые навыки |
| `failure_audit.csv` | 678 | Провал из аудита SkillsBench для трех моделей в OpenHands с вердиктом и категорией |
| `leaderboard.csv` | 24 | Строка итоговой таблицы SkillsBench v1.1 |
| `openhands_skill_use.csv` | 125 | Запуск OpenHands с Opus 4.8 и навыками. Открывал ли агент навык |
| `regression_checks.csv.gz` | 21000 | Смоделированная проверка PR из 12 сценариев. Вид проверки, настоящее падение доли успеха, решение бейзлайна, p-значения MVP после 3 и 5 запусков |
| `failure_predictions_haiku.csv` | 666 | Провал из аудита SkillsBench с логом. Вердикт аудита, ответы бейзлайна и MVP, цитата из лога и цена вызова LLM |
| `failure_llm_cache.jsonl` | 633 | Ответ LLM на один провал. Поле prompt_hash это хеш модели, системного промпта и сжатого лога |

Исход запуска бывает трех видов. `pass` значит проверка прошла полностью, `fail` значит не прошла, `timeout` значит агент не уложился в лимит времени. Задачи на документы и данные это категории office-white-collar и finance-economics.
