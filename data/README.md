# Данные для EDA

Здесь лежат таблицы, которые читает [`notebooks/eda.ipynb`](../notebooks/eda.ipynb). Это выжимка из открытых данных SkillsBench. Текст логов в репозиторий не попадает, только посчитанные из него признаки.

Пересобрать таблицы можно так.

```bash
uv sync --group eda
uv run python scripts/fetch_eda_data.py    # около 60 МБ в data/raw, эта папка не в git
uv run python scripts/build_eda_tables.py  # таблицы в data/processed
```

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

Исход запуска бывает трех видов. `pass` значит проверка прошла полностью, `fail` значит не прошла, `timeout` значит агент не уложился в лимит времени. Задачи на документы и данные это категории office-white-collar и finance-economics.
