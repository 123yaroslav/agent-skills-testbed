# Стенд для тестирования навыков LLM-агентов

[![CI](https://github.com/123yaroslav/agent-skills-testbed/actions/workflows/ci.yml/badge.svg)](https://github.com/123yaroslav/agent-skills-testbed/actions/workflows/ci.yml)

Стенд прогоняет агента на задачах с документами и данными с навыком (Agent Skill) и без него. Он показывает, помогает ли навык, берет ли агент нужный навык из библиотеки, почему прогон провалился и не стало ли хуже после обновления. Навыки сейчас почти не тестируют, а их эффект зависит от модели и бывает отрицательным.

Готовы [дизайн-документ](docs/ml_system_design_doc.md) и модуль статистики для его расчетов. Проект курса ML System Design, ИТМО, 2026.

## Запуск

Нужны Python 3.12 и [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pre-commit install
uv run pre-commit run --all-files       # ruff, mypy, gitleaks
uv run pytest
uv run python -m skills_testbed.power   # таблица чувствительности
```

Линтер и форматер ruff, типы проверяет mypy в строгом режиме, ключи API ловит gitleaks. Все это запускают pre-commit перед коммитом и CI на каждый PR.

| Путь | Что там |
|---|---|
| `docs/` | Дизайн-документ |
| `src/skills_testbed/` | Код стенда |
| `scripts/skillsbench_variance.py` | Разброс исходов по данным SkillsBench |
| `tests/` | Тесты |

Правила веток и PR в [CONTRIBUTING.md](CONTRIBUTING.md).

Автор Ярослав Рогоза ([123yaroslav](https://github.com/123yaroslav)). Ревьюеры ДЗ Анна Шумилова ([ann04ka](https://github.com/ann04ka)) и Иван Александров ([Ivanich-spb](https://github.com/Ivanich-spb)). Лицензия [MIT](LICENSE).
