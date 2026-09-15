# Kanban + GitHub (методологія практики)

Розробка ведеться інкрементально за **Kanban**: обмеження WIP, візуалізація потоку,
безперервна поставка в `main`.

## Дошка (рекомендовані колонки)

Створіть [GitHub Project](https://docs.github.com/en/issues/planning-and-tracking-with-projects)
у репозиторії `c0n666/prediction_market_engine` з колонками:

| To Do | In Progress | Review | Done |
|-------|-------------|--------|------|
| Нові задачі | Активна робота (WIP ≤ 2) | PR / перевірка | Змерджено |

## Типові картки (Issues)

1. REST ingestion + snapshots  
2. Breeden-Litzenberger fair price  
3. Isolation Forest anomalies  
4. BL vs ML comparison metrics  
5. Time-series backtester  
6. Streamlit dashboard / Plotly charts  
7. Practice documentation  

Шаблон issue: `.github/ISSUE_TEMPLATE/kanban_task.md`

## Правила Git

- Гілка роботи: `main` (навчальний проєкт) або `feature/<name>` для великих змін  
- Коміти: імперативний короткий опис («Add…», «Fix…», «Compare…»)  
- Не комітити `.env`, ключі, великі бінарні артефакти без потреби  
- Після завершення картки — перенесення в **Done** на Project board  

## Докази для звіту

- Посилання на репозиторій і скрін GitHub Project  
- Історія комітів (`git log --oneline`)  
- Issues з мітками `kanban`, `practice`
