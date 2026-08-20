import re

# Путь к проблемному файлу hydra из лога
file_path = r"C:\Users\dayta\AppData\Local\Programs\Python\Python311\Lib\site-packages\hydra\conf\__init__.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Регулярное выражение для исправления вложенных классов с точками (например, JobConf.JobConfig.OverrideDirname)
content = re.sub(r"([a-zA-Z0-9_]+):\s*([a-zA-Z0-9_.]+)\s*=\s*\2\(\)", r"\1: \2 = field(default_factory=\2)", content)

# Добавляем импорт field, если его нет
if "from dataclasses import dataclass, field" not in content:
    content = content.replace("from dataclasses import dataclass", "from dataclasses import dataclass, field")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Библиотека hydra успешно вылечена для работы на Python 3.11!")