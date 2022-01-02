1. Создать и активировать окружение:
```
python -m venv mcenv
mcenv\\Scripts\\activate
pip install -r requirements.txt
```

2. Установить кернел для юпитера (в окружении):
```
mcenv\\Scripts\\activate
python -m ipykernel install --name mcenv
```

3. Запустить юпитер (затем перейти по ссылке и открыть \*.ipynb):
```
jupyter notebook
```
