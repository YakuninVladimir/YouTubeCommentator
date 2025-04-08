![image](https://github.com/user-attachments/assets/c8baa0ff-a099-473e-9848-566ee87d3352)## Сервис для генерации комментариев по видео на YouTube

В качестве языковой модели используется finetune gpt2-large, итоговую версию модели можно найти по ссылке [tuned GPT](https://huggingface.co/YakuninVla/youtube_finetuned_gpt2), процесс обработки данных и обучения есть в ноутбуке

### Возможности
Чтобы сгенерировать комментарий нужно указать ссылку на видео, также дополнительно можно настроить тональность комментария (позитивный или негативный), температуру модели и число слов в сгенерированном комментарии

### Зависимости:
1) transformers
2) pandas
3) torch
4) streamlit
5) dotenv
6) google-api-python-client

Для запуска локально потребуется API-KEY для youtube, который надо будет указать в файле .env

запуск осуществляется командой python -m streamlit run app.py

Версия в облаке доступна по [ссылке](https://huggingface.co/spaces/YakuninVla/YouTube_commentator)
