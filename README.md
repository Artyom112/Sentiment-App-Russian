# Sentiment-App-Russian
Русская версия приложения, определяющего тональность отзывов на смартфоны. Предсказания осуществляются с помощью модели tfidf, затренированной на отзывах, полученных с сайта goods.ru. Во избежании ситуации с несбалансированными классами (всего доступно 8786 позитивных отзывов и 1045 негативных), применен positive sampling.

Для запуска приложения, скопируйте репозиторий в свою директиву, используя команду:
```
https://github.com/Artyom112/Sentiment-App-Russian.git
```

После установки необходимых библиотек, перейдите в файл app.py и запустите:
```
if __name__=="__main__":
    app.run(port=5001, debug=True)
```

Затем нажмите на появившуюся ссылку в консоли, это откроет сайт в браузере на локальном сервере.
