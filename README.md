# очередной проект

Я постараюсь быть краток, потому что у меня очень мало времени, я не так много всего пробовал, я как обычно один, мне за это даже не платят, в общем, настроение на нуле.
Я знаю, что код не структурирован, там сплошная каша, хотя я вынес основные смысловые части в отдельные скрипты. Чтобы лучше понять, что я вообще делал, лучше просто заглянуть в ноутбук, там всё более-менее
написано, только некоторые части, особенно в плане выводов устарели

Я брал не совсем такую же задачу, потому что определять аспекты на уровне символов это как-то сложно для меня. В принципе задача то та же самая, только работать с последовательностями символов, но тогда отзывы придётся резать скорее всего - слишком много памяти. Так что я делал разметку по токенам. И учил сетки, разумеется, потому что я могу, умею, практикую. Я в курсе, что кучу задача в нлп и не только можно решить без них, на недавнем хакатоне например моё решение через opencv было топ2, там был детерминированный алгоритм, так что да, я в курсе, но мне это не интересно.

Я никакие данные дополнительно не качад. Я тоже прекрасно понимаю, что дай моей модели любой ood, она сломается, ну да, а что поделать. Так что я фактически сильно переобучился под те немногочисленные данные, что были нам даны и получил более-менее качество. Учить 3 сетки не интересно, поэтому я учил одну но с 3 разными выходами. Чекпойнты есть в вандб, ссылка есть в ноутубке `bert.ipynb`.

Самая большая проблема была с данными, потому что нужно было их как-то выровнять. Я это сделал, но не идеально, где-то процентов 10 инофрмации об аспектах я потерял. И из-за этого дебажить инференс становится очень сложно, потому чтш там то разметки нет вообще никакой. Последний кусочек тетрадки это фактически всё, что нужно, чтобы инференс запустить, там 2 строчки. Я бы мог сделать скрипт, но это слишком запарно, плюс у меня на компе он скорее всегоо не запустится. Если что, имеет смысл, подкрутить batch_size, если вдруг на куду не влезет

Вроде бы бейзлайн я обошёл, но это на моей валидации, где я могу точно отследить, какие токены я добавил. Не знаю, успею я отдебажить инференс или нет, но там возникает проблема с псевдотокенами после паддинга. Паддинг это в принципе проблема, потому что после него почему-то токенов становится меньше, хотя у меня ни одна последовательностб не срезается, это очень странно. Но вот да, это очередной душный момент с выраниванием этого безобразия

Если судить по самим данным, то всё получается вполне адекватно, местами даже лучше, потому что такое ощущение, что в разметке есть не все аспекты, а может быть и нет, кто знает, но я наверное не успеваю вставить пример

Пока на этом наверное всё, если я могу доделать, я могу сделать это ночью после работы, мне нужно буквально часа 4, чтобы всё вылизать. Надеюсь хоть всё нужное успею в репу залить, а то там тоже много, плюс модельки по 2Гб

### Что тут лежит то

1. `bert.ipynb` - тетрадке, где я на самом дел евообще всё учил. Там вся предобработка (если что делается в инференсе тоже, не надо запускать), безлайн для токенов, берт для токенов, сравнение, устаревшие выводы, есть даже комментарии
2. `baseline.ipynb` - это исходная тетрадка, но я её взял, чтобы был настоящий бейзлайн
3. `evaluate.ipynb` - если честно я не помню, по-моему я ничего в нём не делал
4. `bert.py` - скрипт с моделькой берта, токенизатора, функций для трейна, оценки качества. Очень надеюсь, что он работает, потому что вообще-то говоря я не проверял. Ну трейн должен
5. `baseline.py` - тут всякие функции для бейзлайна. Изначально они все были в тетрадке, я их вынес
6. `inference.py` - тут всё, что нужно для самого инференса - но он только для берта, и всё, что нужно, чтобы качество оценивать. Я пользовался тем, что уже было нам дано, хотя не знаю, насколько там адекватные оценки, и патался побить именно их. Там почти везде победа, кроме реколла по-моему
7. `common_utils.py` - если честно я забыл, что тут, но видимо что-то, что я везде таскаю

Комментов в скриптах нет, но они как правило склеены из того, что уже было и пояснялось в тетрадках, я не успеваю, извините -_-

UPD: В определении всего сентимента где-то опять косяк с выравниванием, боже, как мне это надоело. Остальное должно работать
UPD2: Если честно, уже закрадываются сомнения, что там всё нормально с подсчётом качества. У меня уже один раз было такое, что я поменял местами метки ['food', 'service'] и так далее, при этом сентимент был такой же, а качество сломалось. Что-то тут нечисто
UPD3: Надо было вообще самому валидацию и качество писать, это ведь самое важное, хотя я и написал
