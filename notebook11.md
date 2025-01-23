---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 11. ML-Workflow: Modellbewertung und Auswahl

## 11.1 Kreuzvalidierung

In der Praxis ist es entscheidend, dass ein ML-Modell nicht nur gute Prognosen
für die Daten liefert, sondern auch für neue, unbekannte Daten zuverlässig
funktioniert. Durch das Aufteilen der Daten in Trainings- und Testdaten können
wir eine erste Einschätzung über die Verallgemeinerungsfähigkeit eines Modells
treffen. Dieser Ansatz weist jedoch einige Schwächen auf, die wir in diesem
Kapitel näher beleuchten. Im Anschluss lernen wir ein fortschrittlicheres
Verfahren kennen: die Kreuzvalidierung, die über die einfache Aufteilung in
Trainings- und Testdaten hinausgeht und eine robustere Bewertung der
Modellleistung ermöglicht.

### Lernziele Kapitel 11.1

- Sie sind in der Lage, das Konzept der **Kreuzvalidierung (Cross Validation)**
  verständlich zu erklären.
- Sie können die Vor- und Nachteile der Kreuzvalidierung aufzählen und bewerten.
- Sie können mit **KFold** einen Datensatz in verschiedene **Teilmengen
  (Folds)** aufteilen.
- Sie beherrschen die Durchführung einer Kreuzvalidierung mithilfe der Funktion
  **cross_validate()**.

### Idee der Kreuzvalidierung

Ein zentraler Schritt im ML-Workflow ist die Aufteilung der Daten in einen
Trainings- und einen Testdatensatz. Das Modell wird auf den Trainingsdaten
trainiert und anschließend auf den Testdaten bewertet. Diese Methode hat jedoch
auch Nachteile. Besonders bei kleinen Datensätzen ist es problematisch,
beispielsweise 25 % der Daten für den Test zurückhalten zu müssen, da dies die
Datenmenge für das Training reduziert. Zudem kann eine zufällige Aufteilung der
Daten zu unbalancierten Splits führen, die die Trainings- und Testergebnisse
verfälschen. Eine sinnvolle Alternative zu dieser simplen Aufteilung ist die
**Kreuzvalidierung** (engl. **Cross Validation**).

Bei der Kreuzvalidierung werden die Daten in mehrere **Teilmengen**, sogenannte
**Folds**, aufgeteilt. Beispielsweise können die Daten in fünf Folds unterteilt
werden. Das Modell wird dann fünfmal trainiert und getestet, wobei in jedem
Durchlauf eine andere Teilmenge als Testdaten verwendet wird. Im ersten
Durchlauf wird etwa Fold A für den Test zurückgehalten, während die Folds B, C,
D und E zum Training genutzt werden. Im zweiten Durchlauf wird Fold B als
Testdatensatz verwendet und die restlichen Folds dienen wieder dem Training.
Dieser Prozess wird so lange wiederholt, bis jeder Fold einmal als Testdaten
fungiert hat. Am Ende wird die Modellleistung (Score) als Durchschnitt der
Ergebnisse aus den fünf Durchläufen berechnet.

Es müssen jedoch nicht zwingend fünf Folds verwendet werden. Oftmals werden die
Daten in zehn Folds aufgeteilt, sodass 90 % der Daten zum Training und 10 % für
den Test verwendet werden. Ein weiterer Vorteil ist, dass jeder Datenpunkt im
Laufe der Kreuzvalidierung sowohl im Training als auch im Test berücksichtigt
wird, jedoch nie gleichzeitig. Dies verringert die Gefahr, dass unausgewogene
Daten zu verzerrten Testergebnissen führen, wie es bei einer zufälligen
Aufteilung passieren könnte.

Zusammengefasst bietet die Kreuzvalidierung mehrere Vorteile:

- **Effizientere Datennutzung**: Jeder Datenpunkt wird mindestens einmal als
  Testdatenpunkt verwendet, was besonders bei kleinen Datensätzen wichtig ist,
  da die Daten optimal ausgenutzt werden.
- **Stabilere Schätzung der Modellleistung**: Durch das wiederholte Training und
  Testen auf verschiedenen Daten erhöht sich die Robustheit der geschätzten
  Modellleistung (Score), da zufällige Verzerrungen durch unbalancierte Splits
  minimiert werden.

Ein Nachteil der Kreuzvalidierung ist der erhöhte Rechenaufwand, da das Modell
mehrfach trainiert und getestet wird.

Können wir also auf die Aufteilung in Trainings- und Testdaten verzichten? Nein,
denn für das Hyperparameter-Tuning ist der Split weiterhin notwendig. Mehr dazu
im nächsten Kapitel. Zunächst widmen wir uns der praktischen Umsetzung der
Kreuzvalidierung in Scikit-Learn.

### Kreuzvalidierung mit KFold

Um die Kreuzvalidierung in Scikit-Learn zu demonstrieren, generieren wir
zunächst einen künstlichen Datensatz. Mithilfe der Funktion `make_moons()`
erstellen wir 50 Datenpunkte und speichern sie in einem Pandas-DataFrame. Für
eine einfachere Visualisierung mit Plotly Express wandeln wir die Zielvariable
`'Wirkung'` von den Werten 0/1 in boolesche Werte (False/True) um.

```{code-cell}
import pandas as pd
import plotly.express as px
from sklearn.datasets import make_moons 

X_array, y_array = make_moons(noise = 0.5, n_samples=50, random_state=3)
daten = pd.DataFrame({
    'Merkmal 1': X_array[:,0],
    'Merkmal 2': X_array[:,1],
    'Wirkung': y_array
})
daten['Wirkung'] = daten['Wirkung'].astype('bool')

fig = px.scatter(daten, x = 'Merkmal 1', y = 'Merkmal 2', color='Wirkung',
    title='Künstliche Daten')
fig.show()
```

Als Nächstes laden wir die Klasse `KFold` aus dem Untermodul
`sklearn.model_selection`. Wir instanziieren ein KFold-Objekt mit dem Argument
`n_splits=5`, das die Daten in fünf Teilmengen (Folds) aufteilt. Tatsächlich ist
dies die Standardeinstellung, wie uns die [Dokumentation Scikit-Learn →
KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
zeigt. Das Argument könnte also weggelassen werden.

```{code-cell}
from sklearn.model_selection import KFold

kfold = KFold(n_splits = 5)
```

Im Hintergrund wurde ein Generator erzeugt, mit Hilfe dessen wir Daten in fünf
Teilmengen (Folds) aufteilen können. Dazu benutzen wir die Methode `.split()`
und übergeben ihr die Daten, die gesplittet werden sollen.

```{code-cell}
kfold.split(daten)
```

Zwar wurde hiermit die Aufteilung in fünf Teilmengen vollzogen, doch die
eigentlichen Trainings- und Testdaten wurden noch nicht gespeichert und
weiterverarbeitet. Mithilfe einer for-Schleife greifen wir in jedem Durchgang
auf die Trainings- und Testindizes zu, die die Methode `split()` als Tupel
zurückgibt. Das erste Element enthält die Indizes der Trainingsdaten, das zweite
die der Testdaten.

```{code-cell}
for (train_index, test_index) in kfold.split(daten):
  print(f'Index Trainingsdaten: {train_index}')
  print(f'Index Testdaten: {test_index}')
```

Die Aufteilung der Daten erfolgt hierbei sehr systematisch. Im ersten Durchgang
werden die Datenpunkte 0–9 als Testdaten verwendet, im zweiten Durchgang die
Punkte 10–19 und so weiter. Bei sortierten Daten kann dies ungünstig sein. Um
eine zufällige Aufteilung zu gewährleisten, können wir das Argument
`shuffle=True` verwenden, um die Daten vor dem Split zu mischen.

```{code-cell}
kfold = KFold(n_splits = 5, shuffle=True)

for (train_index, test_index) in kfold.split(daten):
  print(f'Index Trainingsdaten: {train_index}')
  print(f'Index Testdaten: {test_index}')
```

Nun verwenden wir diese fünf Aufteilungen, um einen Entscheidungsbaum zu
trainieren. Dabei begrenzen wir die Baumtiefe auf 3 und bewerten in jedem
Durchgang die Genauigkeit (Score) sowohl auf den Trainings- als auch auf den
Testdaten.

```{code-cell}
from sklearn.tree import DecisionTreeClassifier

modell = DecisionTreeClassifier(max_depth=3) 
kfold = KFold(n_splits = 5, shuffle=True, random_state=0)

for (train_index, test_index) in kfold.split(daten):
  X_train = daten.loc[train_index, ['Merkmal 1', 'Merkmal 2']]
  y_train = daten.loc[train_index, 'Wirkung']
  X_test = daten.loc[test_index, ['Merkmal 1', 'Merkmal 2']]
  y_test = daten.loc[test_index, 'Wirkung']
  
  modell.fit(X_train, y_train)
  score_train = modell.score(X_train, y_train)
  score_test = modell.score(X_test, y_test)

  print(f'Score Training: {score_train:.2f}, Score Test: {score_test:.2f}')
```

Die Scores auf den Trainingsdaten könnten den Eindruck erwecken, dass der
Entscheidungsbaum sehr gut funktioniert. Doch die Testdaten zeigen Schwankungen
zwischen 0.4 und 0.8. Hätten wir eine einfache Aufteilung in Trainings- und
Testdaten vorgenommen und zufällig den dritten Split erwischt, hätten wir
wahrscheinlich eine zu optimistische Einschätzung der Modellqualität getroffen.
Aus didaktischen Gründen verwenden wir das Argument `random_state=0`, um die
Ergebnisse mit dem Vorlesungsskript vergleichbar zu machen.

### Automatische Kreuzvalidierung mit cross_validate

Wie so oft bietet Scikit-Learn eine elegantere und einfachere Möglichkeit, die
Kreuzvalidierung (Cross Validation) durchzuführen, ohne manuell eine for-Schleife programmieren zu
müssen. Die Funktion `cross_validate()` übernimmt die Durchführung der
Kreuzvalidierung automatisch. Wir importieren sie aus dem Untermodul
`sklearn.model_selection` und teilen anschließend die Daten in Eingabedaten `X`
und Zielgröße `y` auf.

Die Funktion `cross_validate()` wird mit dem ML-Modell (hier einem
Entscheidungsbaum), den Eingabedaten `X` und der Zielgröße `y` aufgerufen.
Standardmäßig wird eine 5-fache Kreuzvalidierung ohne Mischen durchgeführt. Mit
dem optionalen Argument `cv=` kann jedoch auch ein benutzerdefinierter
Aufteilungsgenerator übergeben werden, wie zum Beispiel `KFold`. Das zusätzliche
Argument `return_train_score=True` sorgt dafür, dass auch die Trainingsscores in
jedem Durchlauf gespeichert werden. Der entsprechende Code sieht folgendermaßen
aus:

```{code-cell}
from sklearn.model_selection import cross_validate

X = daten[['Merkmal 1', 'Merkmal 2']]
y = daten['Wirkung']

cv_results = cross_validate(modell, X,y, cv=kfold, return_train_score=True)
```

Die Funktion `cross_validate()` gibt ein Dictionary zurück, das wie folgt
aufgebaut ist:

```{code-cell}
print(cv_results)
```

In diesem Dictionary sind zunächst die Rechenzeiten für das Training
(`'fit_time'`) und die Prognose (`'score_time'`) gespeichert. Danach folgen die
Scores der Testdaten (`'test_score'`). Falls das Argument
`return_train_score=True` gesetzt wurde, enthält das Dictionary auch die Scores
der Trainingsdaten (`'train_score'`). Die Scores können wir wie folgt anzeigen
lassen:

```{code-cell}
print(cv_results['test_score'])
print(cv_results['train_score'])
```

Weitere Details zu der Funktion `cross_validate()` finden Sie in der
[Dokumentation Scikit-Learn →
cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html).

### Zusammenfassung und Ausblick Kapitel 11.1

Die Kreuzvalidierung ist ein wichtiges Werkzeug, insbesondere wenn es um die
Feinjustierung der Hyperparameter geht, also das sogenannte
Hyperparameter-Tuning. Im nächsten Kapitel werden wir uns mit der Kombination
von Kreuzvalidierung (Cross Validation) und einer Gittersuche (Grid Search)
beschäftigen, um die optimalen Hyperparameter für ein Modell zu finden.

## 11.2 Gittersuche

Die Kreuzvalidierung wird selten isoliert verwendet. Sie ist jedoch ein
unverzichtbares Werkzeug, wenn es darum geht, die Hyperparameter eines Modells
zu optimieren. In diesem Kapitel vertiefen wir daher zunächst das Verständnis
der Kreuzvalidierung, bevor wir sie im Rahmen der Gittersuche anwenden.

### Lernziele Kapitel 11.2

- Sie verstehen, dass Daten für die Modellauswahl in Trainingsdaten,
  **Validierungsdaten** und Testdaten unterteilt werden.
- Sie sind in der Lage, Hyperparameter mittels Gittersuche und Kreuzvalidierung
  mithilfe von **GridSearchCV** zu optimieren.

### Kreuzvalidierung zur Modellauswahl

Im letzten Kapitel haben wir die Kreuzvalidierung eingeführt. Ihr Ziel ist es,
eine robustere Bewertung der Modellleistung zu ermöglichen. Besonders bei der
Beurteilung und der verbesserung der Verallgemeinerungsfähigkeit eines Modells
(Reduktion von Overfitting), ist die Kreuzvalidierung ein wertvolles Werkzeug.
In diesem Abschnitt nutzen wir die Kreuzvalidierung, um zwischen zwei Modellen
zu wählen.

Aus didaktischen Gründen verwenden wir weiterhin künstliche Daten, die mit der
Funktion `make_moons()` aus dem Modul `sklearn.datasets` erzeugt werden. Diese
speichern wir in einem Pandas DataFrame und visualisieren sie anschließend mit
Plotly Express.

```{code-cell}
import pandas as pd
import plotly.express as px
from sklearn.datasets import make_moons

X_array, y_array = make_moons(noise = 0.5, n_samples=100, random_state=3)
daten = pd.DataFrame({
    'Merkmal 1': X_array[:,0],
    'Merkmal 2': X_array[:,1],
    'Wirkung': y_array
})
daten['Wirkung'] = daten['Wirkung'].astype('bool')

fig = px.scatter(daten, x = 'Merkmal 1', y = 'Merkmal 2', color='Wirkung',
    title='Künstliche Daten')
fig.show()
```

Als nächstes trainieren wir einen Entscheidungsbaum. Da Entscheidungsbäume
häufig zur Überanpassung (Overfitting) neigen, entscheiden wir uns, die
Baumtiefe zu begrenzen. Aber welche Baumtiefe ist optimal? Die Baumtiefe ist ein
Hyperparameter, der vor dem Training des Modells festgelegt wird. Mithilfe der
Kreuzvalidierung können wir untersuchen, wie sich die Baumtiefe auf die
Modellqualität auswirkt. Wir testen die Baumtiefen 3, 4, 5 und 6 und geben die
Scores auf den Testdaten aus, wobei wir uns mit einer for-Schleife die Arbeit
erleichtern.

```{code-cell}
from sklearn.model_selection import cross_validate, KFold
from sklearn.tree import DecisionTreeClassifier

# Adaption der Daten
X = daten[['Merkmal 1', 'Merkmal 2']]
y = daten['Wirkung']

# Vorbereitung der Kreuzvalidierung mit 10 Splits
kfold = KFold(n_splits=10)

# wiederholte Kreuzvalidierung für Baumtiefe 3, 4, 5 und 6
for max_tiefe in [3, 4, 5, 6]:
    modell = DecisionTreeClassifier(max_depth=max_tiefe)
    cv_results = cross_validate(modell, X,y, cv=kfold)
    test_scores = cv_results['test_score']
    print(f'Testscores: {test_scores}')
```

Die Ausgabe von 10 Testscores ist jedoch unübersichtlich. Stattdessen berechnen
wir besser den Mittelwert (Mean) und die Standardabweichung (Standard Deviation)
der Scores. Dazu importieren wir `mean()` und `std()` aus dem NumPy-Modul und
passen die `print()`-Anweisung entsprechend an.

```{code-cell}
from numpy import mean, std

for max_tiefe in [3, 4, 5, 6]:
    modell = DecisionTreeClassifier(max_depth=max_tiefe)
    cv_results = cross_validate(modell, X,y, cv=kfold)
    test_scores = cv_results['test_score']
    print(f'Mittelwert Testscores: {mean(test_scores):.2f}, Standardabweichung: {std(test_scores):.2f}')
```

Das beste Ergebnis erzielen wir mit einem Entscheidungsbaum der Tiefe 3. Diesen
könnten wir nun als finales Modell wählen.

Es gibt jedoch ein Problem: Wir haben die Modellauswahl mit den Scores der
Testdaten begründet, wodurch diese in das Modelltraining eingeflossen sind. Daher
benötigen wir einen frischen Datensatz, um die Prognosequalität zu testen. Die
Lösung dafür ist `train_test_split()`.

Zuerst teilen wir die Daten in Trainings- und Testdaten. Dann verwenden wir die
Kreuzvalidierung auf den Trainingsdaten, um die Hyperparameter zu bewerten. Die
Kreuzvalidierung teilt die Trainingsdaten erneut in Trainings- und Testdaten
auf.  Damit diese »internen« Testdaten nicht mit den richtigen Testdaten
verwechselt werden, nennt man sie auch **Validierungsdaten**. Die Mittelwerte
der Scores speichern wir in einem Dictionary, um später das beste Modell zu
ermitteln. Schließlich trainieren wir das beste Modell auf allen Trainingsdaten
und bewerten es mit den Testdaten.

Das Hyperparameter-Tuning bzw. die Modellwahl mit Kreuzvalidierung funktioniert
komplett also wie folgt:

```{code-cell}
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

X = daten[['Merkmal 1', 'Merkmal 2']]
y = daten['Wirkung']

X_train, X_test, y_train, y_test = train_test_split(X,y)

kfold = KFold(n_splits=10)

mean_scores = {}
for max_tiefe in [3, 4, 5, 6]:
    modell = DecisionTreeClassifier(max_depth=max_tiefe)
    cv_results_modell = cross_validate(modell, X_train, y_train, cv=kfold)
    test_scores = cv_results_modell['test_score']
    mean_scores[max_tiefe] = mean(test_scores)
    print(f'Mittelwert Testscores: {mean(test_scores):.2f}, Standardabweichung: {std(test_scores):.2f}')

# Ermitteln der besten Baumtiefe (argmax o.ä. wäre einfacher)
tiefe = 3
score = mean_scores[3]
for t in [4,5,6]:
    if mean_scores[t] > score:
        tiefe = t
        score = mean_scores[t]
print(f'\nWähle Baumtiefe {tiefe} mit dem besten Score {score:.2f}.')

# Finale Modellauswahl, Training und Bewertung
finales_modell = DecisionTreeClassifier(max_depth=tiefe)
finales_modell.fit(X_train, y_train)
finaler_score = finales_modell.score(X_test, y_test)
print(f'Testscore finales Modell: {finaler_score:.2f}') 
```

Um die Hyperparameter zu optimieren und das beste Modell zu finden, haben wir
eine for-Schleife und manuelle Auswahl verwendet. Scikit-Learn bietet jedoch
eine einfachere Lösung, die wir im nächsten Abschnitt behandeln: die Gittersuche
mit Kreuzvalidierung **GridSearchCV**.

### Gittersuche mit Kreuzvalidierung: GridSearchCV

Die Gittersuche mit Kreuzvalidierung wird als **GridSearchCV** aus dem Modul
`sklearn.model_selection` importiert. Zunächst legen wir fest, welche Parameter
optimiert werden sollen und welche Werte dafür in Betracht kommen. Technisch
benötigen wir dafür ein Dictionary, in dem die Schlüssel die Parameternamen und
die Werte Listen der möglichen Einstellungen sind. In unserem Fall soll die
Baumtiefe `'max_depth'` des Entscheidungsbaums justiert werden. Wie zuvor in der
for-Schleife, untersuchen wir die Baumtiefen 3, 4, 5 und 6, die im folgenden
Dictionary `parameter_gitter` definiert werden.

```{code-cell}
from sklearn.model_selection import GridSearchCV

# Festlegung des Suchraumes
parameter_gitter = {'max_depth': [3, 4, 5, 6]}
```

Nun instanziieren wir ein neues `GridSearchCV`-Modell. Als erstes Argument
übergeben wir das eigentliche Modell, hier also den Entscheidungsbaum, und als
zweites das Dictionary mit den Hyperparametern. Das dritte Argument ist die
Methode zur Kreuzvalidierung. Weitere Details können Sie der [Dokumentation
Scikit-Learn →
GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)
entnehmen.

```{code-cell}
optimiertes_modell = GridSearchCV(DecisionTreeClassifier(), param_grid=parameter_gitter, cv=kfold)
```

Mit der Methode `.fit()` wird die Gittersuche samt Kreuzvalidierung
durchgeführt. Dabei werden systematisch alle Parameterkombinationen getestet,
und das optimierte Modell wird abschließend erneut auf den gesamten
Trainingsdaten trainiert.

```{code-cell}
optimiertes_modell.fit(X_train, y_train)
```

Mit der Methode `.score()` können wir die Modellgüte sowohl auf den Trainings-
als auch auf den Testdaten bewerten. Auch die Methode `.predict()` funktioniert
wie gewohnt.

```{code-cell}
opt_score_train = optimiertes_modell.score(X_train, y_train)
opt_score_test  = optimiertes_modell.score(X_test, y_test)

print(f'optimierter Entscheidungsbaum Score Trainingsdaten: {opt_score_train:.2f}')
print(f'optimierter Entscheidungsbaum Score Testdaten: {opt_score_test:.2f}')
```

Zusätzlich zu den Standardmethoden wie `.fit()`, `.predict()` und `.score()`
können wir mit dem Attribut `best_params_` herausfinden, welche
Hyperparameter-Kombination am besten abgeschnitten hat.

```{code-cell}
print(optimiertes_modell.best_params_)
```

In diesem Fall ergibt die Gittersuche, dass die optimale Baumtiefe 3 beträgt.

Warum sprechen wir von einer **Gittersuche**? Normalerweise wollen wir nicht nur
einen Hyperparameter optimieren, sondern mehrere gleichzeitig. Beispielsweise
könnten wir neben der Baumtiefe auch die minimale Anzahl an Datenpunkten pro
Blatt (`min_samples_leaf`) optimieren. Dies führt dazu, dass wir jede
Kombination von `max_depth` mit jedem Wert von `min_samples_leaf` testen. So
entsteht ein zweidimensionales Gitter, das die Gittersuche effizient durchläuft.
Wir müssen lediglich das Dictionary entsprechend erweitern. In diesem Beispiel
werden 4 Baumtiefen und 3 Werte für `min_samples_leaf` kombiniert, was zu
insgesamt 4 x 3 = 12 Hyperparameter-Kombinationen führt. Da wir 10-fache
Kreuzvalidierung verwenden, werden insgesamt 120 Modelle trainiert und bewertet.

```{code-cell}
parameter_gitter = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_leaf': [1, 2, 3]
}

optimiertes_modell = GridSearchCV(DecisionTreeClassifier(), param_grid=parameter_gitter, cv=kfold)
optimiertes_modell.fit(X_train, y_train)

opt_score_train = optimiertes_modell.score(X_train, y_train)
opt_score_test  = optimiertes_modell.score(X_test, y_test)

print(f'optimierter Entscheidungsbaum Score Trainingsdaten: {opt_score_train:.2f}')
print(f'optimierter Entscheidungsbaum Score Testdaten: {opt_score_test:.2f}')

print(optimiertes_modell.best_params_)
```

Auch wenn bei diesem einfachen Beispiel die Unterschiede zwischen den Modellen
gering sind und die Vorteile der Gittersuche mit Kreuzvalidierung nicht sofort
ersichtlich werden, ist diese Methode bei größeren Datensätzen und komplexeren
Modellen ein sehr wertvolles Werkzeug zur Modelloptimierung, bei der alle
möglichen Kombinationen von Hyperparametern systematisch getestet werden. Dies
kann jedoch sehr *rechenintensiv* sein, besonders wenn der Suchraum groß ist
oder komplexe Modelle verwendet werden. Daher unterstützt GridSearchCV die
*Parallelisierung* der Berechnungen, indem es mehrere Kerne verwendet, um die
Rechenzeit signifikant zu verkürzen, was besonders bei größeren Datensätzen von
Vorteil ist.

Eine Alternative zu GridSearchCV ist **RandomizedSearchCV**. Dieses Verfahren
testet eine zufällige Auswahl von Parametern testet und spart so Zeit, während
es dennoch gute Ergebnisse liefert. Mehr Details dazu finden Sie in der
[Dokumentation Scikit-Learn →
RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV).

```{dropdown} Video "GridSearchCV" von Normalized Nerd
<iframe width="560" height="315" src="https://www.youtube.com/embed/TvB_3jVIHhg?si=s2jDNKOmqBEcJcAd" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
```

### Zusammenfassung und Ausblick Kapitel 11.2

In diesem Kapitel haben wir erstmals systematisch Hyperparameter optimiert und
dabei die Gittersuche mit Kreuzvalidierung angewendet. Im nächsten Kapitel
lernen wir ein weiteres Werkzeug kennen, das nicht nur verschiedene Modelle,
sondern auch deren Hyperparameter optimiert und anschließend Modellvorschläge
basierend auf den besten Einstellungen macht.

## Übung

Der Datensatz Pinguine stammt von
[HuggingFace](https://huggingface.co/datasets/SIH/palmer-penguins). Der
Datensatz umfasst Daten von Pinguinen, insbesondere die Merkmale

- Art,
- Insel,
- Schnabellaenge und Schnabeltiefe in Millimetern
- Flossenlaenge in Millimetern,
- Koerpergewicht in Gramm,
- Geschlecht und
- Jahr der Geburt.

Laden Sie den Datensatz und führen Sie eine explorative Datenanalyse durch.
Legen Sie 10 % der Daten als Testdaten zurück. Trainieren Sie dann ML-Modelle
ggf. mit Gittersuche und wählen Sie das beste Modell aus. Welchen Score erreicht
Ihr Modell für die Testdaten?

In der folgenden Code-Zelle finden Sie import-Statements, die Sie bei Bedarf
auskommentieren können.

```{code-cell} ipython3
# import pandas as pd 

# from sklearn.model_selection import GridSearchCV, KFold, train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
```

### Überblick über die Daten

Welche Daten enthält der Datensatz? Wie viele Pinguine sind in der Tabelle
enthalten? Wie viele Merkmale werden dort beschrieben? Sind die Daten
vollständig?

```{code-cell} ipython3
#
```

### Datentypen

Welchen Datentyp haben die Merkmale? Welche Merkmale sind numerisch und welche sind kategorial?

```{code-cell} ipython3
#
```

### Fehlende Einträge

In welcher Spalte fehlen am meisten Einträge? Filtern Sie den Datensatz nach den
fehlenden Einträgen und geben Sie eine Liste mit den Indizes (Zeilennummern)
aus, wo Einträge fehlen. Löschen Sie anschließend diese Zeilen aus dem
Datensatz. Sind jetzt alle Einträge gültig?

```{code-cell} ipython3
#
```

### Analyse numerische Daten

Erstellen Sie eine Übersicht der statistischen Merkmale für die numerischen
Daten. Visualisieren Sie anschließend die statistischen Merkmale mit Boxplots.
Verwenden Sie ein Diagramm für die Merkmale, die in Millimetern gemessen werden
und ein Diagramm für das Körpergewicht. Interpretieren Sie die statistischen
Merkmale. Gibt es Ausreißer? Sind die Werte plausibel?

```{code-cell} ipython3
#
```

### Analyse der kategorialen Werte

Untersuchen Sie die kategorialen Daten. Sind es wirklich kategoriale Daten?
Prüfen Sie für jedes kategoriale Merkmal die Einzigartigkeit der auftretenden
Werte und erstellen Sie ein Balkendiagramm mit den Häufigkeiten.

Kommen alle Pinguin-Arten auf allen Inseln vor?

```{code-cell} ipython3
#
```

### ML-Modell

Im Folgenden soll die Art der Pinguine anhand der numerischen Merkmale
Schnabellaenge_mm, Schnabeltiefe_mm, Flossenlaenge_mm und Koerpergewicht_g
klassifiziert werden.

Trainieren Sie nun drei ML-Modelle:

- Entscheidungsbaum (Decision Tree),
- Random Forests und
- SVM.

Führen Sie dazu vorab einen Split in Trainings- und Testdaten durch. Verwenden
Sie Kreuzvalidierung und/oder Gittersuche, um die Hyperparameter zu justieren.
Für welches Modell würden Sie sich entscheiden? Begründen Sie Ihre Wahl.

```{code-cell} ipython3
#
```
