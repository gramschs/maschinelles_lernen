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

# 9. Ensemble-Methoden (Random Forests und XGBoost)

## 9.1 Stacking, Bagging und Boosting

Eins, zwei, viele ... im Bereich des maschinellen Lernens sind Ensemble-Methoden
eine leistungsstarke Technik zur Verbesserung der Modellgenauigkeit und
Robustheit. Diese Methoden kombinieren mehrere Modelle, um die Gesamtleistung zu
steigern, indem sie die individuellen Stärken der Modelle nutzen und deren
Schwächen ausgleichen. In diesem Kapitel werden wir die grundlegenden
Konzepte und Unterschiede zwischen diesen drei Methoden erläutern, um ein
besseres Verständnis ihrer Funktionsweise und Anwendungen zu vermitteln.

### Lernziele Kapitel 9.1

* Sie können in eigenen Worten erklären, was **Ensemble-Methoden** sind.
* Sie können die drei Ensemble-Methoden
  * **Stacking**,
  * **Bagging** und
  * **Boosting**

  mit Hilfe einer Skizze erklären.

### Ensemble-Methoden

Der Begriff »Ensemble« wird im Allgemeinen eher mit Musik und Kunst in
Verbindung gebracht als mit Informatik. In der Musik bezeichnet ein Ensemble
eine kleine Gruppe von Musikern, die entweder das gleiche Instrument spielen
oder verschiedene Instrumente kombinieren. Im Theater bezeichnet man eine Gruppe
von Schauspielern ebenfalls als Ensemble, und in der Architektur beschreibt der
Begriff eine Gruppe von Gebäuden, die in einem besonderen Zusammenhang
zueinander stehen.

Auch im Bereich des maschinellen Lernens hat sich der Begriff Ensemble
etabliert. Mit **Ensemble-Methoden** (Ensemble Learning) wird eine Gruppe von
maschinellen Modellen bezeichnet, die zusammen eine Prognose treffen sollen.
Ähnlich wie bei Musik-Ensembles können beim **Ensemble Learning** entweder
identische Modelle oder verschiedene Modelle kombiniert werden. Diese Modelle
können entweder gleichzeitig eine Prognose treffen, die dann kombiniert wird,
oder nacheinander verwendet werden, wobei ein Modell auf den Ergebnissen eines
anderen aufbaut. Je nach Vorgehensweise unterscheidet man im maschinellen Lernen
zwischen **Stacking**, **Bagging** und **Boosting**.

In dieser Vorlesung konzentrieren wir uns auf Bagging und Boosting mit ihren
bekanntesten Vertretern, den Random Forests und XGBoost. Das Konzept des
Stackings wird hier nur kurz ohne weitere Details vorgestellt. Eine allgemeine
Einführung in Ensemble-Methoden mit Scikit-Learn findet sich in der
[Dokumentation Scikit-Learn →
Ensemble](https://scikit-learn.org/stable/modules/ensemble.html).

![Stacking](https://gramschs.github.io/book_ml4ing/_images/concept_stacking.svg)

Stacking bedeutet auf Deutsch »Stapeln«, es werden sozusagen verschiedene
ML-Modelle gestapelt. In einem ersten Schritt werden mehrere ML-Modelle
unabhängig voneinander auf den Trainingsdaten trainiert. Jedes dieser Modelle
liefert eine Prognose, die dann auf verschiedene Arten miteinander kombiniert
werden können. Bei Klassifikationsaufgaben ist **Voting**, also die Wahl durch
**Mehrheitsentscheidung**, eine beliebte Methode, um die Einzelprognosen zu
kombinieren. Wurden beispielsweie für das Stacking drei ML-Modellen gewählt, die
jeweils ja oder nein prognostizieren, dann wird für die finale Prognose das
Ergebnis genommen, das die Mehrheit der einzelnen Modelle vorausgesagt hat.
Scikit-Learn bietet dafür einen Voting Classifier an, siehe [Dokumentation
Scikit-Learn → Voting
Classifier](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier).

Bei Regressionsaufgaben werden die einzelnen Prognosen häufig gemittelt. Dabei
kann entweder der übliche arithmetische Mittelwert verwendet werden oder ein
**gewichteter Mittelwert**, was als  **Weighted Averaging** bezeichnet wird.
Nichtsdestotrotz wird die Mittelwertbildung bei Regressionsaufgaben von
Scikit-Learn ebenfalls als Voting bezeichnet, siehe [Dokumentation Scikit-Learn
→ Voting
Regressor](https://scikit-learn.org/stable/modules/ensemble.html#voting-regressor).

Eine alternative Kombinationsmethode ist die Verwendung eines weiteren
ML-Modells. In diesem Fall werden die Modelle, die die einzelnen Prognosen
liefern, als Basismodelle bezeichnet. In der ML-Community ist auch der
Fachbegriff **Weak Learner**, also schwache Lerner, für diese Basismodelle
gebräuchlich. Die Prognosen der Basismodelle dienen dann als Trainingsdaten für
ein weiteres ML-Modell, das als **Meta-Modell** bezeichnet wird. Diese
Ensemble-Methode wird **Stacking** genannt. Weitere Informationen liefert die
[Scikit-Learn Dokumentation → Stacked
Generalization](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization).

Stacking bietet viele Vorteile. Der wichtigste Vorteil ist, dass die
Prognosefähigkeit des Gesamtmodells in der Regel deutlich besser ist als die der
einzelnen Basismodelle. Die Stärken der Basismodelle werden kombiniert und die
Schwächen ausgeglichen. Allerdings erfordert Stacking sehr viel Feinarbeit. Auch
steigt die Trainingszeit für das Gesamtmodell, selbst wenn die Basismodelle bei
genügend Rechenleistung parallel trainiert werden können. Aus diesem Grund
werden wir in dieser Vorlesung kein Stacking verwenden.

### Bagging

![Bagging](https://gramschs.github.io/book_ml4ing/_images/concept_bagging.svg)

Bagging ist eine Ensemble-Methode, ähnlich wie Stacking. Im Gegensatz zum
Stacking wird beim Bagging jedoch dasselbe Modell für die Einzelprognosen
verwendet. Die Unterschiede in den Einzelprognosen entstehen dadurch, dass für
das Training der einzelnen Modelle unterschiedliche Daten verwendet werden.

Im ersten Schritt werden zufällige Datenpunkte aus den Trainingsdaten ausgewählt
und in einen neuen Datensatz, „Stichprobe 1“, aufgenommen. Nachdem ein
Datenpunkt ausgewählt wurde, kehrt er in die ursprüngliche Menge der
Trainingsdaten zurück und kann erneut ausgewählt werden. Dieser Prozess wird in
der Mathematik als **Ziehen mit Zurücklegen** bezeichnet, auf Englisch
**Bootstrapping**. Durch Bootstrapping werden dann noch weitere Stichproben
gebildet.

Im zweiten Schritt wird ein ML-Modell gewählt und für jede Bootstrap-Stichprobe
trainiert. Da die Stichproben unterschiedliche Trainingsdaten enthalten,
entstehen unterschiedlich trainierte Modelle, die für neue Daten verschiedene
Einzelprognosen liefern. Diese Einzelprognosen werden kombiniert bzw. nach
festgelegten Regeln zu einer finalen Prognose zusammengefasst. In der Statistik
wird die Zusammenfassung von Daten als Aggregation bezeichnet. Auf Englisch
heißt der Vorgang des Zusammenfassens **Aggregating**.

Die beiden wesentlichen Schritte der Bagging-Methode sind also **B**ootstrapping
und **Agg**regat**ing**, was zu der Abkürzung Bagging geführt hat. Scikit-Learn
bietet sowohl für Klassifikations- als auch für Regressionsaufgaben eine
allgemeine Implementierung der Bagging-Methode an (siehe [Dokumentation
Scikit-Learn →
Bagging](https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator)).
Die bekannteste Bagging-Methode ist das Verfahren **Random Forests**, bei dem
Entscheidungsbäume (Decision Trees) auf unterschiedlichen Stichproben trainiert
und aggregiert werden. Random Forests werden wir im nächsten Kapitel
detaillierter betrachten. Vorab beschäftigen wir uns noch mit dem Konzept der
Boosting-Methoden.

### Boosting

![Boosting](https://gramschs.github.io/book_ml4ing/_images/concept_boosting.svg)

Das englische Verb „to boost sth.“ hat viele Bedeutungen. Insbesondere wird es
im Deutschen mit „etwas verstärken“ übersetzt. Im Kontext des maschinellen
Lernens bezeichnet **Boosting** eine Ensemble-Methode, bei der mehrere ML-Modelle
hintereinander geschaltet werden, um die Genauigkeit der Prognose zu verstärken.
Die Idee des Boosting besteht darin, dass jedes Modell die Fehler des
Vorgängermodells reduziert. Es gibt mehrere Varianten zur Fehlerreduktion, aus
denen sich unterschiedliche Boosting-Methoden ableiten. Die wichtigsten
Varianten sind:

* Adaboost,
* Gradient Boosting und
* Stochastic Gradient Boosting.

Beim **Adaboost**-Verfahren wird im ersten Schritt ein Modell (z.B. ein
Entscheidungsbaum) auf den Trainingsdaten trainiert. Anschließend werden die
Prognosen dieses Modells mit den tatsächlichen Werten verglichen. Im zweiten
Schritt wird ein neuer Datensatz erstellt, wobei die Datenpunkte, die falsch
prognostiziert wurden, ein größeres Gewicht erhalten. Nun wird erneut ein Modell
trainiert; und dessen Prognosen werden wieder mit den echten Werten verglichen.
Dieser Vorgang wird mehrfach wiederholt. Das Training der Modelle erfolgt
sequentiell, da jedes Vorgängermodell die neue Gewichtung der Trainingsdaten
liefert. Am Ende werden alle Einzelprognosen gewichtet zu einer finalen Prognose
kombiniert. Weitere Details finden sich in der [Dokumentation Scikit-Learn →
Adaboost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost).

Beim **Gradient Boosting** wird ebenfalls ein sequentieller Ansatz verfolgt,
aber der Fokus liegt auf der Minimierung der Fehler. Im ersten Schritt wird ein
ML-Modell (häufig ein Entscheidungsbaum) trainiert. Danach wird für jeden
Datenpunkt der Fehler des Modells, das sogenannte **Residuum**, berechnet, indem
die Differenzen zwischen dem tatsächlichen Wert und der Prognosen bestimmt wird.
Im nächsten Schritt wird ein neues Modell trainiert, das darauf abzielt, diese
Residuen vorherzusagen. Dieses neue Modell wird dann zu dem vorherigen Modell
hinzugefügt, um die Gesamtprognose zu verbessern. Dieser Prozess wird
wiederholt, wobei in jeder Iteration ein neues Modell trainiert wird, das die
Fehler der bisherigen Modelle reduziert (mit Hilfe einer Verlustfunktion und
eines Gradientenverfahrens). Am Ende ergibt sich eine starke Vorhersage, indem
alle Modelle kombiniert werden. Da sehr häufig Entscheidungsbäume als Modell
gewählt werden, bietet Scikit-Learn eine Implementierung der sogenannten
**Gradient Boosted Decision Trees** an, siehe [Dokumentation Scikit-Learn →
Gradient-boosted
trees](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosted-trees).

**Stochastic Gradient Boosting** ist eine Erweiterung des Gradient Boosting, bei
der zusätzlich Stochastik eingeführt wird. Hierbei wird in jedem Schritt nur
eine zufällige Stichprobe der Trainingsdaten verwendet, um ein Modell zu
trainieren. Der Trainingsprozess ähnelt dem von Gradient Boosting, wobei in
jeder Runde ein neues Modell trainiert wird, das die Fehler der vorherigen
Modelle korrigiert. Durch die zufällige Auswahl der Trainingsdaten in jeder
Iteration wird eine höhere Robustheit gegenüber Overfitting (Überanpassung)
erreicht. Stochastic Gradient Boosting wird nicht direkt von Scikit-Learn
unterstützt. Eine sehr bekannte Implmentierung davon ist XGBoost (siehe
[https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/en/stable/)),
die wir in einem der nächsten Kapitel noch näher betrachten werden.

### Zusammenfassung und Ausblick Kapitel 9.1

In diesem Kapitel haben Sie die drei Konzepte Stacking, Bagging und Boosting
eher theoretisch kennengelernt. Alle drei Methoden sind Ensemble-Methoden, bei
denen mehrere ML-Modelle parallel oder sequentiell kombiniert werden. Obwohl
diese Ensemble-Methoden allgemein für verschiedene ML-Modelle eingesetzt werden
können, haben sich vor allem Random Forests (Bagging für Entscheidungsbäume) und
Stochastic Gradient Boosting als besonders effektiv erwiesen. Letztere sind
nicht in Scikit-Learn implementiert, sondern werden durch eine eigene Bibliothek
namens XGBoost bereitgestellt. In den nächsten beiden Kapiteln werden wir beide
auch mit praktischen Beispielen vertiefen.

## 9.2 Random Forests

Im letzten Kapitel haben wir verschiedene Ensemble-Methoden in der Theorie
kennengelernt: Stacking, Bagging und Boosting. Für die beiden letzteren
Ensemble-Methoden werden besonders häufig Entscheidungsbäume (Decision Trees)
eingesetzt. Daher betrachten wir in diesem Kapitel die praktische Umsetzung von
Bagging mit Entscheidungsbäumen, die sogenannten Random Forests.

### Lernziele Kapitel 9.2

* Sie können das ML-Modell **Random Forest** in der Praxis anwenden.
* Sie können mit Hilfe der **Feature Importance** bewerten, wie groß der
  Einfluss eines Merkmals auf die Prognosegenauigkeit des Random Forests ist.

### Random Forests mit Scikit-Learn

Entscheidungsbäume (Decision Trees) haben wir bereits betrachtet. Sie sind
aufgrund ihrer Einfachheit und vor allem aufgrund ihrer Interpretierbarkeit sehr
beliebt. Allerdings ist ihre Tendenz zum Overfitting problematisch. Daher
kombinieren wir die Ensemble-Methode Bagging mit Entscheidungsbäumen (Decision
Trees). Indem aus den Trainingsdaten zufällig kleinere Bootstrap-Stichproben
ausgewählt werden, erhalten wir unterschiedliche Entscheidungsbäume (Decision
Trees). Zusätzlich wird beim Training der Entscheidungsbäume nicht mit allen
Merkmalen (Features) trainiert, sondern auch hier wählen wir die Merkmale
zufällig aus. Durch diese zwei Maßnahmen wird die Anpassung der
Entscheidungsbäume an die Trainingsdaten (Overfitting) reduziert.

Um den Random Forest von Scikit-Learn praktisch auszuprobieren, erzeugen wir
künstliche Daten. Dazu verwenden wir die Funktion `make_moons` von Scikit-Learn,
die Zufallszahlen generiert und interpretieren die Zufallszahlen als
Kilometerstände und Preise von Autos bei einer fiktiven Verkaufsaktion.
Zusätzlich lassen wir zufällig Nullen und Einsen erzeugen, die wir als
»verkauft« oder »nicht verkauft« interpretieren.

```{code-cell} ipython3
import pandas as pd 
from sklearn.datasets import make_moons

# Erzeugung künstlicher Daten
X_array, y_array = make_moons(n_samples=120, random_state=0, noise=0.3)

daten = pd.DataFrame({
    'Kilometerstand [km]': 10000 * (X_array[:,0] + 2),
    'Preis [EUR]': 5000 * (X_array[:,1] + 2),
    'verkauft': y_array,
    })

# Adaption der Daten
X = daten[['Kilometerstand [km]', 'Preis [EUR]']].values
y = daten['verkauft'].values
```

Diesmal werden in dem Autohaus 120 Autos zum Verkauf angeboten (siehe Option
`n_samples=120`). Nach Aktionsende werden die Merkmale Kilometerstand und Preis
tabellarisch erfasst und notiert, ob das Auto verkauft wurde (True bzw. 1) oder
nicht verkauft wurde (False bzw. 0). Wir visualisieren die Daten.

```{code-cell} ipython3
import plotly.express as px
# plot artificial data
fig = px.scatter(daten, x = 'Kilometerstand [km]', y = 'Preis [EUR]', color=daten['verkauft'].astype(bool),
        title='Künstliche Daten: Verkaufsaktion Autohaus',
        labels = {'x': 'Kilometerstand [km]', 'y': 'Preis [EUR]', 'color': 'verkauft'})
fig.show()
```

Nachdem wir die Vorbereitungen für die Daten abgeschlossen haben, können wir
Scikit-Learn einen Random Forest trainieren lassen. Dazu importieren wir den
Algorithmus aus dem Modul `ensemble`. Da der Random Forest ein Ensemble von
Entscheidungsbäumen (Decision Trees) ist, haben wir nun die Möglichkeit, die
Anzahl der Entscheidungsbäume festzulegen. Voreingestellt sind 100
Entscheidungsbäume. Aus didaktischen Gründen reduzieren wir diese Anzahl auf
vier und setzen das Argument `n_estimators=` auf `4`. Ebenfalls aus didaktischen
Gründen fixieren wir die Zufallszahlen, mit Hilfer derer das Bootstrapping und
die Auswahl der Merkmale (Features) umgesetzt wird, mit `random_state=0`. In
einem echten Projekt würden wir das unterlassen. Zuletzt führen wir das Training
mit der `.fit()`-Methode durch. Weitere Details finden Sie unter [Scikit-Learn
Dokumentation →
RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

```{code-cell} ipython3
from sklearn.ensemble import RandomForestClassifier

model_random_forest = RandomForestClassifier(n_estimators=4, random_state=0)
model_random_forest.fit(X,y)
```

Als nächstes lassen wir den Random Forest für jeden Punkte des Gebiets
prognostizieren, ob ein Auto mit diesem Kilometerstand und diese Preis
verkaufbar wäre oder nicht.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay

my_colormap = ListedColormap(['#EF553B33', '#636EFA33'])
fig = DecisionBoundaryDisplay.from_estimator(model_random_forest, X,  cmap=my_colormap)
fig.ax_.scatter(X[:,0], X[:,1], c=y, cmap=my_colormap)
fig.ax_.set_xlabel('Kilometerstand [km]');
fig.ax_.set_ylabel('Preis [EUR]');
fig.ax_.set_title('Random Forest: Entscheidungsgrenzen');
```

Möchte man die vier Entscheidungsbäume (Decision Trees) analysieren, aus denen
der Random Forest kombiniert wurde, kann man mit dem Attribut `estimators_`
darauf zugreifen. Wir lassen uns jetzt die Entscheidungsgrenzen einzeln für
jeden Entscheidungsbaum anzeigen.

```{code-cell} ipython3
my_colormap = ListedColormap(['#EF553B33', '#636EFA33'])
for (nummer, baum) in zip(range(4), model_random_forest.estimators_):
    fig = DecisionBoundaryDisplay.from_estimator(baum, X,  cmap=my_colormap)
    fig.ax_.scatter(X[:,0], X[:,1], c=y, cmap=my_colormap)
    fig.ax_.set_xlabel('Kilometerstand [km]');
    fig.ax_.set_ylabel('Preis [EUR]');
    fig.ax_.set_title(f'Entscheidungsbaum {nummer+1}');
```

### Feature Importance

Der Random Forest reduziert das Overfitting und ist damit für zukünftige
Prognosen besser gerüstet, verliert aber seine leichte Interpretierbarkeit. Ein
großer Vorteil des Entscheidungsbaumes (Decision Trees) ist ja, dass wir die
Entscheidungen als eine Abfolge von Entscheidungsfragen gut nachvollziehen
können. Jeder der einzelnen Entscheidungsbäume kommt jedoch zu einer anderen
Reihenfolge der Entscheidungsfragen und zu anderen Grenzen. Dafür bietet der
Random-Forest-Algorithmus eine alternative Bewertung, wie wichtig einzelne
Merkmale (Features) sind, die sogenannte **Feature Importance**.

Feature Importance bewertet, wie wichtig der Einfluss eines Merkmals (Features)
auf die Prognoseleistung ist. Ist die Feature Importance eines Merkmals
(Features) höher, so trägt dieses Merkmal (Feature) auch mehr zu der Genauigkeit
der Prognose bei. Bei Entscheidungsbäumen wird für jedes Merkmal (Feature)
berechnet, wie groß die Reduktion der Gini-Impurity ist. Gibt es ein Merkmal,
das eindeutig die Gini-Impurity reduziert, dann hat dieses Merkmal auch einen
großen Einfluss auf die Prognosefähigkeit des Modells. Wir könnten nach dem
Training des Entscheidungsbaumes zusammenfassen, wie oft und wieviel ein
bestimmtes Merkmal zur Reduktion beiträgt. In der Praxis kommt es aber oft vor,
dass bei einem Split mehrere Merkmale gleichermaßen die Gini-Impurity
reduzieren. Dann wird eines der Merkmale zufällig ausgewählt. Daher kann es
schwieirg sein, bei einem Entscheidungsbaum die Feature Importance zu bewerten.
Bei einem Random Forest hingegen werden viele Entscheidungsbäume trainiert. Wenn
wir jetzt bei allen Entscheiungsbäumen die Feature Importance berechnen und den
Mittelwert bilden, erhalten wir ein aussagekräftiges Bewertungskriterium, wie
stark einzelne Merkmale die Prognosefähigkeit beeinflussen.

Wir trainieren nun einen Random Forest mit der Standardeinstellung von 100
Entscheidungsbäumen und lassen uns dann die Feature Importance ausgeben.

```{code-cell} ipython3
model = RandomForestClassifier(random_state=0)
model.fit(X,y)

print(model.feature_importances_)
```

Der erste Wert gibt die Feature Importance für das erste Merkmal an und der zweite Wert entsprechend für das zweite Merkmal. Es ist üblich, die Feature Importance als Balkendiagramm zu visualisieren.

```{code-cell} ipython3
feature_importances = pd.Series(model.feature_importances_, index=['Kilometerstand [km]', 'Preis [EUR]'])

fig = px.bar(feature_importances, orientation='h',
  title='Verkaufsaktion im Autohaus', 
  labels={'value':'Feature Importance', 'index': 'Merkmal'})
fig.update_traces(showlegend=False) 
fig.show()
```

Der Preis ist demnach wichtiger als der Kilometerstand (wobei es hier ja ein
künstliches Beispiel ist).

### Zusammenfassung und Ausblick 9.2

Random Forests sind einfachen Entscheidungsbäumen vorzuziehen, da sie das
Overfitting reduzieren. Die Erzeugung der einzelnen Entscheidungsbäume kann
parallelisiert werden, so dass das Training eines Random Forests sehr schnell
durchgeführt werden kann. Auch für große Datenmengen mit sehr unterschiedlichen
Eigenschaften arbeitet der Random Forest sehr effizient. Er ermöglicht auch eine
Interpretation, welche Eigenschaften ggf. einen größeren Einfluss haben als
andere Eigenschaften

## 9.3 XGBoost

In der bisherigen Vorlesung haben wir vor allem Pandas und Scikit-Learn benutzt.
Zwar bietet Scikit-Learn Boosting-Verfahren an, in vielen Wettbewerben hat sich
jedoch eine andere Bibliothek durchgesetzt, die eine optimierte Variante des
Stochastic Gradient Boosting anbietet: **XGBoost**.

**Warnung:** Falls bei Ihnen XGBoost nicht installiert sein sollte, folgen Sie bitte den Anweisungen auf der Internesetseite [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io/en/stable/install.html) und installieren Sie XGBoost jetzt nach.

### Lernziele Kapitel 9.3

* Sie können XGBoost für Regressions- und Klassifikationsaufgaben einsetzen.
* Sie wissen, wie Sie mit Analysen der Maßzahlen Fehler und Logloss für
  Trainings- und Testdaten beurteilen können, ob Überanpassung (Overfitting)
  vorliegt.
* Sie kennen die Methode **Frühes Stoppen** zur Reduzierung der Überanpassung
  (Overfitting).
* Sie wissen, dass XGBoost nicht manuell feinjustiert werden sollte, sondern mit
  Gittersuche oder weiteren Bibliotheken (z.B. Optuna).

### XGBoost benutzt Scikit-Learn API

XGBoost steht für e**X**treme **G**radient **Boost**ing und ist aus
Performancegründen in der Programmiersprache C++ implementiert. Für
Python-Programmier wurde ein Python-Modul mit dem Ziel geschaffen, die gleichen
Schnittstellen wie Scikit-Learn anzubieten, so dass kaum Einarbeitungszeit in
eine neue Bibliothek erforderlich ist. Vor allem benötigen Data Scientists auch
keine C++\-Programmierkenntnisse, sondern können weiterhin mit Python arbeiten.

Wir bleiben bei unserem Beispiel mit der Verkaufsaktion im Autohaus aus dem
vorherigen Kapitel.

```{code-cell}
import pandas as pd 
from sklearn.datasets import make_moons

# Erzeugung künstlicher Daten
X_array, y_array = make_moons(n_samples=120, random_state=0, noise=0.3)

daten = pd.DataFrame({
    'Kilometerstand [km]': 10000 * (X_array[:,0] + 2),
    'Preis [EUR]': 5000 * (X_array[:,1] + 2),
    'verkauft': y_array,
    })
```

XGBoost kann Pandas DataFrames nicht verarbeiten, sondern benötigt die reinen
Zahlenwerte in Form von Matrizen. Das ist in der Tat kein Problem, denn die
Datenstruktur DataFrame stellt die reinen Matrizen über die Methode `.values`
direkt zur Verfügung.

```{code-cell}
# Adaption der Daten
X = daten[['Kilometerstand [km]', 'Preis [EUR]']].values
y = daten['verkauft'].values
```

Als nächstes importieren wir XGBoost. Es ist üblich, das ganze Modul zu
importieren und mit `xgb` abzukürzen. Danach initialisieren wir das
Klassifikationsmodell `XGBClassifier` und trainieren es auf den Daten.

```{code-cell}
import xgboost as xgb 

modell = xgb.XGBClassifier()
modell.fit(X,y)
```

Als nächstes visualisieren wir die Prognose des trainierten
XGBoost-Klassifikators.

```{code-cell}
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay

my_colormap = ListedColormap(['#EF553B33', '#636EFA33'])
fig = DecisionBoundaryDisplay.from_estimator(modell, X,  cmap=my_colormap)
fig.ax_.scatter(X[:,0], X[:,1], c=y, cmap=my_colormap)
fig.ax_.set_xlabel('Kilometerstand [km]');
fig.ax_.set_ylabel('Preis [EUR]');
fig.ax_.set_title('XGBoost: Entscheidungsgrenzen');
```

Die Entscheidungsgrenzen sehr plausibel aus.

### XGBoost neigt stark zur Überanpassung (Overfitting)

XGBoost ist bekannt für Überanpassung (Overfitting) an die Trainingsdaten. Um
das an unserem Beispiel mit der Verkaufsaktion im Autohaus zu zeigen, fügen wir
noch neue, unbekannte Testdaten hinzu. Dazu verdoppeln wir die Anzahl der Autos
(`n_samples=2000`).

```{code-cell}
# Erzeugung künstlicher Daten
X_array, y_array = make_moons(n_samples=2000, random_state=0, noise=0.3)

daten = pd.DataFrame({
    'Kilometerstand [km]': 10000 * (X_array[:,0] + 2),
    'Preis [EUR]': 5000 * (X_array[:,1] + 2),
    'verkauft': y_array,
    })

X = daten[['Kilometerstand [km]', 'Preis [EUR]']].values
y = daten['verkauft'].values
```

Anschließend teilen wir die 2000 Autos in zwei Gruppen: Trainings- und
Testdaten.

```{code-cell}
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.5, random_state=0)
```

Diesmal legen wir explizit fest, aus wievielen Modellen das Boosting-Verfahren
bestehen soll. Dazu setzen wir `n_estimators=200`. Oft wird auch von der Anzahl
der »Boosting-Runden« gesprochen. Das Training auf den Trainingsdaten liefert
sehr gute Ergebnisse:

```{code-cell}
import xgboost as xgb

modell = xgb.XGBClassifier(n_estimators=200)

modell.fit(X_train, y_train)

score_train = modell.score(X_train, y_train)
print(f'Score bezogen auf Trainingsdaten: {score_train:.2f}')
score_test = modell.score(X_test, y_test)
print(f'Score bezogen auf Testdaten: {score_test:.2f}')
```

Die Trainingsdaten werden perfekt prognostiziert. Auch bei den Testdaten
erhalten wir ein gutes Ergebnis, das aber im Vergleich zu dem sehr guten Score
bei den Trainingsdaten abfällt. Es fällt schwer, zu entscheiden, ob eine
Überanpassung (Overfitting) vorliegt. XGBoost ist ein iteratives Verfahren.
Zunächst wird Modell Nr. 1 trainiert, darauf aufbauend Modell Nr. 2 usw. Wir
wiederholen jetzt das Training des XGBoost-Klassifikators, aber lassen durch ein
weiteres Argument mitprotokollieren, was in den einzelnen Iterationen passiert.

Zuerst legen wir fest, welche internen Bewertungskennzahlen (= Metrik, Maßzahl)
mitprotokolliert werden sollen. Wir wählen als erste Maßzahl den Fehler, also
die relative Anzahl der falsch klassifizierten Autos. Die zweite Maßzahl
berechnet, wie weit die Wahrscheinlichkeit für »verkauft« oder »nicht verkauft«
vom tatsächlichen Ergebnis weg ist. Mathmatisch etwas präziser betrachten wir
die [Kreuzentropie](https://de.wikipedia.org/wiki/Kreuzentropie), bekannt als
»Losslog«.

Technisch setzen wir dies um, indem wir bei der Initialisierung des
XGBoost-Modells das optionale Argument `eval_metric=['error', 'logloss']`
setzen.

```{code-cell}
modell = xgb.XGBClassifier(n_estimators=200, eval_metric=['error', 'logloss'])
```

Allerdings ist damit noch nicht festgelegt, auf welchen Daten die Fehler-Maßzahl
und die Logloss-Maßzahl berechnet werden. Zunächst sollen beide Maßzahlen für
die Trainingsdaten berechnet werden, dann für die Testdaten. Das erreichen wir
mit dem optionalen Argument `eval_set=`, dem wir folgendermaßen die Trainings-
und Testdaten mitgeben.

```{code-cell}
modell.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
```

Wir setzen noch `verbose=False`, damit nicht für jedes Modell bzw. jede
Iteration die vier Maßzahlen auf dem Bildschirm ausgegeben werden. Nach dem
Training können wir die vier Maßzahlen mit der Methode `.evals_result()` aus dem
trainierten Modell extrahieren. Um die Maßzahlen zu visualisieren, packen wir
sie in einen Pandas-DataFrame.

```{code-cell} ipython3
masszahlen = modell.evals_result()
fehler = pd.DataFrame({
    'Fehler Trainingsdaten': masszahlen['validation_0']['error'],
    'Fehler Testdaten': masszahlen['validation_1']['error']
    })
losslog = pd.DataFrame({
    'Losslog Trainingsdaten': masszahlen['validation_0']['logloss'],
    'Losslog Testdaten': masszahlen['validation_1']['logloss']
    })
```

Wir visualisieren Fehler und Losslog getrennt voneinander.

```{code-cell}
import plotly.express as px 

fig = px.scatter(fehler,
    title='Fehler in jeder Iteration (Boosting-Runde)',
    labels={'value': 'Fehler', 'index': 'Iteration', 'variable': 'Legende'})
fig.show()
```

Der Fehler bei den Trainingsdaten wird von Boosting-Runde zu Boosting-Runde
kleiner, aber der Fehler der Testdaten wächst. Zunächst wird der Fehler der
Testdaten kleiner, erreicht in Minimum in der 6. Iteration, um dann wieder zu
steigen. Dieses Verhalten ist typisch für Überanpassung (Overfitting). Etwas
deutlicher wird dieses Phänomen, wenn wir uns die (transoformierte) Differenz
der Wahrscheinlichkeiten ansehen, die Losslog-Maßzahl.

```{code-cell}
import plotly.express as px 

fig = px.scatter(losslog,
    title='Losslog in jeder Iteration (Boosting-Runde)',
    labels={'value': 'Losslog', 'index': 'Iteration', 'variable': 'Legende'})
fig.show()
```

Am kleinsten ist die Losslog-Maßzahl für die Iteration 9, danach steigt die
Losslog-Maßzahl wieder an. Am besten wäre es nach dieser Analyse gewesen, nach
der 6. oder 9. Iteration aufzuhören, da dann die Überanpassung (Overfitting) an
die Trainingsdaten einsetzt.

### Bekämpfen von Überanpassung (Overfitting)

Es gibt einige Hyperparamter von XGBoost, die helfen, Überanpassung
(Overfitting) zu reduzieren. Eine Möglichkeit ist es, früher zu stoppen und
nicht die voreingestellte Anzahl an Modellen bzw. Iterationen / Boosting-Runden
zu durchlaufen. Das wird durch das optionale Argument `early_stopping_rounds=`
im Konstruktor ermöglicht. Die Zahl, die diesem Parameter übergeben wird, gibt
die Anzahl der Boosting-Runden vor, nach denen gestoppt wird, falls sich kaum
etwas an der Maßzahl geändert hat.

```{code-cell}
modell = xgb.XGBClassifier(n_estimators=200, early_stopping_rounds=10, eval_metric=['error', 'logloss'])
modell.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
```

Visualisiert sieht die Losslog-Statistik für das obige Beispiel so aus:

```{code-cell}
masszahlen = modell.evals_result()
fehler = pd.DataFrame({
    'Fehler Trainingsdaten': masszahlen['validation_0']['error'],
    'Fehler Testdaten': masszahlen['validation_1']['error']
    })
losslog = pd.DataFrame({
    'Losslog Trainingsdaten': masszahlen['validation_0']['logloss'],
    'Losslog Testdaten': masszahlen['validation_1']['logloss']
    })

fig = px.scatter(fehler,
    title='Frühes Stoppen: Fehler',
    labels={'value': 'Fehler', 'index': 'Iteration', 'variable': 'Legende'})
fig.show()

fig = px.scatter(losslog,
    title='Frühes Stoppen: Losslog',
    labels={'value': 'Losslog', 'index': 'Iteration', 'variable': 'Legende'})
fig.show()
```

Eine weitere Möglichkeit, Überanpassung (Overfitting) zu reduzieren, besteht
darin, die Tiefe der Entscheidungsbäume zu begrenzen. Wir benutzen
Entscheidungsbaum-Stümpfe, die eine Tiefe von Eins haben. Das erreichen wir mit
dem optionalen Argument `max_depth=1`.

```{code-cell}
modell = xgb.XGBClassifier(max_depth=1, n_estimators=200, eval_metric=['error', 'logloss'])
modell.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

masszahlen = modell.evals_result()
fehler = pd.DataFrame({
    'Fehler Trainingsdaten': masszahlen['validation_0']['error'],
    'Fehler Testdaten': masszahlen['validation_1']['error']
    })
losslog = pd.DataFrame({
    'Losslog Trainingsdaten': masszahlen['validation_0']['logloss'],
    'Losslog Testdaten': masszahlen['validation_1']['logloss']
    })

fig = px.scatter(fehler,
    title='Begrenzte Entscheidungsbäume: Fehler',
    labels={'value': 'Fehler', 'index': 'Iteration', 'variable': 'Legende'})
fig.show()

fig = px.scatter(losslog,
    title='Begrenzte Entscheidungsbäume: Losslog',
    labels={'value': 'Losslog', 'index': 'Iteration', 'variable': 'Legende'})
fig.show()
```

Es gibt noch einige weitere Hyperparameter, die für "das" beste Modell
feinjustiert werden können. Händisch gelingt es kaum, alle Hyperparameter
optimal einzustellen, so dass hier eine Gittersuche oder gar eine Bibliothek wie
[Optuna](https://github.com/optuna/optuna) eingesetzt werden sollte.

### Zusammenfassung und Ausblick Kapitel 9.3

Mit XGBoost haben Sie ein ML-Modell für das überwachte Lernen kennengelernt, das
in den vergangen Jahren sehr viele Wettbewerbe gewonnen hat. Die Mächtigkeit der
Algorithmen führt aber häufig zur Überanpassung (Overfitting), so dass die
sorgsame Feinjustierung der Hyperparameter besonders wichtig ist.

## Übungen

Beschäftigen Sie sich zum Abschluss des Jahres 2024 mit Ihren eigenen
Kontodaten. Laden Sie sich von Ihrer Bank Ihre Kontoauszüge als Excel- oder
csv-Datei herunter und importieren Sie sie. Behandeln Sie dann Ihre Ein- und
Ausgaben wie ein ML-Projekt. Sind die Daten vollständig? Was sind numerische
Eigenschaften, was kategoriale Merkmale? Führen Sie eine explorative
Datenanalyse durch.

Recherchieren Sie im Internet, Was sind passende Kategorien für ein
Haushaltsbuch? Nutzen Sie die replace()-Methode, um Python automatisch die
Einträge in Kategorien sortieren zu lassen (führen Sie die Kategorisierung ggf.
nur für einen Monat oder ein Quartal durch). Welches ML-Modell ist am besten
geeignet, um eine Prognose Ihres Kontostandes für 2025 zu erstellen?
