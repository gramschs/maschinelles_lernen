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

# 10. Support Vector Machines

Das maschinelle Lernverfahren Support Vector Machines gehört zu den überwachten
Lernverfahren. Sie können sowohl für Klassifikations- also auch
Regressionsprobleme eingesetzt werden. Prinzipiell könnten wir den Namen Support
Vector Machines ins Deutsche übersetzen, also das Verfahren als
Stützvektor-Maschine bezeichnen. Jedoch ist der deutsche Begriff so unüblich,
dass wir beim englischsprachigen Begriff bleiben oder einfach die Abkürzung SVM
verwenden. Zur Einführung der SVMs betrachten wir ein binäres
Klassifikationsproblem.

## 10.1 Maximiere den Rand, aber soft

### Lernziele 10.1

* Sie kennen die Abkürzung **SVM** für **Support Vector Machines**.
* Sie kennen die Idee, bei Support Vector Machines den **Margin** (=
  Randabstand) zu maximieren.
* Sie wissen, was Stützvektoren bzw. **Support Vectors** sind.
* Sie wissen, dass ein harter Randabstand nur bei linear trennbaren Datensätzen
  möglich ist.
* Sie wissen, dass eigentlich nicht trennbare Datensätzen mit der Technik **Soft
  Margin** (= weicher Randabstand) dennoch klassifiziert werden können.

### Welche Trenn-Gerade soll es sein?

Support Vector Machines (SVM) können sowohl für Klassifikations- als auch
Regressionsprobleme genutzt werden. Insbesondere wenn viele Merkmale (Features)
vorliegen, sind SVMs gut geeignet. Auch neigen SVMs weniger zu Overfitting.
Daher lohnt es sich, Support Vector Machines anzusehen.

Warum es weniger zu Overfitting neigt und mit Ausreißern besser umgehen kann,
sehen wir bereits an der zugrundeliegenden Idee, die hinter dem Verfahren
steckt. Um das Basis-Konzept der SVMs zu erläutern, besorgen wir uns zunächst
künstliche Messdaten. Dazu verwenden wir die Funktion `make_blobs` aus dem
Scikit-Learn-Modul. Mehr Details zum Aufruf der Funktion finden Sie in der
[Scikit-Learn-Dokumentation/make_blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html?highlight=make+blobs#sklearn.datasets.make_blobs).

```{code-cell} ipython3
from sklearn.datasets import make_blobs

# generate artificial data
X, y = make_blobs(n_samples=60, centers=2, random_state=0, cluster_std=0.50)

print(X)
print(y)
```

Die Funktion `make_blobs` erzeugt standardmäßig zwei Input-Features, da die
Option `n_features` auf den Wert 2 voreingestellt ist, und einen Output, bei dem
die Labels entweder durch 0 oder 1 gekennzeichnet sind. Durch die Option
`random_state=0` wird der Zufall ausgeschaltet.

Wenn wir die Daten in einen Pandas-DataFrame packen und anschließend
visualisieren, erhalten wir folgenden Plot.

```{code-cell} ipython3
import pandas as pd 
import plotly.express as px

daten = pd.DataFrame({
    'Feature 1': X[:,0],
    'Feature 2': X[:,1],
    'Status': y.astype(bool),
    })

fig = px.scatter(daten, x = 'Feature 1', y = 'Feature 2',  color='Status',
                 title='Künstliche Daten', color_discrete_sequence=['#b40426','#3b4cc0'])
fig.show()
```

Wir können uns jetzt verschiedene Geraden vorstellen, die die blauen Punkte von
den roten Punkten trennen. In der folgenden Grafik sind drei eingezeichnet.
Welche würden Sie nehmen und warum?

![verschiedene Trenngeraden in einem Scatterplot](https://gramschs.github.io/book_ml4ing/_images/fig10_01_annotated.pdf)

Alle drei Geraden trennen die blauen von den roten Punkten. Jedoch könnte Gerade
3 problematisch werden, wenn beispielsweise ein neuer blauer Datenpunkt an der
Position (2.3, 3.3) dazukäme. Dann würde Gerade 3 diesen Punkt als rot
klassifizieren. Ähnlich verhält es sich mit Gerade 1. Ein neuer blauer
Datenpunkt an der Position (0.5, 3) würde fälschlicherweise als rot
klassifiziert werden. Gerade 2 bietet den sichersten Abstand zu den bereits
vorhandenen Datenpunkten. Wir können diesen "Sicherheitsstreifen" folgendermaßen
visualisieren.

![Sicherheitsstreifen](https://gramschs.github.io/book_ml4ing/_images/fig10_02_annotated.pdf)

Der Support-Vector-Algorithmus sucht nun die Gerade, die die Datenpunkte mit dem
größten Randabstand (= Margin) voneinander trennt. Im Englischen sprechen wir
daher auch von **Large Margin Classification**. Die Suche nach dieser Geraden
ist dabei etwas zeitaufwändiger als die Berechnung der Gewichte bei der
logistischen Regression. Wenn aber einmal das Modell trainiert ist, ist die
Prognose effizienter, da nur die sogenannten **Stützvektoren**, auf englisch
**Support Vectors** gespeichert und ausgewertet werden. Die Stützvektoren sind
die Vektoren, die vom Ursprung des Koordinatensystems zu den Punkten zeigen, die
auf der Grenze des Sicherheitsbereichs liegen.  

![Stützvektoren eingezeichnet](https://gramschs.github.io/book_ml4ing/_images/fig10_03.pdf)

### Großer, aber weicher Randabstand

Bei dem oben betrachteten Beispiel lassen sich blaue und rote Datenpunkte
komplett voneinander trennen. Für den Fall, dass einige wenige Datenpunkte
"falsch" liegen, erlauben wir Ausnahmen. Wie viele Ausnahmen wir erlauben
wollen, die im Sicherheitsstreifen liegen, steuern wir mit dem Parameter `C`.
Ein großes `C` bedeutet, dass wir eine große Mauer an den Grenzen des
Sicherheitsabstandes errichten. Es kommt kaum vor, dass Datenpunkte innerhalb
des Margins liegen. Je kleiner `C` wird, desto mehr Datenpunkte sind innerhalb
des Sicherheitsbereichs erlaubt.

Im Folgenden betrachten wir einen neuen künstlichen Datensatz, bei dem die
blauen von den roten Punkte nicht mehr ganz so stark getrennt sind. Schauen Sie
sich die fünf verschiedenen Margins an, die entstehen, wenn der Parameter `C`
variiert wird.

```{code-cell} ipython3
:tags: [remove-input]
from IPython.display import HTML
HTML('../assets/chapter10/fig04.html')
```

### Zusammenfassung Kapitel 10.1

In diesem Abschnitt haben wir die Ideen kennengelernt, die den Support Vector
Machines zugrunde liegen. Im nächsten Abschnitt schauen wir uns an, wie ein
SVM-Modell mit Scikit-Learn trainiert wird.

## 10.2 Training SVM mit Scikit-Learn

### Lernziele Kapitel 10.2

* Sie können ein SVM-Modell mit Scikit-Learn trainieren.

### Scikit-Learn bietet mehrere Implementierungen

Wenn wir in der Dokumentation von Scikit-Learn
[Scikit-Learn/SVM](https://scikit-learn.org/stable/modules/svm.html) die Support
Vector Machines nachschlagen, so finden wir drei Einträge

* SVC,
* NuSVC und
* LinearSVC.

Die Beispiele des letzten Abschnittes sind linearer Natur, so dass sich
eigentlich die Klasse "LinearSVC" aus Effiziengründen anbieten würde. Da wir
aber im nächsten Abschnitt uns auch mit nichtlinearen Problemen beschäftigen
werden, fokussieren wir uns gleich auf den SVC-Algorithmus mit seinen Optionen.
NuSVC ist ähnlich zu SVC, bietet aber die zusätzliche Möglichkeit, die Anzahl
der Stützvektoren einzuschränken.

Vielleicht wundern Sie sich, dass die Klasse SVC und nicht SVM heißt. Das C in
SVC soll deutlich machen, dass wir die Support Vector Machines nutzen wollen, um
ein Klassifikationsproblem (= Classification Problem) zu lösen.

### Training mit fit und score

Zuerst importieren wir aus Scikit-Learn das entsprechende Modul 'SVM' und
instantiieren ein Modell. Da wir die etwas allgemeinere Klasse SVC anstatt
LinearSVC verwenden, müssen wir bereits bei der Erzeugung die Option `kernel=`
auf linear setzen, also `kernel='linear'`.

```{code-cell} ipython3
from sklearn import svm
svm_modell = svm.SVC(kernel='linear')
```

Wir erzeugen uns erneut künstliche Messdaten.

```{code-cell} ipython3
from sklearn.datasets import make_blobs
import matplotlib.pylab as plt; plt.style.use('bmh')

# generate artificial data
X, y = make_blobs(n_samples=60, centers=2, random_state=0, cluster_std=0.50)

# plot artificial data
import plotly.express as px

fig = px.scatter(x = X[:,0], y = X[:,1],  color=y, color_continuous_scale=['#3b4cc0', '#b40426'],
                 title='Künstliche Daten',
                 labels={'x': 'Feature 1', 'y': 'Feature 2'})
fig.show()
```

Als nächstes teilen wir die Messdaten in Trainings- und Testdaten auf.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

Nun können wir unser SVM-Modell trainieren:

```{code-cell} ipython3
svm_modell.fit(X_train, y_train);
```

Und als nächstes analysieren, wie viele der Testdaten mit dem trainierten Modell
korrekt klassifiziert werden.

```{code-cell} ipython3
svm_modell.score(X_test, y_test)
```

Ein super Ergebnis! Schön wäre es jetzt noch, die gefundene Trenngerade zu
visualisieren. Dazu modifizieren wir einen Code-Schnippsel aus dem Buch: »Data
Science mit Python« von Jake VanderPlas (mitp Verlag 2017), ISBN 978-3-95845-
695-2, siehe
[https://github.com/jakevdp/PythonDataScienceHandbook](https://github.com/jakevdp/PythonDataScienceHandbook).

```{code-cell} ipython3
# Quelle: VanderPlas "Data Science mit Python", S. 482
# modified by Simone Gramsch
import numpy as np

def plot_svc_grenze(model):
    # aktuelles Grafik-Fenster auswerten
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Raster für die Auswertung erstellen
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    # Entscheidungsgrenzen und Margins darstellen
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    # Stützvektoren darstellen
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1, facecolors='none', edgecolors='orange');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
```

```{code-cell} ipython3

fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('SVM mit Soft Margin');

plot_svc_grenze(svm_modell)
```

### Der Parameter C

Im letzten Abschnitt haben wir uns mit dem Parameter `C` beschäftigt, der
Ausnahmen innerhalb des Sicherheitsstreifens erlaubt. Ein großes `C` bedeutet
ja, dass die Wand des Margins hoch ist und kaum (oder gar keine) Punkte
innerhalb des Sicherheitsstreifens liegen dürfen. Als nächstes schauen wir uns
an, wie der Parameter `C` gesetzt wird.  

Die Option zum Setzen des Parameters C lautet schlicht und einfach `C=`. Dabei
muss C immer positiv sein.

Damit aber besser sichtbar wird, wie sich C auswirkt, vermischen wir die
künstlichen Daten stärker.

```{code-cell} ipython3
# generate artificial data
X, y = make_blobs(n_samples=60, centers=2, random_state=0, cluster_std=0.80)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# plot artificial data
import plotly.express as px

fig = px.scatter(x = X[:,0], y = X[:,1],  color=y, color_continuous_scale=['#3b4cc0', '#b40426'],
                 title='Künstliche Daten',
                 labels={'x': 'Feature 1', 'y': 'Feature 2'})
fig.show()
```

```{code-cell} ipython3
# Wahl des Modells mit linearem Kern und großem C
svm_modell = svm.SVC(kernel='linear', C=1000000)

# Training und Bewertung
svm_modell.fit(X_train, y_train);
svm_modell.score(X_test, y_test)

# Visualisierung
fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('SVM mit Soft Margin');
plot_svc_grenze(svm_modell)

```

```{code-cell} ipython3
# Wahl des Modells mit linearem Kern und kleinem C
svm_modell = svm.SVC(kernel='linear', C=1)

# Training und Bewertung
svm_modell.fit(X_train, y_train);
svm_modell.score(X_test, y_test)

# Visualisierung
fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('SVM mit Soft Margin');
plot_svc_grenze(svm_modell)
```

### Zusammenfassung Kapitel 10.2

Verwenden wir den SVC-Klassifikator aus dem Modul SVM von Scikit-Learn, können
wir mittels der Option `kernel='linear'` eine binäre Klassifikation durchführen,
bei der die Trennungsgerade den größtmöglichen Abstand zwischen den
Punkteclustern erzeugt, also einen möglichst großen Margin. Sind die Daten nicht
linear trennbar, so können wir mit der Option `C=` steuern, wie viele Ausnahmen
erlaubt werden sollen. Mit Ausnahmen sind Punkte innerhalb des Margins gemeint.
Im nächsten Abschnitt betrachten wir nichtlineare Trennungsgrenzen.

## 10.3 Nichtlineare SVM

### Lernziele Kapitel 10.3

* Sie kennen den **Kernel-Trick**.
* Sie können mit den **radialen Basisfunktionen** als neue Option für
  SVM-Verfahren nichtlinear trennbare Daten klassifizieren.

### Nichtlineare trennbare Daten

Für die Support Vector Machines sind wir bisher davon ausgegangen, dass die
Daten -- ggf. bis auf wenige Ausnahmen -- linear getrennt werden können. Im
Folgenden betrachten wir nun einen künstlichen Messdatensatz, bei dem das
offensichtlich nicht geht. Dazu nutzen wir die in Scikit-Learn integrierte
Funktion `make_circles()`.

```{code-cell} ipython3
from sklearn.datasets import make_circles

# künstliche Messdaten generieren
X,y = make_circles(100, random_state=0, factor=0.3, noise=0.1)

# künstliche Messdaten visualisieren
import plotly.express as px

fig = px.scatter(x = X[:,0], y = X[:,1],  color=y, color_continuous_scale=['#3b4cc0', '#b40426'],
                 title='Künstliche Daten',
                 labels={'x': 'Feature 1', 'y': 'Feature 2'})
fig.show()
```

Das menschliche Auge erkennt sofort das Muster in den Daten. Ganz offensichtlich
sind die roten und blauen Punkte kreisförmig angeordnet und können
dementsprechend auch durch einen Kreis getrennt werden. Allerdings wird ein
SVM-Klassifikator, so wie wir das SVM-Verfahren bisher kennengelernt haben,
versagen. Eine Gerade zur Klassifikation der roten und blauen Punkte passt
einfach nicht.

+++

### Aus 2 mach 3

Die Idee zur Überwindung dieses Problems klingt zunächst einmal absurd. Wir
machen aus zwei Features drei Features. Als drittes Feature wählen wir das
Quadrat des Abstandes eines Punktes zum Ursprung.

```{code-cell} ipython3
import numpy as np
import plotly.express as px

# Extraktion der Daten, damit leichter darauf zugegriffen werden kann
X1 = X[:,0]
X2 = X[:,1]

# neues Feature als Quadrat des Abstandes zum Ursprung
X3 = np.sqrt( X1**2 + X2**2 )

fig = px.scatter_3d(x=X1, y=X2, z=X3, color=y, color_continuous_scale=['#3b4cc0', '#b40426'])
fig.show()
```

Bitte drehen Sie die Ansicht solange, bis die z-Achse nach oben zeigt. Die
Punkte bilden eine Art Paraboloiden. In dieser neuen Ansicht können wir eine
Ebene finden, die die roten von den blauen Punkten trennt.

In der folgenden Grafik ist eine Trennebene eingezeichnet. Wenn wir nun den
Schnitt der Trennebene mit dem Paraboloiden bilden, entsteht eine Kreislinie.
Drehen wir wieder unsere Ansicht zurück, so dass wir von oben auf die
X1-X2-Feature-Ebene blicken, so ist dieser Kreis genau das, was wir auch als
Menschen genommen hätten, um die roten von den blauen Punkten zu trennen.

![3D-Scatterplot mit Trenngerade](https://gramschs.github.io/book_ml4ing/_images/fig10_06_with_plane.png)

![Trennebene](https://gramschs.github.io/book_ml4ing/_images/fig10_07_with_circle.png)

### Kernel-Trick

Bei diesem künstlichen Datensatz hat das Quadrat der Abstände zum Ursprung als
neues Feature sehr gut funktioniert. Das lag aber unter anderem daran, dass die
Punkte tatsächlich in Kreisen um den Ursprung verteilt waren. Was ist, wenn das
nicht der Fall ist? Wenn der Schwerpunkt der Kreise verschoben wäre, müssten wir
auch die Transformationsfunktion zum Erzeugen des dritten Features in diesen
Schwerpunkt verschieben.

Glücklicherweise übernimmt Scikit-Learn für uns die Suche nach einer passenden
Transformationsfunktion automatisch. Das Verfahren, das dazu in die
SVM-Algorithmen eingebaut ist, wird **Kernel-Trick** genannt. Es beruht darauf,
dass manche Funktionen in ein Skalarprodukt umgewandelt werden können. Und dann
wird nicht das dritte Feature mit der Transformationsfunktion aus den ersten
beiden Features berechnet, was sehr zeitaufwendig werden könnte, sondern die
Transformationsfunktion wird direkt in das Lernverfahren eingebaut. Da
Funktionen, die dafür geeignet sind, werden als **Kernel-Funktionen**
bezeichnet.

Am häufigsten zum Einsatz kommt dabei die sogenannte **radiale Basisfunktion**.
Die radialen Basisfunktionen werden mit **RBF** abgekürzt. Sie haben die tolle
Eigenschaft, dass sie nur vom Abstand eines Punktes zum Ursprung abhängen; so
wie unser Beispiel oben.

Um nichtlinear trennbare Daten zu klassifizieren, nutzen wir in Scikit-Learn das
SVC-Lernverfahren. Doch diesmal wählen wir als Kern nicht die linearen
Funktionen, sondern die sogenannten radialen Basisfunktionen RBF.

```{code-cell} ipython3
from sklearn import svm
svm_modell = svm.SVC(kernel='rbf')
```

Danach erfolgt das Training wie gewohnt mit der `fit()`-Methode, die Bewertung
mit der `score()`-Methode.

```{code-cell} ipython3
svm_modell.fit(X,y);
score = svm_modell.score(X,y)

print('Score: {:.2f}'.format(score))
```

Wir können erneut die Funktion `plot_svc_grenze()`aus dem vorherigen Abschnitt
nutzen, um die Stützvektoren mit einem orangefarbenem Kreis zu markieren und die
Entscheidungsgrenze zu visualisieren. Durch die radialen Basisfunktionen
erhalten wir keinen Kreis, sondern ein deformiertes Ei. Dafür brauchen wir uns
aber keine Gedanken über die Wahl der Funktion zu machen, um das neue Feature
aus den bisherigen zu berechnen.

```{code-cell} ipython3
# Quelle: VanderPlas "Data Science mit Python", S. 482
# modified by Simone Gramsch
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pylab as plt; plt.style.use('bmh')

def plot_svc_grenze(model):
    # aktuelles Grafik-Fenster auswerten
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Raster für die Auswertung erstellen
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    # Entscheidungsgrenzen und Margins darstellen
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    # Stützvektoren darstellen
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1, facecolors='none', edgecolors='orange');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(X1, X2, c=y, cmap='coolwarm')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Künstliche Messdaten');
plot_svc_grenze(svm_modell)
```

### Zusammenfassung

In diesem Abschnitt haben wir uns mit nichtlinearen Support Vector Machines
beschäftigt. Die Idee zur Klassifizierung nichtlinearer Daten ist, ein neues
Feature hinzuzufügen. Mathematisch gesehen projizieren wir also die Daten mit
einer nichtlinearen Transformationsfunktion in einen höherdimensionalen Raum und
trennen sie in dem höherdimensionalen Raum. Dnn kehren wir durch den Schnitt der
Trennebene mit der Transformationsfunktion wieder in den ursprünglichen Raum
zurück. Wenn wir als Transformationsfunktion die sogenannten Kernel-Funktionen
verwenden, können wir auf die Transformation der Daten verzichten und die
Transformation direkt in die SVM einbauen. Das wird Kernel-Trick genannt und
sorgt für die Effizienz und damit Beliebtheit von SVMs.
