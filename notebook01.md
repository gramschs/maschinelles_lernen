---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 1. Grundbegriffe des maschinellen Lernens

## 1.1 Was ist maschinelles Lernen?

KI (Künstliche Intelligenz) ist in aller Munde. Etwas seltener wird der Begriff
**maschinelles Lernen** verwendet. Maschinelles Lernen, oft auch **Machine
Learning** genannt, ist ein Teilgebiet der Künstlichen Intelligenz.

Wir kürzen in dieser Vorlesung maschinelles Lernen oft mit **ML** ab. Damit
umgehen wir die Diskussion, warum Künstliche Intelligenz mit einem
Großbuchstaben beginnt und maschinelles Lernen mit einem Kleinbuchstaben.
Gleichzeitig ist das auch die gängige Abkürzung im englischen Sprachgebrauch.
Dieses Kapitel klärt, was maschinelles Lernen ist und führt in die
grundlegenenden Bestandteile eines ML-Systems ein.

### Lernziele Kapitel 1.1

* Sie wissen, wie langes es das Forschungsgebiet **maschinelles Lernen** gibt
  und warum es sich in den letzten beiden Jahrzehnten so stark entwickelt hat.
* Sie kennen die Bestandteile eines ML-Systems: **Daten**, **Algorithmus** und
  **Modell**.

### Ein wenig Geschichte

Viele glauben, dass die Forschungsgebiete Künstliche Intelligenz und
maschinelles Lernen Neuentwicklungen des 21. Jahrhunderts sind. Doch tatsächlich
hat Arthur L. Samuel bereits 1959 maschinelles Lernen wie folgt definiert:

»... ein Forschungsgebiet, das Computer in die Lage versetzen soll, zu lernen,
ohne explizit darauf programmiert zu sein.«

Arthur L. Samuel, 1959

[Wikipedia → Maschinelles Lernen](https://de.wikipedia.org/wiki/Maschinelles_Lernen)
bietet eine weitere Definition:

»Maschinelles Lernen (ML) ist ein Oberbegriff für die „künstliche“ Generierung
von Wissen aus Erfahrung: Ein künstliches System lernt aus Beispielen und kann
diese nach Beendigung der Lernphase verallgemeinern.«

Auch hier wird der Aspekt betont, dass das künstliche System selbst lernt. Aber
was ist mit selbst Lernen gemeint? Ein Kind lernt beispielsweise selbst das
Laufen. Auch wenn Eltern präzise beschreiben könnten, welcher Muskel zu welchem
Zeitpunkt kontrahiert werden muss und mit welcher Geschwindigkeit in welche
Richtung das Bein bewegt werden soll, würde das Kind die Anweisungen nicht
verstehen können. Ein Kind lernt selbst durch Versuch und Irrtum. In der
Anfangszeit der Robotik versuchten Forscherinnen und Forscher, Roboter durch
explizite Befehle zu steuern. Doch bei unvorhergesehenen Hindernissen stießen
solche Roboter an ihre Grenzen. Aus der Notwendigkeit, dass Roboter ähnlich wie
Menschen lernen, entwickelte sich das Teilgebiet maschinelles Lernen innerhalb
der Künstlichen Intelligenz.

Doch nicht nur in der Robotik spielt maschinelles Lernen eine wichtige Rolle.
Auch bei der Datenanalyse kann es hilfreich sein, wenn ein Computersystem
eigenständig Muster in den Daten erkennt. Ein Klassiker dafür ist die
Spam-Erkennung. Natürlich ist es möglich, den Spam-Filter mit expliziten Regeln
zu programmieren. Schon nach kurzer Zeit ändern jedoch Spammer die E-Mail-Texte
und schon greifen die expliziten Regeln nicht mehr. Hier helfen maschinell
gelernte Regeln. Durch die Markierung von E-Mails als Spam lernt das
E-Mail-Programm nach und nach selbst Regeln, um Spam-E-Mails zu identifizieren.

Die exponentiell wachsende Datenmenge der letzten beiden Jahrzehnte hat das
Interesse an maschinellen Lernverfahren stark erhöht. Es gibt aber auch noch
andere Gründe, die zum aktuellen Boom des maschinellen Lernens beigetragen
haben.

**Mini-Übung**
Schauen Sie sich das folgende Video an. Welche drei Gründe werden dort genannt,
warum maschinelles Lernen zuletzt so stark nachgefragt wurde?

<https://youtu.be/l_HSWmxMRlU?feature=shared>

Es gibt einige Gründe, warum ML in den letzten zwei Jahrzehnten so stark an
Bedeutung gewonnen hat. Im Folgenden werden die wichtigsten Gründe für den
Bedeutungsgewinn erläutert.

**Datenverfügbarkeit**: Die Produktion von Daten hat mit der Digitalisierung
massiv zugenommen. Mit der Einführung der Smartphones ist auch die Anzahl der
Bilder und Videos deutlich gewachsen, die täglich aufgenommen werden. Das
Kaufverhalten von Kunden in Online-Shops wird beobachtet, Fußballspiele werden
statistisch analysiert oder Maschinen mit Messsensoren bestückt. Das
Beratungsunternehmen IDC (International Data Corporation) prognostiziert, dass
sich die Datenmenge von 33 Zettabytes im Jahr 2018 auf 175 Zettebytes im Jahr
2025 mehr als verfünffachen wird {cite}`reinsel2018`.

**Rechenleistung**: Der rasante Fortschritt in der Computertechnologie hat die
Rechenleistung, die für maschinelles Lernen erforderlich ist, drastisch erhöht.
Speziell für die Entwicklung sogenannter neuronaler Netze werden sogar häufig
Grafikkarten (GPUs) anstatt eines Prozessors (CPU) bevorzugt.

**Algorithmen und Modelle**: Natürlich ist auch die Erforschung neuer
Algorithmen und Modelle nicht stehengeblieben. Ein Durchbruch in der Forschung
war dabei die Entwicklung von den sogenannten Deep-Learning-Modellen, einer
Variante der neuronalen Netze.

**Software und Tools**: Es gibt jetzt eine Vielzahl von Softwarebibliotheken und
Tools (z.B. [TensorFlow](https://www.tensorflow.org),
[PyTorch](https://pytorch.org),
[Scikit-learn](https://scikit-learn.org/stable/index.html) usw.), die es
ermöglichen, auch ohne tiefergehende Mathematik- und Programmierkenntnissen
maschinelles Lernen in der Praxis einzusetzen. Daher werden immer mehr
Anwendungen mit maschinellem Lernen analysiert und optimiert.

Bisher haben wir nicht besprochen, was es mit Künstlicher Intelligenz und Deep
Learning auf sich hat. Beide Begriffe werden oft in einem Atemzug mit ML
genannt. Das folgende Video gibt eine Einführung dazu.

<https://youtu.be/tCApwsdijDk?feature=shared>

### Was sind Algorithmen und Modelle?

Ein notwendiger Baustein des maschinellen Lernens sind Daten, am besten ganz,
ganz viele! Aber selbst ein riesiger Haufen an Daten ist alleine wertlos. Erst
durch Algorithmen, die in diesen Daten Muster finden, gewinnen wir neues Wissen,
können Prozesse analysieren und Entscheidungen treffen.

Aber was ist nun ein Algorithmus? In dem folgenden Video wird zuerst erklärt,
was ein Algorithmus ist. Danach werden die ersten grundlegenden Fachbegriffe des
maschinellen Lernens eingeführt.

**Mini-Übung**

Schauen Sie sich das folgende Video an und beantworten Sie die folgenden Fragen:

1. Welche drei Beispiele für Algorithmen werden aufgezählt?
2. In dem Video werden die Fachbegriff »Feature« und »Label« eingeführt. Was bedeuten die
  Begriffe?
3. Was bedeutet »überwachtes Lernen«?

<https://youtu.be/HmUzceKCI9I?feature=shared>

Ein **Algorithmus** ist also eine Anleitung, wie ein Problem zu lösen ist.
Typisch für einen Algorithmus ist, das in sehr kleinen Schritten detailliert
Anweisungen formuliert werden, um das Problem zu lösen. In der Informatik sind
Algorithmen besonders wichtig, da durch sie festgelegt wird, wie der Computer
Daten verarbeiten und ein Problem lösen soll. Jeder einzelne Schritt zur
Problemlösung muss eindeutig und konkret beschrieben werden. Wird der
Algorithmus in einer Programmiersprache formuliert, verwendet man den Begriff
**Computerprogramm**.

**Was ist ... ein Algorithmus?**

Ein Algorithmus ist eine spezifische Anleitung mit einzelnen Anweisungen, wie ein
bestimmtes Problem gelöst werden soll.

Der Begriff **Modell** hat viele verschiedenen Bedeutungen (siehe [Wikipedia →
Modell
(Begriffsklärung)](https://de.wikipedia.org/wiki/Modell_(Begriffsklärung))).
Zunächst einmal bedeutet Modell, das ein Original vereinfacht beschrieben wird.
Vereinfacht heißt, dass beispielsweise Details weggelassen werden oder die
Abmessungen geändert werden. In der Architektur wird beispielsweise ein Haus in
kleinem Maßstab gebaut, um potentiellen Kunden durch das Modell einen besseren
Eindruck zu vermitteln, wie das Haus in echt aussehen wird. Im Maschinenbau wird
das Modell eines Flugzeugs in einen Windkanal gehalten, um die Flugzeuggeometrie
zu optimieren. In manchen naturwissenschaftlichen Museen gibt es begehbare
Modelle von Organen wie beispielsweise dem Darm, um den Aufbau des Darms
begreifbar zu machen. Es gibt auch auch virtuelle Modelle wie  das
[Vier-Ohren-Modell](https://de.wikipedia.org/wiki/Vier-Seiten-Modell) des
Kommunikationswissenschaftlers Friedemann Schulz von Thun, das das
Kommunikationsverhalten von Menschen beschreibt.

In der Welt des maschinellen Lernens bezieht sich der begriff Modell darauf,
Daten zu interpretieren und basierend auf diesen Daten Vorhersagen oder
Entscheidungen zu treffen.

**Was ist ... ein Modell?**

Ein Modell ist ein vereinfachtes Abbild der Wirklichkeit. Im Kontext das
maschinellen Lernens ist das ML-Modell eine abstrake Beschreibung eines Systems,
das unbekannte Daten interpretieren kann oder basierend auf diesen Daten
Vorhersagen oder Entscheidungen treffen kann.

### Maschinelles Lernen ist wie Kuchenbacken

Damit kommen beim maschinellen Lernen drei Dinge zusammen: Daten, Algorithmus
und Modell. Um das Verhältnis zwischen den drei Konzepten zu verdeutlichen,
vergleichen wir die Konzepte mit dem Kuchenbacken. Die Daten sind die Zutaten,
aus denen ein Kuchen gebacken werden soll. Der Algorithmus ist das Rezept mit
einer detaillierten Schritt-für-Schritt-Anleitung, wie der Kuchen gebacken
werden soll. Das Modell hingegen ist der fertige Kuchen, der aus dem Prozess
herauskommt. Es ist somit das Endprodukt, das erstellt wird, indem man den
Anweisungen des Rezepts (dem Algorithmus) folgt und die Zutaten zusammenfügt.

Sobald das Modell bzw. der Kuchen fertig ist, wird dieses Modell
dann verwendet, um Vorhersagen zu treffen oder Entscheidungen zu treffen, genau
wie man einen Kuchen essen würde, nachdem er gebacken ist.

![Analogie zwischen dem ML-Workflow und dem Kuchenbacken (Quelle: eigene Darstellung)](https://gramschs.github.io/book_ml4ing/_images/ml_as_baking.png)

Allerdings ist es damit nicht getan. Je nachdem, wie viele und welche Gäste
erwartet werden, benötigen wir einen anderen Kuchen. Bei einer großen
Geburtstagsparty brauchen wir einen Blechkuchen, damit jeder Gast ein Stückchen
Kuchen bekommt. Haben wir Diabetiker eingeladen, sollten wir keine
Schokoladentorte anbieten. Auch beim maschinellen Lernen ist es daher sehr
wichtig, je nach Einsatzzweck das passende Modell bzw. den passenden Algorithmus
zu wählen.

Natürlich hängt die Wahl des Kuchens auch von den vorhandenen Zutaten ab. Fehlt
die Schokolade, so kann ich keinen Schokoladenkuchen backen. Entweder backen wir
dann einen anderen Kuchen oder wir entscheiden uns, noch schnell zum Supermarkt
zu gehen und Schokolade einzukaufen. Vielleicht stellen wir auch fest, dass die
Milch abgelaufen ist und nicht mehr genießbar ist. Dann ist unser Plan nicht
durchführbar. Und auch hier können wir uns entscheiden, einen anderen Kuchen zu
backen oder die Zutaten zu erneuern. Diese Analogie passt auch zu maschinellem
Lernen. Fehlen Daten oder sind die Daten nicht qualititativ hochwertig, können
wir die Datenlage verbessern, indem wir beispielsweise mehr Experimente
durchführen oder offensichtlich schiefgelaufene Experimente wiederholen. Diese
Phase des maschinellen Lernen wird auch **Datenerkundung** oder
**Datenexploration** genannt. Sollten wir die Daten jedoch nicht verbessern
können (oder wollen, weil zu teuer oder die Abgabefrist der Bachelorarbeit
ansteht), dann müssen wir die Auswahl des Modells an die vorhandenen Daten
anpassen.

Zutaten komplett, Rezept ausgewählt, Kuchen gebacken, der Gast beißt in den
Kuchen und verzieht das Gesicht ... Zucker und Salz verwechselt. Hätten wir den
Kuchen lieber einmal vor dem Servieren probiert. Auch beim maschinellen Lernen
ist es mit dem "Backen" des Modells nicht getan. Ist ein Modell erstellt, so
muss es auch bewertet werden. Der Prozess des maschinellen Lernens wird mit der
**Validierung** abgeschlossen, bevor das Modell dann produktiv eingesetzt wird.
Die Erstellung und Verwendung von Modellen im maschinellen Lernen ist ein
fortlaufender Prozess. Modelle werden oft mehrfach getestet und angepasst, um
ihre Leistung zu verbessern. Beim Kuchenbacken könnte der Bäcker auf die Idee
kommen, den Kuchen nicht bei 160 °C, sondern bei 162 °C zu backen, weil dann der
Kuchen noch besser schmeckt. Solche Parameter zum Finetunen eines Modells werden
**Hyperparameter** genannt. Hyperparameter haben nichts mit den vorhanden Daten
zu tun, sondern gehören zum ML-Modell. Aber auch wenn sich Daten verändern und
neue Daten hinzukommen, muss das Modell aktualisiert werden, um mit den sich
ändernden Bedingungen zurechtzukommen.

Die folgende Skizze zeigt den schematischen Ablauf eines typischen ML-Projektes.
Dabei benutzen wir das sogenannte QUA<sup>3</sup>CK-Modell nach einem Vorschlag
von {cite}`stock2020`. Das QUA<sup>3</sup>CK-Modell zeigt den typischen Ablauf
eines ML-Projektes von der wissenschaftlichen Fragestellung (Q -- Question) bis
zu deren Beantwortung (K -- Knowledge Transfer). Dazu gehört das Sammeln und
Erkunden der Daten (U -- Understanding the data), mit Hilfe derer die Frage
beantwortet werden soll. Die Phase der ML-Modellbildung wird mehrfach
durchlaufen und besteht aus der Auswahl und dem Training des Algorithmus bzw.
des Modells (A -- Algorithm selection and training), dazu passend der Auswahl
und Anpassung der Daten (A -- Adaption of the data) sowie der Anpassung der
Hyperparamter (A -- Adjustement of the hyperparameter). Die Modelle, die durch
diese Schleife erstellt werden, werden letztendlich miteinander verglichen und
bewertet (C -- Comparison and Conclusion), bevor sie produktiv eingesetzt
werden.

![Typischer Ablauf eines ML-Projektes als QUA<sup>3</sup>CK-Prozess dargestellt
(Quelle: in Anlehnung an {cite}`stock2020`)](https://gramschs.github.io/book_ml4ing/_images/qua3ck_process.png)

Das folgende Video erklärt den ML-Workflow etwas detaillierter, als wir es mit
der Kuchenbacken-Analogie getan haben. Als Ausblick auf die weitere Vorlesung
bietet dieses Video dennoch eine sehr gute Übersicht über die Vorgehenweise in
einem ML-Projekt und ist daher sehr empfehlenswert.

<https://youtu.be/f9QT2dqeW04?feature=shared>

## 1.2 Überwachtes, unüberwachtes und verstärkendes Lernen

Nachdem im letzten Kapitel erklärt wurde, was machinelles Lernen überhaupt
ist, betrachten wir in diesem Kapitel die drei großen Kategorien von
ML-Modellen: überwachtes Lernen (Supervised Learning), unüberwachtes Lernen
(Unsupervised Learning) und verstärkendes Lernen (Reinforcement Learning).

### Lernziele Kapitel 1.2

* Sie können anhand eines Beispiels erklären, was die Fachbegriffe
  * **überwachtes Lernen (Supervised Learning)**,
  * **unüberwachtes Lernen (Unsupervised Learning)** und
  * **verstärkendes Lernen (Reinforcement Learning)** bedeuten.
* Sie können beim überwachten Lernen zwischen **Regression** und
  **Klassifikation** unterscheiden.

### Überwachtes Lernen (Supervised Learning)

Im letzten Kapitel haben wir im Video [»So lernen Maschinen:
Algorithmen«](https://youtu.be/HmUzceKCI9I) die Aufgabenstellung kennengelernt,
auf Fotos Hunde von Katzen zu unterscheiden. Diese Art von Problemstellung ist
typisch für **überwachtes Lernen**. Die Daten werden vorab gekennzeichnet, sie
erhalten ein **Label**. So lernen auch Kinder. Stellen Sie sich vor, in einem
Korb liegen Äpfel und Bananen und ein Kind soll den Unterschied erlernen. Jedes
Stück Obst wird aus dem Korb genommen und dem Kind gezeigt. Dazu sagen wir dann
entweder »Apfel« oder »Banane«. Das Kind hat also einen Lehrer oder Trainer. Mit
der Zeit wird das Kind zwischen beiden Obstsorten unterscheiden können.

**Was ist ... überwachtes Lernen?**

Überwachtes Lernen ist eine Kategorie des maschinellen Lernens. Beim überwachten
Lernen liegen die Daten als Eingabe- und Ausgabedaten mit Labels vor. Ein
maschineller Lernalgorithmus versucht ein Modell zu finden, das bestmöglich den
Eingabedaten die Ausgabedaten zuordnet.

Beim überwachten Lernen können die Prognosen des Modells für bekannte Daten mit
den korrekten Ergebnissen (Labels) verglichen werden. Das Modell wird also
überwacht.

Prinzipiell werden dabei wiederum zwei Arten von Labels unterschieden:

* kontinuierliche Labels und
* diskrete Labels.

Bei dem Beispiel mit den Hunde- und Katzenfotos sind die Labels diskret. Mit
**diskreten Labels** ist gemeint, dass nur wenige verschiedene Labels existieren.
In diesem Fall sind es genau zwei verschiedene Labels, nämlich zum einen das
Label »Hund« und zum anderen das Label »Katze«. Ein anderes Beispiel für
diskrete Labels sind die Schulnoten sehr gut, gut, befriedigend, ausreichend,
mangelhaft und ungenügend. Es gibt nur sechs verschiedene Noten, die eine
Schülerin oder ein Schüler in einem Test erreichen kann. Dabei müssen die
diskreten Labels keine Texte sein. Die Schulnoten könnten wir auch mit den Labels
1, 2, 3, 4, 5 und 6 kennzeichnen.

Bei den **kontinuierlichen Labels** gibt es sehr viele, normalerweise unendliche
viele verschiedene Labels. Textbezeichnungen sind dann nicht mehr sinnvoll, so
dass kontinuierliche Labels durch Zahlen repräsentiert werden. Ein Beispiel für
kontinuierliche Ausgabedaten ist der Verkaufspreis eines Autos abhängig vom
Kilometerstand. Normalerweise kosten Neuwagen mit einem Kilometerstand von 0 km
am meisten und der Preis sinkt, je mehr Kilometer das Auto bereits gefahren
wurde. Die Verkaufspreise könnte man nun als ganze Zahlen darstellen, wenn man
sie in ganzen Euros angibt, oder als Fließkommazahl, wenn der Preis auf den Cent
genau angegeben wird. Es gibt nicht unendlich viele Verkaufspreise, aber sehr
viele verschiedene mögliche Werte.

Viele ML-Modelle funktionieren sowohl für diskrete als auch kontinuierliche
Daten, aber nicht alle. Daher ist es notwendig, bereits zu Beginn zu
entscheiden, ob das Modell für diskrete oder kontinuierliche Ausgabedaten
eingesetzt werden soll.

Das überwachte Lernen wird daher wiederum in zwei Arten unterteilt:

* **Regression** für kontinuierliche Ausgabedaten und
* **Klassifikation**  für diskrete Ausgabedaten.

Auf beide Problemstellungen gehen die nächsten Videos ein.

<https://youtu.be/gaYYJAEt0zI?feature=shared>

#### Regression

**Was ist ... Regression?**

Regression ist das Teilgebiet des überwachten maschinellen Lernens, bei dem
Modelle den Zusammenhang zwischen Eingabedaten und *kontinuierlichen* Ausgabedaten
prognostizieren sollen.

<https://youtu.be/NCCctUdfA3E?feature=shared>

#### Klassifikation

**Was ist ... Klassifikation?**

Klassifikation ist das Teilgebiet des überwachten maschinellen Lernens, bei dem
Modelle den Zusammenhang zwischen Eingabedaten und *diskreten* Ausgabedaten
prognostizieren sollen.

<https://youtu.be/g6zuVEDlAzo?feature=shared>

### Unüberwachtes Lernen (Unsupervised Learning)

Beim überwachten Lernen liegen Eingabedaten und Ausgabedaten mit Labels vor. Die
Prognosen eines Modells können für bekannte Paare von Eingabe- und Ausgabedaten
überwacht werden. Das ist beim unüberwachten Lernen nicht der Fall. Beim
**unüberwachten Lernen (Unsupervised Learning)** gibt es keine Ausgabedaten,
also keine Labels. Stattdessen soll der maschinelle Lernalgorithmus eigenständig
Muster erlernen und Strukturen in den Daten finden.

**Was ist ... unüberwachtes Lernen (Unsupervised Learning)?**

Unüberwachtes Lernen ist ein Teilgebiet des maschinellen Lernens, bei dem ein
Algorithmus versucht, Muster und Strukturen in Daten zu finden. Dabei sind die
Daten nicht vorab in Eingabe- und Ausgabedaten aufgeteilt bzw. mit Labels
gekennzeichnet.

Ein Kind könnte auch selbstständig einen Obstkorb erkunden. Vielleicht würde das
Kind mit der Zeit lernen, dass es Obst gibt, das ihm schmeckt, wohingegen
anderes Obst dem Kind nicht schmeckt. Vielleicht würde das Kind das Obst auch in
großes Obst und kleines Obst unterteilen oder nach Farbe sortieren. Das Kind
gruppiert also Obst nach selbst gewählten Eigenschaften. Es bildet Cluster,
dementsprechend heißt dieser Vorgang **Clustering**.

<https://youtu.be/P2Qwc63iCVQ?feature=shared>

<https://youtu.be/yKcGVt3xfiE?feature=shared>

### Verstärkendes Lernen (Reinforcement Learning)

Wir schließen unsere Übersicht der maschinellen Lernverfahren mit dem
verstärkendem Lernen ab.

**Was ist ... verstärkendes Lernen (Reinforcement Learning)?**

 **Verstärkendes Lernen (Reinforcement Learning)** ist eine Art des maschinellen
Lernens, bei dem ein ML-Algorithmus durch versuch und Irrtum erlernt, was das
optimale Verhalten ist, um ein bestimmtes Ziel zu erreichen. Es werden Aktionen
ausgeführt und entweder bestraft oder belohnt, je nachdem, ob durch diese
Aktionen das Ziel besser oder schlechter erreicht wird.

Ein Beispiel aus dem Alltag für verstärkendes Lernen ist das Training eines
Haustieres, eines Hundes beispielsweise. Folgt der Hund dem Befehl »Sitz!«, so
erhält er ein Leckerli. Mit der Zeit wird der Hund auf das Kommando »Sitz!«
reagieren und sich setzen, auch wenn es nicht immer eine Belohnung dafür gibt.

Ein bekanntes Beispiel aus dem Bereich Künstliche Intelligenz für verstärkendes
Lernen sind Schachsysteme. Anfangs kennt das Schachsystem nur die grundlegenden
Schachregeln, aber keinerlei Strategie. Durch das Spielen vieler Spiele, wobei
der Computer bei jedem Sieg eine "Belohnung" erhält und bei jeder Niederlage
eine "Strafe", lernt das Schachsystem allmählich, welche Züge gewinnbringend
sind und welche eher zu Niederlagen führen. Nach Tausenden oder sogar Millionen
von Spielen kann das Schachsystem dann auf einem sehr hohen Niveau spielen -
alles durch verstärkendes Lernen.

<https://youtu.be/5HhQgFCQGIY?feature=shared>

<https://youtu.be/EAX12jlMlUw?feature=shared>

## 1.3 Technische Voraussetzungen

Für maschinelles Lernen ist **Python** die Programmiersprache der Wahl. Das
liegt vor allem auch daran, dass Google eine sehr wichtige ML-Bibliothek für
Python zur Verfügung stellt, die sogenannte Bibliothek
[TensorFlow](https://www.tensorflow.org). Glücklicherweise müssen wir die
Algorithmen nicht selbst in Python umsetzen, sondern können schon fertige
Modelle aus Bibliotheken wie beispielsweise
[scikit-learn](https://scikit-learn.org/stable/index.html) verwenden, die wir
dann noch an die Daten anpassen müssen. In diesem Kapitel werden die technischen
Voraussetzungen beschrieben, um maschinelles Lernen mit Python und den
sogenannten **Jupyter Notebooks** umzusetzen.

### Lernziele Kapitel 1.3

* Sie haben eine lauffähige **Python-Distribution** installiert.
* Sie können **JupyterLab** starten und ein **Jupyter Notebook** erzeugen.
* Sie kennen den prinzipiellen Aufbau eines Jupyter Notebooks mit
  **Markdown-Zellen** und **Code-Zellen**.

### Was sind Jupyter Notebooks?

Die Vorlesung wird in Form von **Jupyter Notebooks** zur Verfügung gestellt.
Jupyter Notebooks sind interaktive digitale Notizbücher, die sowohl Texte,
Bilder oder Videos enthalten können als auch Python-Code, der direkt im
Notizbuch ausführbar ist.

Die Kombination von Text, Python-Code und Visualisierungen macht Jupyter
Notebooks zu einem sehr leistungsstarken Werkzeug für die Datenanalyse. Daten
können direkt in die Jupyter Notebooks eingegeben oder importiert werden.
Fehlende Daten oder Ausreißer können direkt korrigiert werden, ohne dass mit
einer externen Software die Korrektur dokumentiert werden muss. Die Ergebnisse
der Analysen oder ML-Modelle können sofort im Jupyter Notebook dargestellt
werden, ohne dass eine externe Anwendung gestartet werden müssen. Daher sind sie
eine der bekanntesten Anwendungen im Bereich Data Science und werden oft zur
Datenanalyse, maschinellem Lernen und Visualisierung eingesetzt.

### Installation Python

Python wird in der Regel mit dem Betriebsystem ausgeliefert. Für maschinelles
Lernen benötigen wir jedoch weitere **Python-Module**, die die grundlegenden
Funktionalitäten von Python um ML-Funktionalitäten erweitern. Diese sind
normalerweise nicht vorinstalliert, sondern müssen nachinstalliert werden. Bevor
man sich die Module aus verschiedenen Internetquellen zusammensucht, ist es
einfacher, eine sogenannte Python-Distribution zu benutzen.

Eine **Distribution** ist eine Zusammenstellung von Software oder
Bibliotheken/Modulen. Die Firma Anaconda, Inc. wurde 2012 mit dem Ziel
gegründet, Python in Unternehmen speziell für die Geschäftsfeldanalyse (Business
Analytics) einzuführen, was die Open Source Community so nicht leisten konnte.
Daher enthält die Python-Distribution [Anaconda](https://www.anaconda.com) eine
Reihe von nützlichen Paketen und Bibliotheken für wissenschaftliche
Berechnungen, Datenanalyse, maschinelles Lernen und andere Anwendungen. Da sie
eine Benutzeroberfläche beinhaltet, mit der die Python-Bibliotheken verwaltet
werden, ist sie auch gerade für Einsteiger in Python eine gute Wahl.

Die Python-Distribution Anaconda gibt es in verschiedenen Editionen mit
entsprechenden Preismodellen. Für diese Vorlesung ist die sogenannte »Individual
Edition« ausreichend, die von Anaconda kostenlos zur Verfügung gestellt wird.

Hier ist eine Schritt-für-Schritt-Anleitung zum Installieren von Python mit der
Distribution Anaconda für Windows und MacOS:

1. Öffnen Sie die offizielle Anaconda-Website unter
   <https://www.anaconda.com/products/individual> und laden Sie die neueste
   Version von Anaconda für Ihr Betriebssystem herunter.
2. Führen Sie die Installationsdatei aus und folgen Sie den Anweisungen auf dem
   Bildschirm.
3. Öffnen Sie nach der Installation das Anaconda-Navigator-Programm, das im
   Startmenü oder Launchpad verfügbar sein sollte.

### Mit welcher App wird ein Jupyter Notebook bearbeitet?

Es gibt mehrere Applikationen, die Jupyter Notebooks bearbeiten können. Am
bekanntesten ist sicherlich [JupyterLab](https://jupyter.org), das wir auch in
dieser Vorlesung verwenden. Neben JupyterLab gibt es aber auch weitere
Möglichkeiten, um Jupyter Notebooks zu bearbeiten.

Die beiden Entwicklungsumgebungen

* [PyCharm](https://www.jetbrains.com/help/pycharm/jupyter-notebook-support.html)
* [Microsoft Visual Studio Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

ermöglichen ebenfalls die direkte Bearbeitung von Jupyter Notebooks. Auch
zahlreiche Cloudanbieter bieten direkt das Bearbeiten und Ausführen von Jupyter
Notebooks an, z.B.

* [Google Colab](https://colab.research.google.com/notebook)
* [Microsoft Azure](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-run-jupyter-notebooks)
* [Deepnote](https://deepnote.com)
* [replit](https://replit.com/template/jupyter-notebook)

Wie bei allen Clouddiensten sollte man sich jedoch eingehend mit den
Datenschutzbestimmungen des Anbieters vertraut machen, bevor man den Dienst in
Anspruch nimmt. Aufgrund des Datenschutzes empfehle ich stets, Python/Anaconda
lokal zu installieren.

### Start von JupyterLab und das erste Jupyter Notebook

Anaconda installiert JupyterLab automatisch mit, so dass wir direkt loslegen
können. Sollte es Probleme geben, finden Sie hier die [Dokumentation von
JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html).

JupyterLab startet im Hintergrund einen sogenannten Jupyter-Kernel, der die
interaktiven Python-Code-Zellen ausführt. Der Client ist in der Regel der
Standard-Browser.

Die folgende Schritt-für-Schritt-Anleitung zeigt, wie ein neues Jupyter Notebook
in JupyterLab erstellt wird.

1. Um ein neues Jupyter Notebook zu erstellen, klicken Sie auf "Home" im
   Anaconda-Navigator und wählen "JupyterLab" aus. Alternativ können Sie
   JupyterLab auch mit dem Befehl "jupyter-lab" aus einem Terminal oder einer
   Konsole starten (Linux oder MacOS).
2. Wählen Sie "Python 3 (ipykernel)" aus, um ein neues Notebook zu erstellen.
3. Sie können jetzt Python-Code in dem Notebook schreiben und ausführen. Wenn
   Sie zusätzliche Pakete benötigen, können Sie diese über den
   "Environments"-Tab im Anaconda-Navigator installieren.

![Screenshot JupyterLab](https://gramschs.github.io/book_ml4ing/_images/fig_chap01_sec02_jupyterlab.png)

### Grundlegende Funktionalitäten von Jupyter Notebooks

Ein Jupyter Notebook besteht aus einer Abfolge von Zellen, in denen Text, Code
und Visualisierungen eingebettet werden. Die Zellen können entweder in der
Programmiersprache Python oder in einer Reihe anderer Programmiersprachen wie R,
Julia oder JavaScript geschrieben werden. Erkennbar sind Jupyter Notebooks an
der Dateiendung `ipynb`, die die Abkürzung für »**i**ntelligentes **Py**thon
**N**ote**b**ook« darstellt.

![Screenshot eines Jupyter Notebooks mit einer nicht ausgeführten Markdown-Zelle (1), einer ausgeführten Code-Zelle (2) und dem Run-Button (3)](https://gramschs.github.io/book_ml4ing/_images/fig_chap01_sec03_zellen.png)

Wie im obigen Screenshot zu sehen, sind Zellen mit einem Rahmen versehen. Eine
Zelle kann entweder eine Text-Zelle (1) oder eine Code-Zelle (2) sein. In
Text-Zellen wird die sogenannte
[Markdown-Formatierung](https://de.wikipedia.org/wiki/Markdown) benutzt, weshalb
sie **Markdown-Zellen** genannt werden. Bei dieser Art, Text zu formatieren,
werden Textzeichen benutzt anstatt auf einen Button zu klicken. Um
beispielsweise ein Wort fettgedruckt anzuzeigen, werden zwei Sternchen `**` vor
und hinter das Wort gesetzt, also ich bin `**fett**` gedruckt.

In **Code-Zellen** (2) können Sie direkt Python-Code eingeben. Sie erkennen eine
Code-Zelle daran, dass eckige Klammern links daneben stehen. Eine Code-Zelle
wird ausgeführt, indem Sie auf "Run" klicken. Der Run-Button verbirgt sich
hinter dem kleinen nach rechts gerichteten Dreick in der Menü-Leiste des Jupyter
Notebooks (3). Danach erscheint die Ausgabe, die der Python-Interpreter ggf.
produziert. Wird eine Code-Zelle ausgeführt, so erscheint eine Zahl in den
eckigen Klammern. Diese Zahl zeigt die Reihenfolge an, in der Code-Zellen
ausgeführt wurden.

## Übungen

### Übung 1.1

Installieren Sie Anaconda und Python auf Ihrem PC oder Laptop. Starten Sie die
Software JupyterLab. Laden Sie dieses Jupyter Notebook herunter und öffnen
Sie es mit der Software JupyterLab.

### Übung 1.2

Benutzen Sie Python als Taschenrechner. Fügen Sie dazu diesem Jupyter Notebook eine Code-Zelle hinzu und lassen Sie die folgenden Ausdrücke berechnen:

* 2 + 3
* 2 - 3
* 4 * 5
* 16 / 4
* 16 / 3
* 5**2

### Übung 1.3

In der vorhergehenden Aufgabe haben Sie den Ausdruck `5**2` berechnen lassen.
Fügen Sie jetzt eine Markdown-Zelle ein. Schreiben Sie auf, was Ihrer Vermutung
nach der `**`-Operator für eine Bedeutung hat. Recherchieren Sie anschließend im
Internet, nach der Bedeutung und vergleichen Sie Ihre Antwort mit der Recherche.
Fügen Sie die Internetseite als URL in die Markdown-Zelle ein.

### Übung 1.4

Speichern Sie das bearbeitete Jupyter Notebook unter einem anderen Namen ab.
Exportieren Sie es als HTML-Dokument.
