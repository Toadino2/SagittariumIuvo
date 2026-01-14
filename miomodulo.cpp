#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <array>
#include <cmath>
#include <thread>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/lognormal.hpp>


namespace py = pybind11;
using namespace boost::math;


constexpr double pigreco = 3.141592653589793;
constexpr double duepigreco = 6.283185307179586;
constexpr double logduepigreco = 1.8378770664093453;


struct CoseTest {
    double statistica;
    bool accettazione;
    double pvalue;
};


// Funzione per calcolare i due insiemi di autocorrelazioni angolari. Esistono tante versioni circolari
// del coefficiente di correlazione; qui utilizzeremo un coefficiente di correlazione a rango
// circolare-circolare, descritto a pagina 250 di Mardia, Jupp (2000). In particolare, cerchiamo
// le autocorrelazioni tra il nostro dataset e la sua versione ritardata di un passo. Queste si dividono
// in due, ossia le autocorrelazioni positive, e le autocorrelazioni negative: l'idea di base è che
// le due variabili sono correlate se le osservazioni seguono un andamento simile, e quindi si muovono
// verso sinistra o destra insieme o in modo esattamente opposto (correlazione negativa).
// La funzione deve ricevere una lista con gli angoli in radianti, e la sua lunghezza;
// restituisce una coppia di vettori, uno con le autocorrelazioni positive e uno con quelle negative.
std::array<std::vector<double>, 2> autocorrelazioniangolari(const std::vector<double> &angoli,
    const int taglia){
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (angoli.size() != taglia){throw std::runtime_error("Il dataset non corrisponde alla taglia");}
    // Inizializziamo due contenitori vuoti per le autocorrelazioni; essi conterranno n-1 elementi,
    // ossia quanto il numero massimo possibile di ritardi analizzabili.
    std::vector<double> autocorrelazionipositive(taglia-1);
    std::vector<double> autocorrelazioninegative(taglia-1);
    // Esaminiamo uno per uno tutti i ritardi k da 1 a n-1.
    for (int k = 1; k <= taglia - 1; k++){
        // Una volta fissato k, costruiamo dal nostro dal nostro dataset le due serie storiche
        // di angoli di cui vogliamo calcolare le correlazioni, che saranno appunto le
        // autocorrelazioni a ritardo k. Tali serie semplicemente saranno i primi n-k angoli,
        // e gli ultimi n-k. Associamo a ogni angolo la sua posizione nella serie.
        std::vector<std::pair<double, int>> angolidopoordinati(taglia-k);
        std::vector<std::pair<double, int>> angoliprimaordinati(taglia-k);
        int indice = 0;
        for (int i = k; i < taglia; i++){
            angolidopoordinati[indice] = {angoli[i], indice};
            indice++;
        }
        indice = 0;
        for (int i = 0; i < taglia-k; i++){
            angoliprimaordinati[indice] = {angoli[i], indice};
            indice++;
        }
        // Adesso dobbiamo associare a ogni angolo il suo rango nella serie, ossia la posizione che occupa
        // quando la serie viene ordinata. Per farlo, ordiniamo le due serie in ordine crescente di angoli,
        // in modo che la posizione di ogni angolo corrisponda al suo rango, mantenendogli comunque
        // associata la sua posizione originale. Dopo di che, creiamo due liste di ranghi. Per riempirle,
        // inseriamo i numeri da 1 a n-k in modo che il numero i si trovi alla stessa posizione
        // in cui si trovava, nella serie originaria non ordinata, l'angolo i-esimo della serie ordinata
        // (così che per esempio da una serie {{dato1, 2}, {dato2, 3}, {dato3, 1}} si ottenga {3, 1, 2}).
        // Se nella serie ordinata troviamo un angolo uguale al precedente, gli assegniamo lo stesso
        // rango di quest'ultimo, e poi saltiamo avanti di un numero.
        std::sort(angolidopoordinati.begin(), angolidopoordinati.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        std::sort(angoliprimaordinati.begin(), angoliprimaordinati.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        std::vector<int> ranghidopo(taglia-k);
        for (int i = 0; i < taglia - k; i++){
            if (i > 0){
                if (angolidopoordinati[i].first == angolidopoordinati[i - 1].first) {
                    ranghidopo[angolidopoordinati[i].second] = ranghidopo[angolidopoordinati[i - 1].second];
                } else {
                    ranghidopo[angolidopoordinati[i].second] = i+1;
                }
            } else {
                ranghidopo[angolidopoordinati[i].second] = i+1;
            }
        }
        std::vector<int> ranghiprima(taglia-k);
        for (int i = 0; i < taglia - k; i++){
            if (i > 0) {
                if (angoliprimaordinati[i].first == angoliprimaordinati[i-1].first) {
                    ranghiprima[angoliprimaordinati[i].second] = ranghiprima[angoliprimaordinati[i-1].second];
                } else {
                    ranghiprima[angoliprimaordinati[i].second] = i+1;                }
            } else {
                ranghiprima[angoliprimaordinati[i].second] = i+1;
            }
        }
        // Adesso possiamo calcolare \beta_i=2\pi r_i/n e \gamma_i=2\pi s_i/n, dove r_i sono i ranghi
        // della prima serie e s_i quelli della seconda.
        std::vector<double> beta(taglia - k);
        std::vector<double> gamma(taglia - k);
        for (int i = 0; i < taglia - k; i++){
            beta[i] = 2.0*pigreco*ranghiprima[i]/taglia;
            gamma[i] = 2.0*pigreco*ranghidopo[i]/taglia;
        }
        // Ora possiamo finalmente calcolare le due autocorrelazioni, seguendo la formula
        // di Mardia e Jupp.
        double sommacoseni = 0.0;
        double sommaseni = 0.0;
        for (int i = 0; i < taglia - k; i++) {
            sommacoseni += std::cos(beta[i]-gamma[i]);
            sommaseni += std::sin(beta[i]-gamma[i]);
        }
        const double rpositivo = sommacoseni*sommacoseni+sommaseni*sommaseni;
        sommacoseni = 0.0;
        sommaseni = 0.0;
        for (int i = 0; i < taglia - k; i++){
            sommacoseni += std::cos(beta[i]+gamma[i]);
            sommaseni += std::sin(beta[i]+gamma[i]);
        }
        const double rnegativo = sommacoseni*sommacoseni+sommaseni*sommaseni;
        autocorrelazionipositive[k - 1] = rpositivo/(taglia*taglia);
        autocorrelazioninegative[k - 1] = rnegativo/(taglia*taglia);
    }
    return {autocorrelazionipositive, autocorrelazioninegative};
}
// Forse va ricontrollato.


// Funzione per eseguire il test di Hotelling su un dataset bivariato. L'allenamento deve essere
// passato come una lista di ascisse e una di ordinate, e la soglia deve essere pari al quantile 1-alfa
// di una distribuzione F con 2 e n-2 gradi di libertà. Viene restituito true se l'ipotesi nulla
// viene accettata, altrimenti false.
CoseTest testhotelling(const int taglia, const std::vector<double> &ascisse, const std::vector<double> &ordinate,
    const double alfa){
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (ascisse.size() != taglia || ordinate.size() != taglia){throw std::runtime_error("Il dataset non corrisponde alla taglia");}
    // Calcoliamo il vettore medio e la matrice di covarianze dei dati.
	double mediaascisse = 0.0;
    double mediaordinate = 0.0;
	for (int i = 0; i < taglia; i++){
	    mediaascisse += ascisse[i];
	    mediaordinate += ordinate[i];
	}
	mediaascisse /= taglia;
	mediaordinate /= taglia;
	double varianzaascisse = 0.0;
    double varianzaordinate = 0.0;
    double covarianza = 0.0;
	for (int i = 0; i < taglia; i++){
	    const double differenzaascisse = ascisse[i]-mediaascisse;
	    const double differenzaordinate = ordinate[i]-mediaordinate;
	    varianzaascisse += differenzaascisse*differenzaascisse;
	    varianzaordinate += differenzaordinate*differenzaordinate;
	    covarianza += differenzaascisse*differenzaordinate;
	}
	varianzaascisse /= taglia - 1;
	varianzaordinate /= taglia - 1;
	covarianza /= taglia - 1;
    // Per calcolare la statistica test, abbiamo bisogno dell'inversa della matrice di covarianze;
    // poiché è una matrice 2x2, esiste un'espressione in forma chiusa, che contiene il determinante:
    // in prevenzione del caso in cui la matrice sia singolare, usiamo la regolarizzazione di Tikhonov.
	double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
	if (determinante < 0.000001){
	    varianzaascisse += 0.000001;
	    varianzaordinate += 0.000001;
	    determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
	}
	const double inversa11 = varianzaordinate/determinante;
	const double diagonale = -covarianza/determinante;
	const double inversa22 = varianzaascisse/determinante;
    // Adesso possiamo calcolare la statistica, che poiché \mu=[0 0], è
    // t^2=n\bar{x}^T\Sigma^{-1}\bar{x}. L'ipotesi nulla è rifiutata se
    // (n-p)/(p(n-1))*t^2 è maggiore della soglia, dove qui p=2.
    const double primoprodotto1 = mediaascisse*inversa11+mediaordinate*diagonale;
    const double primoprodotto2 = mediaascisse*diagonale+mediaordinate*inversa22;
    const double secondoprodotto = primoprodotto1*mediaascisse+primoprodotto2*mediaordinate;
    const double statistica = (taglia - 2.0)/(2.0*taglia - 2.0)*taglia*secondoprodotto;
    CoseTest cose;
    fisher_f_distribution<> distribuzione(2.0, taglia-2.0);
    cose.statistica = statistica;
    cose.pvalue = 1.0-cdf(distribuzione, statistica);
    cose.accettazione = cose.pvalue >= alfa;
    return cose;
}


// Funzione che esegue effettivamente i calcoli per trovare la probabilità di errore di II tipo
// per il test di Hotelling, da utilizzare in threading. Gli argomenti da passare sono la dimensione n
// del campione, il quantile 1-alfa di una distribuzione F (2, n-2), distanzabeta, varasc, varord e il
// numero di iterazioni. Poiché va valutata la probabilità che l'ipotesi nulla sia accettata quando è
// vera l'alternativa, si presuppone che sotto l'ipotesi alternativa la distribuzione dei dati
// sia N([0, distanzabeta], [[varasc, 0], [0, varord]]. Restituisce il numero di volte in cui
// l'ipotesi nulla viene accettata.
int calcolobetatesthotelling(const int taglia, const double soglia, const double distanzabeta,
                             const double varasc, const double varord, const int volteperthread){
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (varasc <= 0.0 || varord <= 0.0){throw std::runtime_error("Varianze non valide");}
    // Inizializziamo il contatore delle volte in cui l'ipotesi nulla viene accettata.
    int accettazioni = 0;
    // Prepariamo i generatori di campioni casuali sotto l'ipotesi alternativa.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> distribuzione1(0.0, std::sqrt(varasc));
    std::normal_distribution<> distribuzione2(distanzabeta, std::sqrt(varord));
    // Per ogni iterazione:
    for (int i = 0; i < volteperthread; i++){
        // Estraiamo un campione casuale sotto l'ipotesi alternativa. Poiché assumiamo che la covarianza
        // sia nulla, è sufficente estrarre da due variabili aleatorie indipendenti per ottenere
        // quella bivariata.
        std::vector<double> ascisse(taglia);
        std::vector<double> ordinate(taglia);
        for (int j = 0; j < taglia; j++){
            ascisse[j] = distribuzione1(generatore);
            ordinate[j] = distribuzione2(generatore);
        }
        // Calcoliamo il vettore medio e la matrice di covarianze del campione.
        double mediaascisse = 0.0;
        double mediaordinate = 0.0;
        for (int j = 0; j < taglia; j++){
            mediaascisse += ascisse[j];
            mediaordinate += ordinate[j];
        }
        mediaascisse /= taglia;
        mediaordinate/= taglia;
        double varianzaascisse = 0.0;
        double varianzaordinate = 0.0;
        double covarianza = 0.0;
        for (int j = 0; j < taglia; j++){
            const double differenzaascisse = ascisse[j]-mediaascisse;
            const double differenzaordinate = ordinate[j]-mediaordinate;
            varianzaascisse += differenzaascisse*differenzaascisse;
            varianzaordinate += differenzaordinate*differenzaordinate;
            covarianza += differenzaascisse*differenzaordinate;
        }
        varianzaascisse /= taglia - 1;
        varianzaordinate /= taglia - 1;
        covarianza /= taglia - 1;
        // Ripetiamo il processo per calcolare la statistica test di Hotelling.
        // Se questa è minore del valore critico soglia, l'ipotesi nulla viene accettata,
        // e aumentiamo il contatore.
        double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        if (determinante < 0.000001) {
            varianzaascisse += 0.000001;
            varianzaordinate += 0.000001;
            determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        }
        const double inversa11 = varianzaordinate/determinante;
        const double diagonale = -covarianza/determinante;
        const double inversa22 = varianzaascisse/determinante;
        const double primoprodotto1 = mediaascisse*inversa11+mediaordinate*diagonale;
        const double primoprodotto2 = mediaascisse*diagonale+mediaordinate*inversa22;
        const double secondoprodotto = primoprodotto1*mediaascisse+primoprodotto2*mediaordinate;
        const double statistica = (taglia - 2.0)/(2.0*taglia - 2.0)*taglia*secondoprodotto;
        if (statistica <= soglia){
            accettazioni++;
        }
    }
    return accettazioni;
}


// Funzione preparatoria per calcolare la probabilità di errore di II tipo del test di Hotelling.
double betatesthotelling(const int taglia, const double soglia, const int volte, const double distanzabeta,
                         const double varianzaascisse, const double varianzaordinate){
    if (volte < 0){throw std::runtime_error("Le iterazioni devono essere positive");}
    // Viene ottenuto il numero di thread possibili; se non ci si riesce, ne si mettono quattro.
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    // Calcoliamo quante delle iterazioni indicate fa ciascun thread.
    const int volteperthread = volte/numerothread;
    const int iterazionirestanti = volte%numerothread;
    // Inizializziamo un contatore di volte in cui l'ipotesi nulla è accettata.
    std::vector<int> accettazionithread(numerothread);
    // Tutti i thread fanno volteperthread iterazioni, e aggiungono il loro risultato ad accettazioni.
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread]() {
            accettazionithread[i] = calcolobetatesthotelling(taglia, soglia, distanzabeta, varianzaascisse, varianzaordinate,
                                                     volteperthread);
        });
    }
    // Solo l'ultimo thread fa qualche iterazione in più.
    thread[numerothread - 1] = std::thread([=, &accettazionithread](){
        accettazionithread[numerothread-1] = calcolobetatesthotelling(taglia, soglia, distanzabeta, varianzaascisse, varianzaordinate,
                                                 volteperthread+iterazionirestanti);
    });
    // Aspettiamo che tutti i thread finiscano.
    for (auto& t : thread){
        t.join();
    }
    // La probabilità di errore di II tipo è il numero totale di volte in cui l'ipotesi nulla
    // è stata accettata fratto il numero di tentativi.
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return static_cast<double>(accettazioni)/volte;
}


// NO, HO TOLTO LA STRONZATA DEI TEST MULTIPLI
// Funzione per eseguire il test di Ljung-Box. Deve essere passata una lista di autocorrelazioni,
// la lista dei numeri di ritardi h che si vogliono esaminare, la dimensione n del dataset,
// la lista dei quantili 1-alfa delle chi quadro con gradi di libertà pari ai cutoff, e il numero di cutoff.
// Restituisce true se l'ipotesi nulla è accettata e false altrimenti.
CoseTest ljungbox(const std::vector<double>& autocorrelazioni, const int n, const int h, const double alfa){
    if (n < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (autocorrelazioni.size() >= n){throw std::runtime_error("Troppe autocorrelazioni");}
    // Viene calcolata per ogni cutoff la statistica di Ljung-Box. Se questa supera la soglia critica
    // corrispondente al cutoff, l'ipotesi nulla viene rifiutata; se non accade per nessun cutoff,
    // viene accettata.
    double somma = 0.0;
    for (int i = 0; i < h; i++) {
        const double autocorrelazione = autocorrelazioni[i];
        somma += autocorrelazione*autocorrelazione/(n-i-1);
    }
    CoseTest cose;
    chi_squared_distribution<> distribuzione(h);
    cose.statistica = n*(n+2)*somma;
    cose.pvalue = 1.0-cdf(distribuzione, statistica);
    cose.accettazione = cose.pvalue >= alfa;
    return cose;
}


// Funzione che esegue effettivamente i calcoli per trovare la probabilità di errore di II tipo
// del test di Ljung-Box. Come nella funzione precedente, vanno passati la dimensione del campione,
// i cutoff dei ritardi da utilizzare, i quantili 1-alfa delle chi quadro con gradi di libertà
// corrispondenti, il numero di cutoff e il numero di iterazioni Monte Carlo da compiere.
// Viene restituito il numero di volte in cui l'ipotesi nulla viene accettata.
int calcolobetaljungbox(const int taglia, const int h, const double soglia, const int iterazioni){
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    // Inizializziamo un contatore delle volte in cui l'ipotesi nulla è accettata.
    int accettazioni = 0;
    // Prepariamo la generazione dei campioni casuali.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normale(0.0, 1.0);
    // Per ogni iterazione:
    for (int i = 0; i < iterazioni; i++){
        // Assumiamo che sotto l'ipotesi alternativa i dati seguano un processo AR(1),
        // con \sigma^2=1 e \phi_1=0.5. Al contempo, calcoliamo la media campionaria del processo.
        std::vector<double> serie(taglia);
        double valore = normale(generatore);
        serie[0] = valore;
        double media = valore;
        for (int j = 1; j < taglia; j++){
            valore *= 0.5;
            valore += normale(generatore);
            serie[j] = valore;
            media += valore;
        }
        media /= taglia;
        // A questo punto possiamo calcolare le autocorrelazioni.
        double varianzanondivisa = 0.0;
        for (int j = 0; j < taglia; j++){
            const double addendo = serie[j]-media;
            varianzanondivisa += addendo*addendo;
        }
        if (varianzanondivisa < 0.000001){varianzanondivisa += 0.000001;}
        double somma = 0.0;
        for (int k = 1; k <= h; k++){
            double sommatoria = 0.0;
            for (int t = 0; t < taglia-k; t++){
                sommatoria += (serie[t]-media)*(serie[t+k]-media);
            }
            somma += (sommatoria/varianzanondivisa)/(taglia-k);
        }
        if (taglia*(taglia+2)*somma <= soglia){accettazioni++;}
    }
    return accettazioni;
}


// Funzione preparatoria per calcolare la probabilità di errore di II tipo del test di Ljung-Box.
double betaljungbox(const int taglia, const int h, const double soglia, const int iterazioni){
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (iterazioni < 0){throw std::runtime_error("Le iterazioni devono essere positive");}
    // Otteniamo il massimo numero di thread possibile; se non ci riusciamo, impostiamolo a quattro.
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    // Contiamo quante delle iterazioni deve fare ciascun thread.
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    // Inizializziamo il contatore delle volte in cui l'ipotesi nulla è accettata.
    std::vector<int> accettazionithread(numerothread);
    // Tutti i thread tranne l'ultimo fanno volteperthread iterazioni; all'ultimo ne viene
    // assegnata qualcuna in più.
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread]() {
            accettazionithread[i] = calcolobetaljungbox(taglia, h, soglia, volteperthread);
        });
    }
    thread[numerothread - 1] = std::thread([=, &accettazionithread]() {
        accettazionithread[numerothread-1] = calcolobetaljungbox(taglia, h, soglia, volteperthread+iterazionirestanti);
    });
    // Aspettiamo che tutti i thread finiscano e poi restituiamo beta.
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return static_cast<double>(accettazioni)/iterazioni;
}


// Funzione che calcola effettivamente l'effettiva probabilità di errore di I tipo del test di Ljung-Box.
// Va passata la dimensione n del campione, la lista degli h da utilizzare, i quantili 1-alfa delle
// chi quadro con i corrispondenti gradi di libertà, il numero degli h, e quante iterazioni fare.
int calcoloalfaveroljungbox(const int taglia, const int h, const double soglia, const int iterazioni){
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (cutoff.size() != numerocutoff || cutoff.size() != soglie.size()){throw std::runtime_error("Numero di cutoff sbagliato");}
    // Inizializziamo un contatore del numero di volte in cui l'ipotesi nulla è accettata.
    int accettazioni = 0;
    // Prepariamo il generatore di campioni casuali.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normale(0.0, 1.0);
    // Per ogni iterazione:
    for (int i = 0; i < iterazioni; i++){
        // Estraiamo un campione dall'ipotesi nulla, ossia da normale standard senza introdurre
        // dipendenze temporali; contemporaneamente calcoliamone la media.
        std::vector<double> campione(taglia);
        double media = 0.0;
        for (int j = 0; j < taglia; j++) {
            const double valore = normale(generatore);
            campione[j] = valore;
            media += valore;
        }
        // Ora calcoliamo le autocorrelazioni.
        double varianzanondivisa = 0.0;
        for (int j = 0; j < taglia; j++){
            const double differenza = campione[j]-media;
            varianzanondivisa += differenza*differenza;
        }
        if (varianzanondivisa < 0.000001){varianzanondivisa += 0.000001;}
        double somma = 0.0;
        for (int k = 1; k <= h; k++){
            double sommatoria = 0.0;
            for (int t = 0; t < taglia-k; t++){
                sommatoria += (campione[t]-media)*(campione[t+k]-media);
            }
            somma += (sommatoria/varianzanondivisa)/(taglia-k);
        }
        if (taglia*(taglia+2)*somma <= soglia){accettazioni++;}
    }
    return accettazioni;
}


// Funzione preparatoria per calcolare l'effettiva probabilità di errore di I tipo del test di Ljung-Box.
double alfaveroljungbox(const int taglia, const int h, const double soglia, const int iterazioni){
    if (iterazioni < 0){throw std::runtime_error("Le iterazioni devono essere positive");}
    // Contiamo il numero massimo di thread possibili; se non ci riusciamo, impostiamolo a quattro.
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    // Contiamo quante iterazioni deve fare ciascun thread.
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    // Inizializziamo il contatore di volte in cui l'ipotesi nulla è accettata.
    std::vector<int> accettazionithread(numerothread);
    // Tutti i thread tranne l'ultimo fanno volteperthread iterazioni; solo l'ultimo aggiunge le rimanenti.
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcoloalfaveroljungbox(taglia, h, soglia, volteperthread);
        });
    }
    thread[numerothread - 1] = std::thread([=, &accettazionithread](){
        accettazionithread[numerothread-1] = calcoloalfaveroljungbox(taglia, h, soglia, volteperthread+iterazionirestanti);
    });
    // Aspettiamo che tutti i thread finiscano e poi restituiamo la probabilità di errore di I tipo
    // (ossia la proporzione di volte in cui l'ipotesi nulla è rifiutata nonostante sia vera).
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return 1.0-static_cast<double>(accettazioni)/iterazioni;
}


// Funzione che calcola la statistica test di Henze-Zirkler di un campione per testare l'ipotesi nulla
// che venga da una normale multivariata, più la media e la varianza della sua distribuzione sotto
// l'ipotesi nulla. Devono essere passate una lista di ascisse, una lista di ordinate
// e la dimensione del campione n.
// Adattato dallo script Python per pingouin.multivariate_normality.
CoseTest testhenzezirkler(const std::vector<double>& ascisse, const std::vector<double> &ordinate,
    const double alfa, const int taglia){
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (ascisse.size() != taglia || ordinate.size() != taglia){throw std::runtime_error("Il dataset non corrisponde alla taglia");}
    // Per prima cosa, ci servono due oggetti: il primo è l'inversa della matrice di covarianze
    // campionaria del campione, e la matrice dei dati centrati. Per evitare problemi di matrici
    // singolari, usiamo la regolarizzazione di Tikhonov (questo differisce rispetto all'algoritmo di
    // Python, che risolve il problema usando l'inversa di Penrose).
    double mediaascisse = 0.0;
    double mediaordinate = 0.0;
    for (int i = 0; i < taglia; i++){
        mediaascisse += ascisse[i];
        mediaordinate += ordinate[i];
    }
    mediaascisse /= taglia;
    mediaordinate /= taglia;
    std::vector<double> ascissecentrate(taglia);
    std::vector<double> ordinatecentrate(taglia);
    double varianzaascisse = 0.0;
    double varianzaordinate = 0.0;
    double covarianza = 0.0;
    for (int i = 0; i < taglia; i++){
        const double differenzaascisse = ascisse[i]-mediaascisse;
        const double differenzaordinate = ordinate[i]-mediaordinate;
        ascissecentrate[i] = differenzaascisse;
        ordinatecentrate[i] = differenzaordinate;
        varianzaascisse += differenzaascisse*differenzaascisse;
        varianzaordinate += differenzaordinate*differenzaordinate;
        covarianza += differenzaascisse*differenzaordinate;
    }
    varianzaascisse /= taglia;
    varianzaordinate /= taglia;
    covarianza /= taglia;
    const double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
    double hz;
    double lisciatore;
    double lisciatore2;
    if (determinante < 0.000001){
        hz = 4.0*taglia;
        lisciatore = 1.0/std::sqrt(2.0)*std::pow(5.0/4.0, 1.0/6.0)*std::pow(static_cast<double>(taglia), 1.0/6.0);
        lisciatore2 = lisciatore*lisciatore;
    } else {
        double inversa11 = varianzaordinate/determinante;
        double inversa12 = -covarianza/determinante;
        double inversa22 = varianzaascisse/determinante;
        // Adesso calcoliamo il prodotto matriciale tra la matrice dei dati centrati e la matrice
        // di covarianze. Immagazziniamo la prima e la seconda colonna della matrice risultante
        // come due vettori separati.
        std::vector<double> primoprodotto1(taglia);
        std::vector<double> primoprodotto2(taglia);
        for (int i = 0; i < taglia; i++){
            primoprodotto1[i] = ascissecentrate[i]*inversa11+ordinatecentrate[i]*inversa12;
            primoprodotto2[i] = ascissecentrate[i]*inversa12+ordinatecentrate[i]*inversa22;
        }
        // Adesso otteniamo la matrice Dj come la matrice diagonale del prodotto fatto prima
        // moltiplicato per la matrice dei dati centrati trasposta (di fatto completando il prodotto
        // \bar{x}\Sigma^{-1}\bar{x}^T).
        std::vector<double> Dj(taglia);
        for (int i = 0; i < taglia; i++){
            Dj[i] = primoprodotto1[i]*ascissecentrate[i]+primoprodotto2[i]*ordinatecentrate[i];
        }
        // Adesso, in modo simile a prima, otteniamo la matrice Y=x\Sigma^{-1}x^T.
        std::vector<double> primoprodottodecentrato1(taglia);
        std::vector<double> primoprodottodecentrato2(taglia);
        for (int i = 0; i < taglia; i++){
            primoprodottodecentrato1[i] = ascisse[i]*inversa11+ordinate[i]*inversa12;
            primoprodottodecentrato2[i] = ascisse[i]*inversa12+ordinate[i]*inversa22;
        }
        std::vector<std::vector<double>> Y(taglia, std::vector<double>(taglia));
        for (int i = 0; i < taglia; i++){
            for (int j = 0; j < taglia; j++){
                Y[i][j] = primoprodottodecentrato1[j]*ascisse[i]+primoprodottodecentrato2[j]*ordinate[i];
            }
        }
        // Estraiamo anche di Y la matrice diagonale.
        std::vector<double> Y_diag(taglia);
        for (int i = 0; i < taglia; i++){
            Y_diag[i] = Y[i][i];
        }
        // Adesso calcoliamo la matrice Djk che contiene le distanze di Mahalanobis al quadrato.
        std::vector<std::vector<double>> Djk(taglia, std::vector<double>(taglia));
        for (int i = 0; i < taglia; i++){
            for (int j = 0; j < taglia; j++){
                Djk[i][j] = -2.0*Y[j][i]+Y_diag[i]+Y_diag[j];
            }
        }
        // Parametro di smoothing.
        lisciatore = 1.0/std::sqrt(2.0)*std::pow(5.0/4.0, 1.0/6.0)*std::pow(static_cast<double>(taglia), 1.0/6.0);
        lisciatore2 = lisciatore*lisciatore;
        // Con questi passaggi calcoliamo la statistica test di Henze-Zirkler.
        double sommakernel = 0.0;
        double sommakernel2 = 0.0;
        for (int i = 0; i < taglia; i++){
            for (int j = 0; j < taglia; j++){
                sommakernel += std::exp(-lisciatore2/2.0*Djk[i][j]);
            }
            sommakernel2 += std::exp(-lisciatore2/(2.0+2.0*lisciatore2)*Dj[i]);
        }
        hz = sommakernel/taglia-2.0*sommakernel2/(1.0+lisciatore2)+taglia/(1.0+2.0*lisciatore2);
    }
    // Adesso calcoliamo la media e la varianza della log-normale.
    const double wb = (1.0+lisciatore2)*(1.0+3.0*lisciatore2);
    const double a = 1.0+2.0*lisciatore2;
    const double lisciatore4 = lisciatore2*lisciatore2;
    const double a2 = a*a;
    const double mu = 1.0-(1.0+2.0*lisciatore2/a+8.0*lisciatore4/(2.0*a2))/a;
    double si2 = 2.0/(1.0+4.0*lisciatore2)+2.0/a2*(1.0+4.0*lisciatore4/a2+24.0*lisciatore4*lisciatore4
        /(4.0*a2*a2))-4.0/wb*(1.0+6.0*lisciatore4/(2.0*wb)+8.0*lisciatore4*lisciatore4/(2.0*wb*wb));
    double mu2 = mu*mu;
    if (si2 < 1e-14){si2 = 1e-14;}
    if (mu2 < 1e-14){mu2 = 1e-14;}
    if (si2/mu2 < 0.0 || si2+mu2 == 0.0){throw std::runtime_error("Problemi nei calcoli per il test di Henze-Zirkler");}
    double pmu = std::log(std::sqrt(mu2*mu2/(si2+mu2)));
    double psi = std::sqrt(std::log(1.0+si2/mu2));
    CoseTest cose;
    lognormal_distribution<> distribuzione(std::exp(pmu), psi);
    cose.statistica = hz;
    cose.pvalue = 1.0-cdf(distribuzione, hz);
    cose.accettazione = cose.pvalue >= alfa;
    return cose;
}


// Funzione per eseguire il test di Mardia per la normalità multivariata. Deve ricevere una lista
// di ascisse, una di ordinate, la dimensione n; la soglia per l'asimmetria è il quantile 1-alfa di una
// chi quadro con 4 gradi di libertà, e la soglia della curtosi il quantile 1-alfa di una
// normale standard. (Normalmente si prende la radice di 1-alfa perché si fanno due test assieme).
// Restituisce true se l'ipotesi nulla è accettata e false altrimenti.
// L'algoritmo è adattato dal codice di R per la funzione mardiatest del pacchetto MVET.
std::array<CoseTest, 2> testmardia(const std::vector<double> &ascisse, const std::vector<double> &ordinate,
    const int taglia, const double alfaasimmetria, const double alfacurtosi){
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (ascisse.size() != taglia || ordinate.size() != taglia){throw std::runtime_error("Il dataset non corrisponde alla taglia");}
    // Per prima cosa calcoliamo il vettore medio, la matrice di covarianze e la sua inversa.
    // Per evitare problemi dovuti a matrici singolari, usiamo la regolarizzazione di Tikhonov.
    double mediaascisse = 0.0;
    double mediaordinate = 0.0;
    for (int i = 0; i < taglia; i++){
        mediaascisse += ascisse[i];
        mediaordinate += ordinate[i];
    }
    mediaascisse /= taglia;
    mediaordinate /= taglia;
    double varianzaascisse = 0.0;
    double varianzaordinate = 0.0;
    double covarianza = 0.0;
    for (int i = 0; i < taglia; i++){
        const double differenzaascisse = ascisse[i]-mediaascisse;
        const double differenzaordinate = ordinate[i]-mediaordinate;
        varianzaascisse += differenzaascisse*differenzaascisse;
        varianzaordinate += differenzaordinate*differenzaordinate;
        covarianza += differenzaascisse*differenzaordinate;
    }
    varianzaascisse /= taglia;
    varianzaordinate /= taglia;
    covarianza /= taglia;
    double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
    if (determinante < 0.000001){
        varianzaascisse += 0.000001;
        varianzaordinate += 0.000001;
        determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
    }
    const double inversa11 = varianzaordinate/determinante;
    const double inversa22 = varianzaascisse/determinante;
    const double inversa12 = -covarianza/determinante;
    // Per calcolare l'asimmetria e la curtosi, abbiamo bisogno delle quantità intermedie
    // m_{ij}=(x_i-\bar{x})^T\Sigma^{-1}(x_j-\bar{x}). Abbiamo già \bar{x} e quindi dobbiamo
    // solo ottenere le differenze (x_i-\bar{x}), per poi eseguire i prodotti matriciali;
    // salviamo gli m_{ij} ottenuti da questi nel vettore sommandi, che sarà piatto.
    std::vector<double> ascissecentrate(taglia);
    std::vector<double> ordinatecentrate(taglia);
    for (int i = 0; i < taglia; i++){
        ascissecentrate[i] = ascisse[i]-mediaascisse;
        ordinatecentrate[i] = ordinate[i]-mediaordinate;
    }
    std::vector<double> sommandi(taglia*taglia);
    int indice = 0;
    for (int i = 0; i < taglia; i++){
        for (int j = 0; j < taglia; j++){
            const double primoprodotto1 = ascissecentrate[i]*inversa11+ordinatecentrate[i]*inversa12;
            const double primoprodotto2 = ascissecentrate[i]*inversa12+ordinatecentrate[i]*inversa22;
            sommandi[indice] = primoprodotto1*ascissecentrate[j]+primoprodotto2*ordinatecentrate[j];
            indice++;
        }
    }
    // b_{12}=\sum_i\sum_jm^3_{ij}/n^2:
    double b1 = 0.0;
    for (int i = 0; i < taglia*taglia; i++){
        const double sommando = sommandi[i];
        b1 += sommando*sommando*sommando;
    }
    b1 /= taglia*taglia;
    // b_{22}=\sum_im^2_{ij}/n:
    double b2 = 0.0;
    for (int i = 0; i < taglia; i++){
        const double sommando = sommandi[i*taglia+i];
        b2 += sommando*sommando;
    }
    b2 /= taglia;
    // L'asimmetria è pari a nkb_1/6; se n >= 20, k=1, altrimenti è posto pari a una costante
    // di aggiustamento 3(n+1)(n+3)/(3n(n+1)-6) (nel caso p=2, che è il nostro).
    double asimmetria;
    if (taglia < 20) {
        asimmetria = taglia*3.0*(taglia+1.0)*(taglia+3.0)/(3.0*(taglia+1.0)*taglia-6.0)*b1/6.0;
    } else {
        asimmetria = taglia*b1/6.0;
    }
    // La curtosi, nel caso p=2, è pari a (b2-8)\sqrt{n/64}.
    const double curtosi = (b2-8.0*(taglia-1.0)/(taglia+1.0))*std::sqrt(taglia/64.0);
    // L'ipotesi nulla è rifiutata se l'asimmetria o la curtosi sono sopra le rispettive soglie critiche.
    CoseTest coseasimmetriche;
    chi_squared_distribution<> chiquadro(4);
    coseasimmetriche.statistica = asimmetria;
    coseasimmetriche.pvalue = 1.0-cdf(chiquadro, asimmetria);
    coseasimmetriche.accettazione = coseasimmetriche.pvalue >= alfaasimmetria;
    CoseTest cosecurtotiche;
    normal normale(0.0, 1.0);
    cosecurtotiche.statistica = curtosi;
    cosecurtotiche.pvalue = 1.0-cdf(normale, curtosi);
    cosecurtotiche.accettazione = cosecurtotiche.pvalue >= alfacurtosi;
    return {coseasimmetriche, cosecurtotiche};
}


// Funzione che contiene il corpo del codice per betatesthenzezirkler. Vengono estratti una serie
// di campioni Monte Carlo di cui vengono calcolate le statistiche test di Henze-Zirkler e i momenti
// corrispondenti; queste vengono immagazzinate nella variabile "statistiche", che deve essere passata
// come riferimento. Tali statistiche possono essere poi utilizzate per effettuare sui campioni Monte
// Carlo il test di Henze-Zirkler e quindi ricavare una stima della sua probabilità di errore di II tipo.
// Deve essere passata la dimensione n del campione, la distribuzione da cui vanno estratti i campioni
// Monte Carlo sotto l'ipotesi alternativa (una tra L="laplace", A="normaleasimmetrica", U="uniforme", M="mistura"
// (di normali), N="lognormale" e T="t"), il numero di iterazioni Monte Carlo, 0 come supplementari per tutti
// i thread tranne l'ultimo (che deve avere il numero di iterazioni rimanenti), il riferimento al vettore
// delle statistiche, l'indice del thread attuale, l'indice di asimmetria lungo le ascisse e lungo le
// ordinate per la generazione di una normale asimmetrica, la distanza tra le medie delle due componenti
// per la generazione della mistura di normali e i gradi di libertà per la generazione della t di Student.
// Questi ultimi quattro parametri devono essere passati anche se si utilizza un'altra distribuzione;
// semplicemente non verranno usati in tal caso.
int calcolabetatesthenzezirkler(const int taglia, const char distribuzione, const int iterazioni,
    const int supplementari, double asimmetriaascisse, double asimmetriaordinate,
    const double distanzacomponenti, const int gradit, const double alfa){
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (gradit < 1){throw std::runtime_error("I gradi di libertà devono essere positivi");}
    if (iterazioni < 1){throw std::runtime_error("Le iterazioni devono essere positive");}
    if (distribuzione != 'L' && distribuzione != 'A' && distribuzione != 'U' && distribuzione != 'M' && distribuzione != 'N' && distribuzione != 'T'){throw std::runtime_error("Distribuzione sotto H_1 non contemplata");}
    int accettazioni = 0;
    // Prepariamo il generatore di numeri casuali.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    // Per ogni iterazione:
    for (int volta = 0; volta < iterazioni+supplementari; volta++){
        // Inizializziamo due vettori per contenere il campione Monte Carlo.
        std::vector<double> campioneascisse(taglia);
        std::vector<double> campioneordinate(taglia);
        if (distribuzione == 'L'){
            // Questo metodo è preso dalla pagina di Wikipedia sulla distribuzione di Laplace
            // multivariata. Presupponiamo che \mu=[0 0], di modo che Laplace simmetrica e
            // asimmetrica coincidano, e che \Sigma sia una matrice identità per semplificare i calcoli.
            // Il metodo consiste semplicemente nel generare un valore da un'esponenziale(1)
            // e poi moltiplicare per esso i valori estratti da una normale bivariata a media nulla,
            // o nel nostro caso da una normale bivariata standard.
            std::exponential_distribution<> esponenziale(1.0);
            std::normal_distribution<> normale(0.0, 1.0);
            for (int i = 0; i < taglia; i++){
                const double z = esponenziale(generatore);
                campioneascisse[i] = normale(generatore)*std::sqrt(z);
                campioneordinate[i] = normale(generatore)*std::sqrt(z);
            }
        } else if (distribuzione == 'A'){
            // Questo metodo è preso da Azzalini, Dalla Valle (1996), e nello specifico è il metodo
            // per condizionamento. Partiamo calcolando i valori delta_i a partire dai parametri
            // di asimmetria \lambda_i, che qui sono asimmetriaascisse e asimmetriaordinate.
            double delta1 = asimmetriaascisse/std::sqrt(1.0+asimmetriaascisse*asimmetriaascisse);
            double delta2 = asimmetriaordinate/std::sqrt(1.0+asimmetriaordinate*asimmetriaordinate);
            // Ora dobbiamo calcolare la matrice \Omega^*; questo a sua volta necessita della matrice
            // \Omega, che è pari a \Delta(\Psi+\lambda\lambda^T)\Delta. Qui presupponiamo che \Psi,
            // che è la matrice di covarianze della normale simmetrica sottostante, sia una matrice
            // identità. \Omega^* ha la struttura [[1, \delta_1, \delta_2], [\delta_1, \Omega_{11}, \Omega_{12}],
            // [\delta_2, \Omega_{12}, \Omega_{22}]]; dunque, per economia, calcoliamo e conserviamo
            // solo gli elementi \Omega_{11}, \Omega_{12} e \Omega_{22}.
            const double unomenodelta1 = 1.0-delta1*delta1;
            const double unomenodelta2 = 1.0-delta2*delta2;
            double omega11 = (1.0+asimmetriaascisse*asimmetriaascisse)*unomenodelta1;
            double omega12 = asimmetriaascisse*asimmetriaordinate*std::sqrt(unomenodelta1)*std::sqrt(unomenodelta2);
            double omega22 = (1.0+asimmetriaordinate*asimmetriaordinate)*unomenodelta2;
            // Adesso per generare da una normale trivariata N_3(0, \Omega^*) dobbiamo avere la
            // decomposizione di Cholesky di \Omega^*. Ricordiamo che affinché la decomposizione di
            // Cholesky sia ben definita, \Omega^* deve essere definita positiva. Dunque controlliamo
            // che lo sia valutando se il determinante è positivo; se non lo è, ignoriamo i parametri
            // \lambda_i immessi dall'utente e utilizziamo \lambda_1=\lambda_2=1, che inducono
            // una matrice definita positiva, prima di calcolare la decomposizione.
            // Altrimenti, calcoliamo la decomposizione utilizzando la sua forma chiusa,
            // rinvenibile anche su Wikipedia. Dato che sappiamo che la matrice sarà del tipo
            // [[1, 0, 0], [\delta_1, L_{22}, 0], [\delta_2, L_{32}, L_{33}]], salviamo solo
            // i tre elementi L_{22}, L_{32} e L_{33}.
            if (omega11*omega22+delta1*omega12*omega12+delta2*delta1*omega12-delta2*omega11*delta2-
                delta1*delta1*omega22-omega12*omega12 > 0.0) {
                asimmetriaascisse = 1.0;
                asimmetriaordinate = 1.0;
                delta1 = std::sqrt(2.0)/2.0;
                delta2 = std::sqrt(2.0)/2.0;
                omega11 = std::sqrt(2.0);
                omega12 = 0.5;
                omega22 = std::sqrt(2.0);
            }
            if (omega11-delta1*delta1 <= 0.0){throw std::runtime_error("Parametri di asimmetria invalidi");}
            const double cholesky22 = std::sqrt(omega11-delta1*delta1);
            const double cholesky32 = (omega12-delta1*delta2)/cholesky22;
            if (omega22-delta2*delta2-cholesky32 < 0.0){throw std::runtime_error("Parametri di asimmetria invalidi");}
            const double cholesky33 = std::sqrt(omega22-delta2*delta2-cholesky32);
            // Adesso procediamo ad applicare l'algoritmo. Generiamo da una X~N(0, \Omega^*)
            // premoltiplicando un vettore di tre variabili normali standard per la decomposizione
            // di Cholesky ottenuta. Se il primo elemento x1 così generato è positivo, aggiungiamo
            // [x2 x3] al campione, altrimenti ripetiamo; questo continua finché il campione non
            // ha n elementi.
            int completati = 0;
            std::normal_distribution<> normale(0.0, 1.0);
            while (completati < taglia){
                const double normale1 = normale(generatore);
                if (normale1 > 0.0) {
                    const double normale2 = normale(generatore);
                    const double normale3 = normale(generatore);
                    campioneascisse[completati] = delta1*normale1+cholesky22*normale2;
                    campioneordinate[completati] = delta2*normale1+cholesky32*normale2+cholesky33*normale3;
                    completati++;
                }
            }
        } else if (distribuzione == 'U'){
            // L'estrazione da un'uniforme, anche bivariata, è banale.
            std::uniform_real_distribution<> uniforme(-1.0, 1.0);
            for (int i = 0; i < taglia; i++){
                campioneascisse[i] = uniforme(generatore);
                campioneordinate[i] = uniforme(generatore);
            }
        } else if (distribuzione == 'M'){
            // Per avere la massima semplicità possibile, presupponiamo che la mistura bivariata da cui vogliamo
            // estrarre un campione abbia solo due componenti, equiprobabili e con matrice di covarianze
            // pari alla matrice identità; inoltre, la media delle ordinate sarà entrambe nulla. Solo
            // le medie delle ascisse avranno distanza pari a distanzacomponenti. L'algoritmo
            // per generare il campione date queste premesse è noto e banale.
            std::uniform_real_distribution<> uniforme(0, 1);
            std::normal_distribution<> normale1(-distanzacomponenti/2.0, 1.0);
            std::normal_distribution<> normale2(distanzacomponenti/2.0, 1.0);
            std::normal_distribution<> normale3(0.0, 1.0);
            for (int i = 0; i < taglia; i++){
                const double u = uniforme(generatore);
                if (u > 0.5){
                    campioneascisse[i] = normale1(generatore);
                    campioneordinate[i] = normale3(generatore);
                } else {
                    campioneascisse[i] = normale2(generatore);
                    campioneordinate[i] = normale3(generatore);
                }
            }
        } else if (distribuzione == 'N'){
            // L'estrazione da una lognormale è banale.
            std::normal_distribution<> normale(0.0, 1.0);
            for (int i = 0; i < taglia; i++){
                campioneascisse[i] = std::exp(normale(generatore));
                campioneordinate[i] = std::exp(normale(generatore));
            }
        } else if (distribuzione == 'T'){
            // Per estrarre da una t di Student bivariata con gradit gradi di libertà, usiamo un algoritmo noto
            // (si veda la pagina di Wikipedia), in cui si estrae un numero u da una chi quadro con
            // gradit gradi di libertà e un vettore y da N(0, \Sigma). Qui presupporremo che \Sigma=I,
            // rendendo l'intero processo tutto sommato banale. Il campione finale è dato da y\sqrt{gradit/u}+\mu,
            // ma presupporremo anche che \mu=[0 0].
            std::normal_distribution<> normale(0.0, 1.0);
            std::chi_squared_distribution<> chiquadrato(gradit);
            for (int i = 0; i < taglia; i++){
                double z1 = normale(generatore);
                double z2 = normale(generatore);
                double u = chiquadrato(generatore);
                if (u < 0.000001){u += 0.000001;}
                campioneascisse[i] = z1*std::sqrt(gradit/u);
                campioneordinate[i] = z2*std::sqrt(gradit/u);
            }
        }
        // Adesso eseguiamo lo stesso processo in testhenzezirkler.
        double mediaascisse = 0.0;
        double mediaordinate = 0.0;
        for (int i = 0; i < taglia; i++){
            mediaascisse += campioneascisse[i];
            mediaordinate += campioneordinate[i];
        }
        mediaascisse /= taglia;
        mediaordinate /= taglia;
        std::vector<double> ascissecentrate(taglia);
        std::vector<double> ordinatecentrate(taglia);
        double varianzaascisse = 0.0;
        double varianzaordinate = 0.0;
        double covarianza = 0.0;
        for (int i = 0; i < taglia; i++){
            const double differenzaascisse = campioneascisse[i]-mediaascisse;
            const double differenzaordinate = campioneordinate[i]-mediaordinate;
            ascissecentrate[i] = differenzaascisse;
            ordinatecentrate[i] = differenzaordinate;
            varianzaascisse += differenzaascisse*differenzaascisse;
            varianzaordinate += differenzaordinate*differenzaordinate;
            covarianza += differenzaascisse*differenzaordinate;
        }
        varianzaascisse /= taglia;
        varianzaordinate /= taglia;
        covarianza /= taglia;
        const double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        double hz;
        double lisciatore;
        double lisciatore2;
        if (determinante < 0.000001){
            hz = 4.0*taglia;
            lisciatore = 1.0/std::sqrt(2.0)*std::pow(5.0/4.0, 1.0/6.0)*std::pow(static_cast<double>(taglia), 1.0/6.0);
            lisciatore2 = lisciatore*lisciatore;
        } else {
            double inversa11 = varianzaordinate/determinante;
            double inversa12 = -covarianza/determinante;
            double inversa22 = varianzaascisse/determinante;
            // Adesso calcoliamo il prodotto matriciale tra la matrice dei dati centrati e la matrice
            // di covarianze. Immagazziniamo la prima e la seconda colonna della matrice risultante
            // come due vettori separati.
            std::vector<double> primoprodotto1(taglia);
            std::vector<double> primoprodotto2(taglia);
            for (int i = 0; i < taglia; i++){
                primoprodotto1[i] = ascissecentrate[i]*inversa11+ordinatecentrate[i]*inversa12;
                primoprodotto2[i] = ascissecentrate[i]*inversa12+ordinatecentrate[i]*inversa22;
            }
            // Adesso otteniamo la matrice Dj come la matrice diagonale del prodotto fatto prima
            // moltiplicato per la matrice dei dati centrati trasposta (di fatto completando il prodotto
            // \bar{x}\Sigma^{-1}\bar{x}^T).
            std::vector<double> Dj(taglia);
            for (int i = 0; i < taglia; i++){
                Dj[i] = primoprodotto1[i]*ascissecentrate[i]+primoprodotto2[i]*ordinatecentrate[i];
            }
            // Adesso, in modo simile a prima, otteniamo la matrice Y=x\Sigma^{-1}x^T.
            std::vector<double> primoprodottodecentrato1(taglia);
            std::vector<double> primoprodottodecentrato2(taglia);
            for (int i = 0; i < taglia; i++){
                primoprodottodecentrato1[i] = campioneascisse[i]*inversa11+campioneordinate[i]*inversa12;
                primoprodottodecentrato2[i] = campioneascisse[i]*inversa12+campioneordinate[i]*inversa22;
            }
            std::vector<std::vector<double>> Y(taglia, std::vector<double>(taglia));
            for (int i = 0; i < taglia; i++){
                for (int j = 0; j < taglia; j++){
                    Y[i][j] = primoprodottodecentrato1[j]*campioneascisse[i]+primoprodottodecentrato2[j]*campioneordinate[i];
                }
            }
            // Estraiamo anche di Y la matrice diagonale.
            std::vector<double> Y_diag(taglia);
            for (int i = 0; i < taglia; i++){
                Y_diag[i] = Y[i][i];
            }
            // Adesso calcoliamo la matrice Djk che contiene le distanze di Mahalanobis al quadrato.
            std::vector<std::vector<double>> Djk(taglia, std::vector<double>(taglia));
            for (int i = 0; i < taglia; i++){
                for (int j = 0; j < taglia; j++){
                    Djk[i][j] = -2.0*Y[j][i]+Y_diag[i]+Y_diag[j];
                }
            }
            // Parametro di smoothing.
            lisciatore = 1.0/std::sqrt(2.0)*std::pow(5.0/4.0, 1.0/6.0)*std::pow(static_cast<double>(taglia), 1.0/6.0);
            lisciatore2 = lisciatore*lisciatore;
            // Con questi passaggi calcoliamo la statistica test di Henze-Zirkler.
            double sommakernel = 0.0;
            double sommakernel2 = 0.0;
            for (int i = 0; i < taglia; i++){
                for (int j = 0; j < taglia; j++){
                    sommakernel += std::exp(-lisciatore2/2.0*Djk[i][j]);
                }
                sommakernel2 += std::exp(-lisciatore2/(2.0+2.0*lisciatore2)*Dj[i]);
            }
            hz = sommakernel/taglia-2.0*sommakernel2/(1.0+lisciatore2)+taglia/(1.0+2.0*lisciatore2);
        }
        // Adesso calcoliamo la media e la varianza della log-normale.
        const double wb = (1.0+lisciatore2)*(1.0+3.0*lisciatore2);
        const double a = 1.0+2.0*lisciatore2;
        const double lisciatore4 = lisciatore2*lisciatore2;
        const double a2 = a*a;
        const double mu = 1.0-(1.0+2.0*lisciatore2/a+8.0*lisciatore4/(2.0*a2))/a;
        double si2 = 2.0/(1.0+4.0*lisciatore2)+2.0/a2*(1.0+4.0*lisciatore4/a2+24.0*lisciatore4*lisciatore4
            /(4.0*a2*a2))-4.0/wb*(1.0+6.0*lisciatore4/(2.0*wb)+8.0*lisciatore4*lisciatore4/(2.0*wb*wb));
        double mu2 = mu*mu;
        if (si2 < 1e-14){si2 = 1e-14;}
        if (mu2 < 1e-14){mu2 = 1e-14;}
        if (si2/mu2 < 0.0 || si2+mu2 == 0.0){throw std::runtime_error("Problemi nei calcoli per il test di Henze-Zirkler");}
        double pmu = std::log(std::sqrt(mu2*mu2/(si2+mu2)));
        double psi = std::sqrt(std::log(1.0+si2/mu2));
        lognormal_distribution<> distribuzione(std::exp(pmu), psi);
        if (hz <= quantile(distribuzione, 1.0-alfa)){accettazioni++;}
    }
    return accettazioni;
}


// Funzione preparatoria per calcolare la probabilità di errore di II tipo del test di Henze-Zirkler.
// Restituisce un vettore con terne di statistiche test e parametri; questi devono essere valutati
// all'esterno di questo programma per calcolare la probabilità effettiva, applicando a ciascuna
// il test di Henze-Zirkler.
double betatesthenzezirkler(const char distribuzione, const int taglia, const int iterazioni,
    const int gradit, const double asimmetriaascisse, const double asimmetriaordinate,
    const double distanzacomponenti, const double alfa){
    // Otteniamo il numero massimo possibile di thread. Se non ci riusciamo, impostiamone quattro.
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    // Contiamo quante iterazioni fa ciascun thread.
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    // Inizializziamo un vettore di terne vuoto dove riporre le statistiche. Questo verrà passato
    // per riferimento all'interno dei thread in modo che le terne vengano riempite; proprio per questo
    // è necessario passare il numero del thread come argomento, in modo che le posizioni da riempire
    // per ogni thread siano preallocate e non si sovrappongano tra i diversi thread.
    std::vector<int> accettazionithread(numerothread, 0);
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcolabetatesthenzezirkler(taglia, distribuzione, volteperthread, 0,
                                        asimmetriaascisse, asimmetriaordinate, distanzacomponenti, gradit, alfa);
        });
    }
    // Nell'ultimo thread, a causa della struttura della funzione calcolabetatesthenzezirkler,
    // il numero di iterazioni rimanenti deve essere passato come argomento a sé; infatti, altrimenti,
    // l'ultima riga di tale funzione causerebbe un errore perché l'indice "numerothread*iterazioni+volta"
    // sforerebbe dalla lunghezza predefinita del vettore.
    thread[numerothread - 1] = std::thread([=, &accettazionithread](){
        accettazionithread[numerothread-1] = calcolabetatesthenzezirkler(taglia, distribuzione, volteperthread, iterazionirestanti,
            asimmetriaascisse, asimmetriaordinate, distanzacomponenti, gradit, alfa);
    });
    // Dopo aver atteso che tutti i thread finiscano, restituiamo il vettore di terne.
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return static_cast<double>(accettazioni)/iterazioni;
}


// Funzione che esegue effettivamente i calcoli per trovare la probabilità di errore di II tipo
// del test di Mardia. Va passato il numero di iterazioni da compiere, la dimensione n del campione,
// la distribuzione bivariata da usare sotto l'ipotesi alternativa (una tra "laplace", "normaleasimmetrica",
// "uniforme", "mistura" (di normali), "lognormale", "t"), i gradi di libertà della t di Student,
// i parametri di asimmetria delle ascisse e delle ordinate in caso di uso della normale asimmetrica,
// la distanza tra le medie delle ascisse delle due componenti della mistura, il quanile 1-alfa
// di una chi quadro con 4 gradi di libertà e il quantile 1-alfa di una normale standard.
// Gli argomenti gradit, asimmetriaascisse, asimmetriaordiante e distanzacomponenti vanno passati
// indipendentemente dalla distribuzione scelta; semplicemente non verranno usati tutti.
// Viene restituito il numero di volte in cui l'ipotesi nulla viene accettata.
int calcolabetatestmardia(const int iterazioni, const int taglia, const char distribuzione,
    const int gradit, double asimmetriaascisse, double asimmetriaordinate,
    const double distanzacomponenti, const double sogliaasimmetria, const double sogliacurtosi){
    if (iterazioni < 1){throw std::runtime_error("Le iterazioni devono essere positive");}
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (gradit < 1){throw std::runtime_error("I gradi di libertà devono essere positivi");}
    if (distribuzione != 'L' && distribuzione != 'A' && distribuzione != 'U' && distribuzione != 'M' && distribuzione != 'N' && distribuzione != 'T'){throw std::runtime_error("Distribuzione sotto H_1 non contemplata");}
    // Prepariamo la generazione di numeri casuali e inizializziamo il contatore di accettazioni
    // dell'ipotesi nulla.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    int accettazioni = 0;
    for (int volta = 0; volta < iterazioni; volta++) {
        // Inizializziamo un vettore per contenere il campione Monte Carlo; le ascisse saranno
        // in posizione pari e le ordinate in quelle dispari.
        std::vector<double> campioneascisse(taglia);
        std::vector<double> campioneordinate(taglia);
        if (distribuzione == 'L'){
            // Questo metodo è preso dalla pagina di Wikipedia sulla distribuzione di Laplace
            // multivariata. Presupponiamo che \mu=[0 0], di modo che Laplace simmetrica e
            // asimmetrica coincidano, e che \Sigma sia una matrice identità per semplificare i calcoli.
            // Il metodo consiste semplicemente nel generare un valore da un'esponenziale(1)
            // e poi moltiplicare per esso i valori estratti da una normale bivariata a media nulla,
            // o nel nostro caso da una normale bivariata standard.
            std::exponential_distribution<> esponenziale(1.0);
            std::normal_distribution<> normale(0.0, 1.0);
            for (int i = 0; i < taglia; i++){
                const double z = esponenziale(generatore);
                campioneascisse[i] = normale(generatore)*std::sqrt(z);
                campioneordinate[i] = normale(generatore)*std::sqrt(z);
            }
        } else if (distribuzione == 'A'){
            // Questo metodo è preso da Azzalini, Dalla Valle (1996), e nello specifico è il metodo
            // per condizionamento. Partiamo calcolando i valori delta_i a partire dai parametri
            // di asimmetria \lambda_i, che qui sono asimmetriaascisse e asimmetriaordinate.
            double delta1 = asimmetriaascisse/std::sqrt(1.0+asimmetriaascisse*asimmetriaascisse);
            double delta2 = asimmetriaordinate/std::sqrt(1.0+asimmetriaordinate*asimmetriaordinate);
            // Ora dobbiamo calcolare la matrice \Omega^*; questo a sua volta necessita della matrice
            // \Omega, che è pari a \Delta(\Psi+\lambda\lambda^T)\Delta. Qui presupponiamo che \Psi,
            // che è la matrice di covarianze della normale simmetrica sottostante, sia una matrice
            // identità. \Omega^* ha la struttura [[1, \delta_1, \delta_2], [\delta_1, \Omega_{11}, \Omega_{12}],
            // [\delta_2, \Omega_{12}, \Omega_{22}]]; dunque, per economia, calcoliamo e conserviamo
            // solo gli elementi \Omega_{11}, \Omega_{12} e \Omega_{22}.
            const double unomenodelta1 = 1.0-delta1*delta1;
            const double unomenodelta2 = 1.0-delta2*delta2;
            double omega11 = (1.0+asimmetriaascisse*asimmetriaascisse)*unomenodelta1;
            double omega12 = asimmetriaascisse*asimmetriaordinate*std::sqrt(unomenodelta1)*std::sqrt(unomenodelta2);
            double omega22 = (1.0+asimmetriaordinate*asimmetriaordinate)*unomenodelta2;
            // Adesso per generare da una normale trivariata N_3(0, \Omega^*) dobbiamo avere la
            // decomposizione di Cholesky di \Omega^*. Ricordiamo che affinché la decomposizione di
            // Cholesky sia ben definita, \Omega^* deve essere definita positiva. Dunque controlliamo
            // che lo sia valutando se il determinante è positivo; se non lo è, ignoriamo i parametri
            // \lambda_i immessi dall'utente e utilizziamo \lambda_1=\lambda_2=1, che inducono
            // una matrice definita positiva, prima di calcolare la decomposizione.
            // Altrimenti, calcoliamo la decomposizione utilizzando la sua forma chiusa,
            // rinvenibile anche su Wikipedia. Dato che sappiamo che la matrice sarà del tipo
            // [[1, 0, 0], [\delta_1, L_{22}, 0], [\delta_2, L_{32}, L_{33}]], salviamo solo
            // i tre elementi L_{22}, L_{32} e L_{33}.
            if (omega11*omega22+delta1*omega12*omega12+delta2*delta1*omega12-delta2*omega11*delta2-
                delta1*delta1*omega22-omega12*omega12 > 0.0) {
                asimmetriaascisse = 1.0;
                asimmetriaordinate = 1.0;
                delta1 = std::sqrt(2.0)/2.0;
                delta2 = std::sqrt(2.0)/2.0;
                omega11 = std::sqrt(2.0);
                omega12 = 0.5;
                omega22 = std::sqrt(2.0);
            }
            if (omega11-delta1*delta1 <= 0.0){throw std::runtime_error("Parametri di asimmetria invalidi");}
            const double cholesky22 = std::sqrt(omega11-delta1*delta1);
            const double cholesky32 = (omega12-delta1*delta2)/cholesky22;
            if (omega22-delta2*delta2-cholesky32 < 0.0){throw std::runtime_error("Parametri di asimmetria invalidi");}
            const double cholesky33 = std::sqrt(omega22-delta2*delta2-cholesky32);
            // Adesso procediamo ad applicare l'algoritmo. Generiamo da una X~N(0, \Omega^*)
            // premoltiplicando un vettore di tre variabili normali standard per la decomposizione
            // di Cholesky ottenuta. Se il primo elemento x1 così generato è positivo, aggiungiamo
            // [x2 x3] al campione, altrimenti ripetiamo; questo continua finché il campione non
            // ha n elementi.
            int completati = 0;
            std::normal_distribution<> normale(0.0, 1.0);
            while (completati < taglia){
                const double normale1 = normale(generatore);
                if (normale1 > 0.0) {
                    const double normale2 = normale(generatore);
                    const double normale3 = normale(generatore);
                    campioneascisse[completati] = delta1*normale1+cholesky22*normale2;
                    campioneordinate[completati] = delta2*normale1+cholesky32*normale2+cholesky33*normale3;
                    completati++;
                }
            }
        } else if (distribuzione == 'U'){
            // L'estrazione da un'uniforme, anche bivariata, è banale.
            std::uniform_real_distribution<> uniforme(-1.0, 1.0);
            for (int i = 0; i < taglia; i++){
                campioneascisse[i] = uniforme(generatore);
                campioneordinate[i] = uniforme(generatore);
            }
        } else if (distribuzione == 'M'){
            // Per avere la massima semplicità possibile, presupponiamo che la mistura bivariata da cui vogliamo
            // estrarre un campione abbia solo due componenti, equiprobabili e con matrice di covarianze
            // pari alla matrice identità; inoltre, la media delle ordinate sarà entrambe nulla. Solo
            // le medie delle ascisse avranno distanza pari a distanzacomponenti. L'algoritmo
            // per generare il campione date queste premesse è noto e banale.
            std::uniform_real_distribution<> uniforme(0.0, 1.0);
            std::normal_distribution<> normale1(-distanzacomponenti/2.0, 1.0);
            std::normal_distribution<> normale2(distanzacomponenti/2.0, 1.0);
            std::normal_distribution<> normale3(0.0, 1.0);
            for (int i = 0; i < taglia; i++){
                const double u = uniforme(generatore);
                if (u > 0.5){
                    campioneascisse[i] = normale1(generatore);
                    campioneordinate[i] = normale3(generatore);
                } else {
                    campioneascisse[i] = normale2(generatore);
                    campioneordinate[i] = normale3(generatore);
                }
            }
        } else if (distribuzione == 'N'){
            // L'estrazione da una lognormale è banale.
            std::normal_distribution<> normale(0.0, 1.0);
            for (int i = 0; i < taglia; i++){
                campioneascisse[i] = std::exp(normale(generatore));
                campioneordinate[i] = std::exp(normale(generatore));
            }
        } else if (distribuzione == 'T'){
            // Per estrarre da una t di Student bivariata con gradit gradi di libertà, usiamo un algoritmo noto
            // (si veda la pagina di Wikipedia), in cui si estrae un numero u da una chi quadro con
            // gradit gradi di libertà e un vettore y da N(0, \Sigma). Qui presupporremo che \Sigma=I,
            // rendendo l'intero processo tutto sommato banale. Il campione finale è dato da y\sqrt{gradit/u}+\mu,
            // ma presupporremo anche che \mu=[0 0].
            std::normal_distribution<> normale(0.0, 1.0);
            std::chi_squared_distribution<> chiquadrato(gradit);
            for (int i = 0; i < taglia; i++){
                double z1 = normale(generatore);
                double z2 = normale(generatore);
                double u = chiquadrato(generatore);
                if (u < 0.000001){u += 0.000001;}
                campioneascisse[i] = z1*std::sqrt(gradit/u);
                campioneordinate[i] = z2*std::sqrt(gradit/u);
            }
        }
        // Per prima cosa calcoliamo il vettore medio, la matrice di covarianze e la sua inversa.
        // Per evitare problemi dovuti a matrici singolari, usiamo la regolarizzazione di Tikhonov.
        double mediaascisse = 0.0;
        double mediaordinate = 0.0;
        for (int i = 0; i < taglia; i++){
            mediaascisse += campioneascisse[i];
            mediaordinate += campioneordinate[i];
        }
        mediaascisse /= taglia;
        mediaordinate /= taglia;
        double varianzaascisse = 0.0;
        double varianzaordinate = 0.0;
        double covarianza = 0.0;
        for (int i = 0; i < taglia; i++){
            const double differenzaascisse = campioneascisse[i]-mediaascisse;
            const double differenzaordinate = campioneordinate[i]-mediaordinate;
            varianzaascisse += differenzaascisse*differenzaascisse;
            varianzaordinate += differenzaordinate*differenzaordinate;
            covarianza += differenzaascisse*differenzaordinate;
        }
        varianzaascisse /= taglia;
        varianzaordinate /= taglia;
        covarianza /= taglia;
        double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        if (determinante < 0.000001){
            varianzaascisse += 0.000001;
            varianzaordinate += 0.000001;
            determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        }
        const double inversa11 = varianzaordinate/determinante;
        const double inversa22 = varianzaascisse/determinante;
        const double inversa12 = -covarianza/determinante;
        // Per calcolare l'asimmetria e la curtosi, abbiamo bisogno delle quantità intermedie
        // m_{ij}=(x_i-\bar{x})^T\Sigma^{-1}(x_j-\bar{x}). Abbiamo già \bar{x} e quindi dobbiamo
        // solo ottenere le differenze (x_i-\bar{x}), per poi eseguire i prodotti matriciali;
        // salviamo gli m_{ij} ottenuti da questi nel vettore sommandi, che sarà piatto.
        std::vector<double> ascissecentrate(taglia);
        std::vector<double> ordinatecentrate(taglia);
        for (int i = 0; i < taglia; i++){
            ascissecentrate[i] = campioneascisse[i]-mediaascisse;
            ordinatecentrate[i] = campioneordinate[i]-mediaordinate;
        }
        std::vector<double> sommandi(taglia*taglia);
        int indice = 0;
        for (int i = 0; i < taglia; i++){
            for (int j = 0; j < taglia; j++){
                const double primoprodotto1 = ascissecentrate[i]*inversa11+ordinatecentrate[i]*inversa12;
                const double primoprodotto2 = ascissecentrate[i]*inversa12+ordinatecentrate[i]*inversa22;
                sommandi[indice] = primoprodotto1*ascissecentrate[j]+primoprodotto2*ordinatecentrate[j];
                indice++;
            }
        }
        // b_{12}=\sum_i\sum_jm^3_{ij}/n^2:
        double b1 = 0.0;
        for (int i = 0; i < taglia*taglia; i++){
            const double sommando = sommandi[i];
            b1 += sommando*sommando*sommando;
        }
        b1 /= taglia*taglia;
        // b_{22}=\sum_im^2_{ij}/n:
        double b2 = 0.0;
        for (int i = 0; i < taglia; i++){
            const double sommando = sommandi[i*taglia+i];
            b2 += sommando*sommando;
        }
        b2 /= taglia;
        // L'asimmetria è pari a nkb_1/6; se n >= 20, k=1, altrimenti è posto pari a una costante
        // di aggiustamento 3(n+1)(n+3)/(3n(n+1)-6) (nel caso p=2, che è il nostro).
        double asimmetria;
        if (taglia < 20) {
            asimmetria = taglia*3.0*(taglia+1.0)*(taglia+3.0)/(3.0*(taglia+1.0)*taglia-6.0)*b1/6.0;
        } else {
            asimmetria = taglia*b1/6.0;
        }
        // La curtosi, nel caso p=2, è pari a (b2-8)\sqrt{n/64}.
        const double curtosi = (b2-8.0*(taglia-1.0)/(taglia+1.0))*std::sqrt(taglia/64.0);
        // L'ipotesi nulla è rifiutata se l'asimmetria o la curtosi sono sopra le rispettive soglie critiche.
        if (asimmetria < sogliaasimmetria && curtosi < sogliacurtosi && curtosi > -sogliacurtosi) {
            accettazioni++;
        }
    }
    return accettazioni;
}


// Funzione preparatoria per calcolare la probabilità di errore di II tipo del test di Mardia.
double betatestmardia(const int iterazioni, const int taglia, const char distribuzione,
    const double asimmetriaascisse, const double asimmetriaordinate, const double distanzacomponenti,
    const int gradit, const double sogliaasimmetria, const double sogliacurtosi) {
    // Contiamo il massimo numero di thread possibili; se non ci riusciamo, impostiamone quattro.
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    // Calcoliamo quante iterazioni deve fare ciascun thread.
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    // Inizializziamo un contatore di quante volte l'ipotesi nulla viene accettata e poi facciamo
    // partire i thread.
    std::vector<int> accettazionithread(numerothread);
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcolabetatestmardia(volteperthread, taglia, distribuzione, gradit,
                asimmetriaascisse, asimmetriaordinate, distanzacomponenti, sogliaasimmetria, sogliacurtosi);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazionithread]() {
        accettazionithread[numerothread-1] = calcolabetatestmardia(volteperthread+iterazionirestanti, taglia, distribuzione, gradit,
            asimmetriaascisse, asimmetriaordinate, distanzacomponenti, sogliaasimmetria, sogliacurtosi);
    });
    // Aspettiamo che i thread finiscano e poi restituiamo \beta.
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return static_cast<double>(accettazioni)/iterazioni;
}


// Funzione che esegue effettivamente i calcoli per trovare l'effettiva probabilità di errore
// di I tipo del test di Henze-Zirkler. Come per la funzione per trovare \beta, non viene restituito
// niente ma si modifica soltanto un vettore di statistiche test e parametri passato come riferimento.
// Deve essere passata la dimensione n del campione, il numero di iterazioni da compiere più quelle
// supplementari per l'ultimo thread (in modo che le posizioni del vettore da riempire siano preallocate
// e non sovrapposte per ciascun thread), un riferimento al vettore delle statistiche test e l'indice
// del thread attuale.
int calcolaalfaverohenzezirkler(const int taglia, const int iterazioni, const int supplementari, const double alfa) {
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (iterazioni < 1){throw std::runtime_error("Le iterazioni devono essere positive");}
    int accettazioni = 0;
    // Prepariamo il generatore di numeri casuali.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normale(0.0, 1.0);
    for (int iterazione = 0; iterazione < iterazioni+supplementari; iterazione++) {
        // Inizializziamo i due vettori del campione, uno per le ascisse e uno per le ordinate.
        std::vector<double> campioneascisse(taglia);
        std::vector<double> campioneordinate(taglia);
        // A questo punto estraiamo il campione sotto l'ipotesi nulla, che sarà una normale standard.
        // Contemporaneamente, per velocità, calcoliamo il vettore medio.
        double mediaascisse = 0.0;
        double mediaordinate = 0.0;
        for (int i = 0; i < taglia; i++) {
            const double ascissa = normale(generatore);
            const double ordinata = normale(generatore);
            mediaascisse += ascissa;
            mediaordinate += ordinata;
            campioneascisse[i] = ascissa;
            campioneordinate[i] = ordinata;
        }
        mediaascisse /= taglia;
        mediaordinate /= taglia;
        // Adesso eseguiamo lo stesso processo in testhenzezirkler.
        // Per prima cosa, ci servono due oggetti: il primo è l'inversa della matrice di covarianze
        // campionaria del campione, e la matrice dei dati centrati. Per evitare problemi di matrici
        // singolari, usiamo la regolarizzazione di Tikhonov (questo differisce rispetto all'algoritmo di
        // Python, che risolve il problema usando l'inversa di Penrose).
        std::vector<double> ascissecentrate(taglia);
        std::vector<double> ordinatecentrate(taglia);
        double varianzaascisse = 0.0;
        double varianzaordinate = 0.0;
        double covarianza = 0.0;
        for (int i = 0; i < taglia; i++){
            const double differenzaascisse = campioneascisse[i]-mediaascisse;
            const double differenzaordinate = campioneordinate[i]-mediaordinate;
            ascissecentrate[i] = differenzaascisse;
            ordinatecentrate[i] = differenzaordinate;
            varianzaascisse += differenzaascisse*differenzaascisse;
            varianzaordinate += differenzaordinate*differenzaordinate;
            covarianza += differenzaascisse*differenzaordinate;
        }
        varianzaascisse /= taglia;
        varianzaordinate /= taglia;
        covarianza /= taglia;
        const double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        double hz;
        double lisciatore;
        double lisciatore2;
        if (determinante < 0.000001){
            hz = 4.0*taglia;
            lisciatore = 1.0/std::sqrt(2.0)*std::pow(5.0/4.0, 1.0/6.0)*std::pow(static_cast<double>(taglia), 1.0/6.0);
            lisciatore2 = lisciatore*lisciatore;
        } else {
            double inversa11 = varianzaordinate/determinante;
            double inversa12 = -covarianza/determinante;
            double inversa22 = varianzaascisse/determinante;
            // Adesso calcoliamo il prodotto matriciale tra la matrice dei dati centrati e la matrice
            // di covarianze. Immagazziniamo la prima e la seconda colonna della matrice risultante
            // come due vettori separati.
            std::vector<double> primoprodotto1(taglia);
            std::vector<double> primoprodotto2(taglia);
            for (int i = 0; i < taglia; i++){
                primoprodotto1[i] = ascissecentrate[i]*inversa11+ordinatecentrate[i]*inversa12;
                primoprodotto2[i] = ascissecentrate[i]*inversa12+ordinatecentrate[i]*inversa22;
            }
            // Adesso otteniamo la matrice Dj come la matrice diagonale del prodotto fatto prima
            // moltiplicato per la matrice dei dati centrati trasposta (di fatto completando il prodotto
            // \bar{x}\Sigma^{-1}\bar{x}^T).
            std::vector<double> Dj(taglia);
            for (int i = 0; i < taglia; i++){
                Dj[i] = primoprodotto1[i]*ascissecentrate[i]+primoprodotto2[i]*ordinatecentrate[i];
            }
            // Adesso, in modo simile a prima, otteniamo la matrice Y=x\Sigma^{-1}x^T.
            std::vector<double> primoprodottodecentrato1(taglia);
            std::vector<double> primoprodottodecentrato2(taglia);
            for (int i = 0; i < taglia; i++){
                primoprodottodecentrato1[i] = campioneascisse[i]*inversa11+campioneordinate[i]*inversa12;
                primoprodottodecentrato2[i] = campioneascisse[i]*inversa12+campioneordinate[i]*inversa22;
            }
            std::vector<std::vector<double>> Y(taglia, std::vector<double>(taglia));
            for (int i = 0; i < taglia; i++){
                for (int j = 0; j < taglia; j++){
                    Y[i][j] = primoprodottodecentrato1[j]*campioneascisse[i]+primoprodottodecentrato2[j]*campioneordinate[i];
                }
            }
            // Estraiamo anche di Y la matrice diagonale.
            std::vector<double> Y_diag(taglia);
            for (int i = 0; i < taglia; i++){
                Y_diag[i] = Y[i][i];
            }
            // Adesso calcoliamo la matrice Djk che contiene le distanze di Mahalanobis al quadrato.
            std::vector<std::vector<double>> Djk(taglia, std::vector<double>(taglia));
            for (int i = 0; i < taglia; i++){
                for (int j = 0; j < taglia; j++){
                    Djk[i][j] = -2.0*Y[j][i]+Y_diag[i]+Y_diag[j];
                }
            }
            // Parametro di smoothing.
            lisciatore = 1.0/std::sqrt(2.0)*std::pow(5.0/4.0, 1.0/6.0)*std::pow(static_cast<double>(taglia), 1.0/6.0);
            lisciatore2 = lisciatore*lisciatore;
            // Con questi passaggi calcoliamo la statistica test di Henze-Zirkler.
            double sommakernel = 0.0;
            double sommakernel2 = 0.0;
            for (int i = 0; i < taglia; i++){
                for (int j = 0; j < taglia; j++){
                    sommakernel += std::exp(-lisciatore2/2.0*Djk[i][j]);
                }
                sommakernel2 += std::exp(-lisciatore2/(2.0+2.0*lisciatore2)*Dj[i]);
            }
            hz = sommakernel/taglia-2.0*sommakernel2/(1.0+lisciatore2)+taglia/(1.0+2.0*lisciatore2);
        }
        // Adesso calcoliamo la media e la varianza della log-normale.
        const double wb = (1.0+lisciatore2)*(1.0+3.0*lisciatore2);
        const double a = 1.0+2.0*lisciatore2;
        const double lisciatore4 = lisciatore2*lisciatore2;
        const double a2 = a*a;
        const double mu = 1.0-(1.0+2.0*lisciatore2/a+8.0*lisciatore4/(2.0*a2))/a;
        double si2 = 2.0/(1.0+4.0*lisciatore2)+2.0/a2*(1.0+4.0*lisciatore4/a2+24.0*lisciatore4*lisciatore4
            /(4.0*a2*a2))-4.0/wb*(1.0+6.0*lisciatore4/(2.0*wb)+8.0*lisciatore4*lisciatore4/(2.0*wb*wb));
        double mu2 = mu*mu;
        if (si2 < 1e-14){si2 = 1e-14;}
        if (mu2 < 1e-14){mu2 = 1e-14;}
        if (si2/mu2 < 0.0 || si2+mu2 == 0.0){throw std::runtime_error("Problemi nei calcoli per il test di Henze-Zirkler");}
        double pmu = std::log(std::sqrt(mu2*mu2/(si2+mu2)));
        double psi = std::sqrt(std::log(1.0+si2/mu2));
        lognormal_distribution<> distribuzione(std::exp(pmu), psi);
        if (hz <= quantile(distribuzione, 1.0-alfa)){accettazioni++;}
    }
    return accettazioni;
}

// Funzione preparatoria per calcolare la probabilità di errore di I tipo effettiva del test di
// Henze-Zirkler. Va passato soltanto il numero di iterazioni e la dimensione n del campione;
// tuttavia, viene restituito un vettore di statistiche test e parametri distribuzionali, che
// devono essere elaborati esternamente al programma.
double alfaverohenzezirkler(const int iterazioni, const int taglia, const double alfa) {
    // Contiamo il numero di thread disponibili; se non ci riusciamo, impostiamone quattro.
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    // Contiamo quante iterazioni deve fare ciascun thread.
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    // Inizializziamo il vettore delle statistiche.
    std::vector<int> accettazionithread(numerothread);
    // Ora facciamo partire i thread; l'ultimo farà alcune iterazioni in più, che devono essere passate
    // come argomento separato.
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcolaalfaverohenzezirkler(taglia, volteperthread, 0, alfa);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazionithread]() {
        accettazionithread[numerothread-1] = calcolaalfaverohenzezirkler(taglia, volteperthread, iterazionirestanti, alfa);
    });
    // Aspettiamo che i thread finiscano e poi restituiamo le statistiche.
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return 1.0-static_cast<double>(accettazioni)/iterazioni;
}


// Funzione che esegue effettivamente i calcoli per trovare la probabilità effettiva di errore di I tipo
// del test di Mardia. Va passato il numero di iterazioni, la dimensione del campione, il quantile
// 1-alfa di una chi quadro con 4 gradi di libertà e il quantile 1-alfa di una normale standard.
// Viene restituito il numero di volte in cui l'ipotesi nulla viene accettata.
int calcolaalfaverotestmardia(const int iterazioni, const int taglia, const double sogliaasimmetria,
    const double sogliacurtosi) {
    if (iterazioni < 1){throw std::runtime_error("Le iterazioni devono essere positive");}
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    // Prepariamo i generatori di numeri casuali.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normale(0.0, 1.0);
    // Inizializziamo il contatore di volte in cui l'ipotesi nulla viene accettata.
    int accettazioni = 0;
    // Per ogni iterazione:
    for (int i = 0; i < iterazioni; i++) {
        // Inizializziamo i vettori dei campioni, uno per le ascisse e l'altro per le ordinate.
        // Intanto, calcoleremo il vettore medio.
        std::vector<double> campioneascisse(taglia);
        std::vector<double> campioneordinate(taglia);
        double mediaascisse = 0.0;
        double mediaordinate = 0.0;
        for (int j = 0; j < taglia; j++) {
            const double ascissa = normale(generatore);
            const double ordinata = normale(generatore);
            mediaascisse += ascissa;
            mediaordinate += ordinata;
            campioneascisse[j] = ascissa;
            campioneordinate[j] = ordinata;
        }
        mediaascisse /= taglia;
        mediaordinate /= taglia;
        // Per prima cosa calcoliamo il vettore medio, la matrice di covarianze e la sua inversa.
        // Per evitare problemi dovuti a matrici singolari, usiamo la regolarizzazione di Tikhonov.
        double varianzaascisse = 0.0;
        double varianzaordinate = 0.0;
        double covarianza = 0.0;
        for (int j = 0; j < taglia; j++){
            const double differenzaascisse = campioneascisse[j]-mediaascisse;
            const double differenzaordinate = campioneordinate[j]-mediaordinate;
            varianzaascisse += differenzaascisse*differenzaascisse;
            varianzaordinate += differenzaordinate*differenzaordinate;
            covarianza += differenzaascisse*differenzaordinate;
        }
        varianzaascisse /= taglia;
        varianzaordinate /= taglia;
        covarianza /= taglia;
        double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        if (determinante < 0.000001){
            varianzaascisse += 0.000001;
            varianzaordinate += 0.000001;
            determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        }
        const double inversa11 = varianzaordinate/determinante;
        const double inversa22 = varianzaascisse/determinante;
        const double inversa12 = -covarianza/determinante;
        // Per calcolare l'asimmetria e la curtosi, abbiamo bisogno delle quantità intermedie
        // m_{ij}=(x_i-\bar{x})^T\Sigma^{-1}(x_j-\bar{x}). Abbiamo già \bar{x} e quindi dobbiamo
        // solo ottenere le differenze (x_i-\bar{x}), per poi eseguire i prodotti matriciali;
        // salviamo gli m_{ij} ottenuti da questi nel vettore sommandi, che sarà piatto.
        std::vector<double> ascissecentrate(taglia);
        std::vector<double> ordinatecentrate(taglia);
        for (int j = 0; j < taglia; j++){
            ascissecentrate[j] = campioneascisse[j]-mediaascisse;
            ordinatecentrate[j] = campioneordinate[j]-mediaordinate;
        }
        std::vector<double> sommandi(taglia*taglia);
        int indice = 0;
        for (int j = 0; j < taglia; j++){
            for (int k = 0; k < taglia; k++){
                const double primoprodotto1 = ascissecentrate[j]*inversa11+ordinatecentrate[j]*inversa12;
                const double primoprodotto2 = ascissecentrate[j]*inversa12+ordinatecentrate[j]*inversa22;
                sommandi[indice] = primoprodotto1*ascissecentrate[k]+primoprodotto2*ordinatecentrate[k];
                indice++;
            }
        }
        // b_{12}=\sum_i\sum_jm^3_{ij}/n^2:
        double b1 = 0.0;
        for (int j = 0; j < taglia*taglia; j++){
            const double sommando = sommandi[j];
            b1 += sommando*sommando*sommando;
        }
        b1 /= taglia*taglia;
        // b_{22}=\sum_im^2_{ij}/n:
        double b2 = 0.0;
        for (int j = 0; j < taglia; j++){
            const double sommando = sommandi[j*taglia+j];
            b2 += sommando*sommando;
        }
        b2 /= taglia;
        // L'asimmetria è pari a nkb_1/6; se n >= 20, k=1, altrimenti è posto pari a una costante
        // di aggiustamento 3(n+1)(n+3)/(3n(n+1)-6) (nel caso p=2, che è il nostro).
        double asimmetria;
        if (taglia < 20) {
            asimmetria = taglia*3.0*(taglia+1.0)*(taglia+3.0)/(3.0*(taglia+1.0)*taglia-6.0)*b1/6.0;
        } else {
            asimmetria = taglia*b1/6.0;
        }
        // La curtosi, nel caso p=2, è pari a (b2-8)\sqrt{n/64}.
        const double curtosi = (b2-8.0*(taglia-1.0)/(taglia+1.0))*std::sqrt(taglia/64.0);
        // L'ipotesi nulla è rifiutata se l'asimmetria o la curtosi sono sopra le rispettive soglie critiche.
        if (asimmetria < sogliaasimmetria && curtosi < sogliacurtosi && curtosi > -sogliacurtosi){
            accettazioni++;
        }
    }
    return accettazioni;
}


// Funzione preparatoria per calcolare la probabilità effettiva di errore di I tipo del test di Mardia.
double alfaveromardia(const int iterazioni, const int taglia, const double sogliaasimmetria,
    const double sogliacurtosi){
    // Contiamo i thread disponibili; se non ci riusciamo, impostiamone quattro.
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    // Calcoliamo quante iterazioni deve fare ciascun thread.
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    // Inizializziamo il contatore di quante volte l'ipotesi nulla viene accettata e poi facciamo
    // partire i thread.
    std::vector<int> accettazionithread(numerothread);
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcolaalfaverotestmardia(volteperthread, taglia, sogliaasimmetria, sogliacurtosi);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazionithread]() {
        accettazionithread[numerothread-1] = calcolaalfaverotestmardia(volteperthread+iterazionirestanti, taglia, sogliaasimmetria, sogliacurtosi);
    });
    // Aspettiamo che i thread finiscano e restituiamo la proporzione di volte in cui l'ipotesi
    // nulla viene rifiutata.
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return 1.0-static_cast<double>(accettazioni)/iterazioni;
}


std::vector<int> kmeans(std::mt19937 &generatore, const int taglia, const std::vector<double> &ascisse,
    const std::vector<double> &ordinate, const int i, const double minimoascisse, const double massimoascisse,
    const double minimoordinate, const double massimoordinate, const int iterazionikmeans) {
    std::uniform_real_distribution<> uniformeascisse(minimoascisse, massimoascisse);
    std::uniform_real_distribution<> uniformeordinate(minimoordinate, massimoordinate);
    // Inizializziamo tanti centroidi iniziali quanti sono i cluster; li estraiamo
    // da un'uniforme con supporto pari allo spazio occupato dai dati.
    std::vector<double> ascissebaricentri(i);
    std::vector<double> ordinatebaricentri(i);
    for (int k = 0; k < i; k++) {
        ascissebaricentri[k] = uniformeascisse(generatore);
        ordinatebaricentri[k] = uniformeordinate(generatore);
    }
    // Assegniamo ogni unità al centroide più vicino.
    std::vector<int> clusterkmeans(taglia);
    for (int k = 0; k < taglia; k++) {
        int baricentro;
        double distanzaminima;
        for (int l = 0; l < i; l++) {
            const double distanzaascisse = ascisse[k] - ascissebaricentri[l];
            const double distanzaordinate = ordinate[k] - ordinatebaricentri[l];
            const double distanza = std::sqrt(distanzaascisse * distanzaascisse +
                                              distanzaordinate * distanzaordinate);
            if (l == 0) {
                baricentro = 0;
                distanzaminima = distanza;
            } else if (distanza < distanzaminima) {
                baricentro = l;
                distanzaminima = distanza;
            }
        }
        clusterkmeans[k] = baricentro;
    }
    // Adesso siamo pronti a implementare l'algoritmo delle k-medie.
    for (int k = 0; k < iterazionikmeans; k++) {
        // Dichiariamo un vettore che conterrà il numero di unità associate a ogni centroide.
        std::vector<int> tagliecluster(i, 0);
        // Salviamo i centroidi ottenuti all'iterazione precedente (o per la prima iterazione,
        // con l'inizializzazione) in due nuovi oggetti.
        std::vector<double> ascissebaricentriprecedenti = ascissebaricentri;
        std::vector<double> ordinatebaricentriprecedenti = ordinatebaricentri;
        // Riazzeriamo i centroidi in modo da poterli ricalcolare.
        for (int l = 0; l < i; l++) {
            ascissebaricentri[l] = 0.0;
            ordinatebaricentri[l] = 0.0;
        }
        // Calcoliamo i centroidi dei cluster ottenuti all'iterazione precedente.
        for (int l = 0; l < taglia; l++) {
            ascissebaricentri[clusterkmeans[l]] += ascisse[l];
            ordinatebaricentri[clusterkmeans[l]] += ordinate[l];
            tagliecluster[clusterkmeans[l]]++;
        }
        for (int l = 0; l < i; l++) {
            const int tagliacluster = tagliecluster[l];
            if (tagliacluster != 0) {
                ascissebaricentri[l] /= tagliecluster[l];
                ordinatebaricentri[l] /= tagliecluster[l];
            } else {
                ascissebaricentri[l] = ascissebaricentriprecedenti[l];
                ordinatebaricentri[l] = ordinatebaricentriprecedenti[l];
            }
        }
        // Adesso riassegniamo ogni unità al (nuovo) centroide più vicino.
        for (int l = 0; l < taglia; l++) {
            int baricentro;
            double distanzaminima;
            for (int m = 0; m < i; m++) {
                const double distanzaascisse = ascisse[l] - ascissebaricentri[m];
                const double distanzaordinate = ordinate[l] - ordinatebaricentri[m];
                const double distanza = std::sqrt(distanzaascisse * distanzaascisse +
                                                  distanzaordinate * distanzaordinate);
                if (m == 0) {
                    baricentro = 0;
                    distanzaminima = distanza;
                } else if (distanza < distanzaminima) {
                    baricentro = m;
                    distanzaminima = distanza;
                }
            }
            clusterkmeans[l] = baricentro;
        }
        // Diciamo che le k-medie sono arrivate a convergenza se le distanze tra i vecchi
        // centroidi e i corrispondenti nuovi centroidi sono tutte sotto lo 0.001.
        bool convergenza = true;
        for (int l = 0; l < i; l++) {
            const double distanzaascisse = ascissebaricentriprecedenti[l] - ascissebaricentri[l];
            const double distanzaordinate = ordinatebaricentriprecedenti[l] - ordinatebaricentri[l];
            if (std::sqrt(distanzaascisse * distanzaascisse + distanzaordinate * distanzaordinate) > 0.001) {
                convergenza = false;
                break;
            }
        }
        if (convergenza) { break; }
    }
    return clusterkmeans;
}


double silhouette(const int taglia, const std::vector<int> &clusterkmeans, const std::vector<int> &tagliecluster,
    const std::vector<double> &ascisse, const std::vector<double> &ordinate, const int i) {
    double silhouettemedia = 0.0;
    for (int k = 0; k < taglia; k++) {
        const int suocluster = clusterkmeans[k];
        const int tagliasuocluster = tagliecluster[suocluster];
        const double suaascissa = ascisse[k];
        const double suaordinata = ordinate[k];
        if (tagliasuocluster > 1) {
            std::vector<double> distanzedacluster(i, 0.0);
            for (int l = 0; l < taglia; l++) {
                const double differenzaascisse = suaascissa-ascisse[l];
                const double differenzaordinate = suaordinata-ordinate[l];
                const double distanza = std::sqrt(differenzaascisse*differenzaascisse+
                    differenzaordinate*differenzaordinate);
                distanzedacluster[clusterkmeans[l]] += distanza;
            }
            double bi;
            if (suocluster == 0) {
                bi = distanzedacluster[1]/tagliecluster[1];
            } else {bi = distanzedacluster[0]/tagliecluster[0];}
            for (int l = 0; l < i; l++) {
                const double nuovadistanza = distanzedacluster[l]/tagliecluster[l];
                if (nuovadistanza < bi){bi = nuovadistanza;}
            }
            const double ai = distanzedacluster[suocluster]/(tagliasuocluster-1);
            if (ai > bi){silhouettemedia += (bi-ai)/ai;} else {silhouettemedia += (bi-ai)/bi;}
        }
    }
    silhouettemedia /= taglia;
    return silhouettemedia;
}


void inizializza(const int taglia, const double piminimo, std::mt19937 &generatore,
    const std::vector<int> &clustercorrenti, const std::vector<int> &taglieclustercorrenti,
    const std::vector<double> &ascisse, const std::vector<double> &ordinate,
    std::vector<double> &muascisse, std::vector<double> &muordinate, std::vector<double> &pigreci,
    std::vector<double> &sigma11, std::vector<double> &sigma22, std::vector<double> &sigma12,
    const int i) {
    std::uniform_real_distribution<> inizializzatore(-1.0, 1.0);
    for (int k = 0; k < taglia; k++) {
        const int suocluster = clustercorrenti[k];
        muascisse[suocluster] += ascisse[k];
        muordinate[suocluster] += ordinate[k];
    }
    for (int k = 0; k < i; k++) {
        const int tagliadelcluster = taglieclustercorrenti[k];
        if (tagliadelcluster == 0) {
            muascisse[k] = inizializzatore(generatore);
            muordinate[k] = inizializzatore(generatore);
            pigreci[k] = piminimo;
            continue;
        }
        muascisse[k] /= tagliadelcluster;
        muordinate[k] /= tagliadelcluster;
        pigreci[k] = static_cast<double>(tagliadelcluster)/taglia;
        if (pigreci[k] < piminimo){pigreci[k] = piminimo;}
    }
    for (int k = 0; k < taglia; k++) {
        const int suocluster = clustercorrenti[k];
        const double differenzaascisse = ascisse[k]-muascisse[suocluster];
        const double differenzaordinate = ordinate[k]-muordinate[suocluster];
        sigma11[suocluster] += differenzaascisse*differenzaascisse;
        sigma22[suocluster] += differenzaordinate*differenzaordinate;
        sigma12[suocluster] += differenzaascisse*differenzaordinate;
    }
    for (int k = 0; k < i; k++) {
        const int tagliadelcluster = taglieclustercorrenti[k];
        if (tagliadelcluster < 2) {
            sigma11[k] = 0.01;
            sigma22[k] = 0.01;
            sigma12[k] = 0.0;
            continue;
        }
        sigma11[k] /= tagliadelcluster-1;
        sigma22[k] /= tagliadelcluster-1;
        sigma12[k] /= tagliadelcluster-1;
    }
}


void inverti(std::vector<double> &sigma11, std::vector<double> &sigma22, std::vector<double> &sigma12,
    std::vector<double> &determinanti, std::vector<double> &inverse11, std::vector<double> &inverse22,
    std::vector<double> &inverse12, const int i) {
    for (int k = 0; k < i; k++) {
        const double prodottovarianze = std::sqrt(sigma11[k]*sigma22[k]);
        double covarianzacluster = sigma12[k];
        if (covarianzacluster <= -prodottovarianze || covarianzacluster >= prodottovarianze) {
            covarianzacluster = std::copysign(prodottovarianze, covarianzacluster);
            sigma12[k] = covarianzacluster;
        }
        double determinante = sigma11[k]*sigma22[k]-covarianzacluster*covarianzacluster;
        if (determinante < 0.000001) {
            sigma11[k] += 0.001;
            sigma22[k] += 0.001;
            determinante = sigma11[k]*sigma22[k]-covarianzacluster*covarianzacluster;
        }
        determinanti[k] = determinante;
        inverse11[k] = sigma22[k]/determinante;
        inverse22[k] = sigma11[k]/determinante;
        inverse12[k] = -covarianzacluster/determinante;
    }
}


double logverosimiglia(const int i, const int taglia, const std::vector<double> &pigreci,
    const std::vector<double> &determinanti, const std::vector<double> &muascisse, const std::vector<double> &muordinate,
    const std::vector<double> &inverse11, const std::vector<double> &inverse22, const std::vector<double> &inverse12,
    const std::vector<double> &nu, const std::vector<double> &ascisse, const std::vector<double> &ordinate,
    std::vector<std::vector<double>> &logverosimiglianzepigreci, const bool normale) {
    double logverosimiglianzacorrente = 0.0;
    if (normale) {
        std::vector<double> precostanti(i);
        for (int l = 0; l < i; l++){precostanti[l] = std::log(pigreci[l])-logduepigreco-std::log(determinanti[l])/2.0;}
        for (int l = 0; l < taglia; l++) {
            std::vector<double> esponenti(i);
            double esponentemassimo = -std::numeric_limits<double>::infinity();
            const double ascissa = ascisse[l];
            const double ordinata = ordinate[l];
            for (int m = 0; m < i; m++) {
                const double differenzaascissa = ascissa-muascisse[m];
                const double differenzaordinata = ordinata-muordinate[m];
                const double esponente = precostanti[m]-
                    (inverse11[m]*differenzaascissa*differenzaascissa+2.0*inverse12[m]*differenzaascissa*
                        differenzaordinata+inverse22[m]*differenzaordinata*differenzaordinata)/2.0;
                esponenti[m] = esponente;
                if (esponente > esponentemassimo){esponentemassimo = esponente;}
            }
            logverosimiglianzepigreci[l] = esponenti;
            double sommatoria = 0.0;
            for (int m = 0; m < i; m++){sommatoria += std::exp(esponenti[m]-esponentemassimo);}
            logverosimiglianzacorrente += esponentemassimo+std::log(sommatoria);
        }
    } else {
        std::vector<double> precostanti(i);
        for (int l = 0; l < i; l++) {
            const double grado = nu[l];
            precostanti[l] = std::log(pigreci[l])+std::lgamma(grado/2.0+1.0)-std::log(grado*pigreco)-
                std::log(determinanti[l])/2.0-std::lgamma(grado/2.0);
        }
        for (int l = 0; l < taglia; l++) {
            std::vector<double> esponenti(i);
            double esponentemassimo = -std::numeric_limits<double>::infinity();
            const double ascissa = ascisse[l];
            const double ordinata = ordinate[l];
            for (int m = 0; m < i; m++){
                const double differenzaascissa = ascissa-muascisse[m];
                const double differenzaordinata = ordinata-muordinate[m];
                const double grado = nu[m];
                const double esponente = precostanti[m]-(grado+2.0)/2.0*std::log(1.0+
                    (inverse11[m]*differenzaascissa*differenzaascissa+2.0*inverse12[m]*
                        differenzaascissa*differenzaordinata+inverse22[m]*differenzaordinata*
                            differenzaordinata)/grado);
                esponenti[m] = esponente;
                if (esponente > esponentemassimo){esponentemassimo = esponente;}
            }
            logverosimiglianzepigreci[l] = esponenti;
            double sommatoria = 0.0;
            for (int m = 0; m < i; m++){sommatoria += std::exp(esponenti[m]-esponentemassimo);}
            logverosimiglianzacorrente += esponentemassimo+std::log(sommatoria);
        }
    }
    return logverosimiglianzacorrente;
}


void aggiornapesi(const int taglia, const std::vector<std::vector<double>> &logverosimiglianzepigreci,
    std::vector<std::vector<double>> &pesi, const int i){
    // Iniziamo calcolando i pesi w_{ig}.
    for (int l = 0; l < taglia; l++) {
        // Ricordiamo che w_{ig}=\pi_gf_g/\sum_g\pi_gf_g. Abbiamo già immagazzinato
        // le quantità \pi_gf_g in logverosimiglianzepigreci.
        const std::vector<double> wi = logverosimiglianzepigreci[l];
        double wimassimo = wi[0];
        for (int m = 1; m < i; m++) {
            if (wi[m] > wimassimo){wimassimo = wi[m];}
        }
        double sommatoria = 0.0;
        for (int m = 0; m < i; m++){sommatoria += std::exp(wi[m]-wimassimo);}
        if (sommatoria <= 0.0) {
            for (int m = 0; m < i; m++){pesi[l][m] = 1.0/i;}
        } else {
            for (int m = 0; m < i; m++) {
                pesi[l][m] = std::exp(wi[m]-wimassimo-std::log(sommatoria));
            }
        }
}
}


void aggiornaparametri(const int i, const int taglia, const std::vector<std::vector<double>> &pesi,
    std::vector<double> &muascisse, std::vector<double> &muordinate, std::vector<double> &sigma11,
    std::vector<double> &sigma22, std::vector<double> &sigma12, const std::vector<double> &ascisse,
    const std::vector<double> &ordinate, std::vector<double> &pigreci, const double piminimo,
    std::mt19937 &generatore, const std::vector<std::vector<double>> &uig, const bool normale) {
    std::uniform_real_distribution<> inizializzatore(-1.0, 1.0);
    if (normale){
        // Per aggiornare le stime dei parametri, ci serve conoscere \sum_iw_{ig}.
        std::vector<double> sommewi(i, 0.0);
        for (int l = 0; l < taglia; l++) {
            for (int m = 0; m < i; m++) {
                sommewi[m] += pesi[l][m];
            }
        }
        // Ora aggiorniamo la stima di \pi_g e \mu_g.
        for (int l = 0; l < i; l++) {
            muascisse[l] = 0.0;
            muordinate[l] = 0.0;
        }
        for (int l = 0; l < taglia; l++) {
            const std::vector<double> rigapesi = pesi[l];
            const double ascissa = ascisse[l];
            const double ordinata = ordinate[l];
            for (int m = 0; m < i; m++) {
                const double peso = rigapesi[m];
                muascisse[m] += peso*ascissa;
                muordinate[m] += peso*ordinata;
            }
        }
        for (int l = 0; l < i; l++) {
            double sommawi = sommewi[l];
            if (sommawi < 0.000001) {
                pigreci[l] = piminimo;
                muascisse[l] = inizializzatore(generatore);
                muordinate[l] = inizializzatore(generatore);
                sigma11[l] = 0.0;
                sigma22[l] = 0.0;
                sigma12[l] = 0.0;
                continue;
            }
            pigreci[l] = sommawi/taglia;
            if (pigreci[l] < piminimo){pigreci[l] = piminimo;}
            muascisse[l] /= sommawi;
            muordinate[l] /= sommawi;
            // Riazzeriamo nello stesso ciclo le stime delle matrici di covarianze.
            sigma11[l] = 0.0;
            sigma22[l] = 0.0;
            sigma12[l] = 0.0;
        }
        // Adesso invece aggiorniamo la stima delle matrici di covarianze.
        for (int l = 0; l < taglia; l++) {
            const std::vector<double> rigapesi = pesi[l];
            const double ascissa = ascisse[l];
            const double ordinata = ordinate[l];
            for (int m = 0; m < i; m++) {
                const double peso = rigapesi[m];
                const double differenzaascisse = ascissa-muascisse[m];
                const double differenzaordinate = ordinata-muordinate[m];
                sigma11[m] += peso*differenzaascisse*differenzaascisse;
                sigma22[m] += peso*differenzaordinate*differenzaordinate;
                sigma12[m] += peso*differenzaascisse*differenzaordinate;
            }
        }
        for (int l = 0; l < i; l++) {
            double sommawi = sommewi[l];
            if (sommawi < 0.000001) {
                sigma11[l] = 0.01;
                sigma22[l] = 0.01;
                sigma12[l] = 0.0;
                continue;
            }
            sigma11[l] /= sommawi;
            sigma22[l] /= sommawi;
            sigma12[l] /= sommawi;
        }
    } else {
        // Per aggiornare le stime dei parametri, ci serve conoscere \sum_iw_{ig}.
        std::vector<double> sommewi(i, 0.0);
        for (int l = 0; l < taglia; l++) {
            for (int m = 0; m < i; m++) {
                sommewi[m] += pesi[l][m];
            }
        }
        for (int l = 0; l < i; l++) {
            muascisse[l] = 0.0;
            muordinate[l] = 0.0;
            pigreci[l] = sommewi[l]/taglia;
            if (pigreci[l] < piminimo){pigreci[l] = piminimo;}
        }
        std::vector<double> denominatori(i, 0.0);
        for (int l = 0; l < taglia; l++) {
            const double ascissa = ascisse[l];
            const double ordinata = ordinate[l];
            const std::vector<double> rigapesi = pesi[l];
            const std::vector<double> rigau = uig[l];
            for (int m = 0; m < i; m++) {
                const double pesou = rigapesi[m]*rigau[m];
                muascisse[m] += pesou*ascissa;
                muordinate[m] += pesou*ordinata;
                denominatori[m] += pesou;
            }
        }
        for (int l = 0; l < i; l++) {
            double denominatore = denominatori[l];
            if (denominatore < 0.000001){
                muascisse[l] = inizializzatore(generatore);
                muordinate[l] = inizializzatore(generatore);
                sigma11[l] = 0.0;
                sigma22[l] = 0.0;
                sigma12[l] = 0.0;
                continue;
            }
            muascisse[l] /= denominatore;
            muordinate[l] /= denominatore;
            // Riazzeriamo nello stesso ciclo le stime delle matrici di covarianze.
            sigma11[l] = 0.0;
            sigma22[l] = 0.0;
            sigma12[l] = 0.0;
        }
        for (int l = 0; l < taglia; l++) {
            const double ascissa = ascisse[l];
            const double ordinata = ordinate[l];
            const std::vector<double> rigapesi = pesi[l];
            const std::vector<double> rigau = uig[l];
            for (int m = 0; m < i; m++) {
                const double differenzaascisse = ascissa-muascisse[m];
                const double differenzaordinate = ordinata-muordinate[m];
                const double pesou = rigapesi[m]*rigau[m];
                sigma11[m] += pesou*differenzaascisse*differenzaascisse;
                sigma22[m] += pesou*differenzaordinate*differenzaordinate;
                sigma12[m] += pesou*differenzaascisse*differenzaordinate;
            }
        }
        for (int l = 0; l < i; l++) {
            double sommawi = sommewi[l];
            if (sommawi < 0.000001) {
                sigma11[l] = 0.01;
                sigma22[l] = 0.01;
                sigma12[l] = 0.0;
                continue;
            }
            sigma11[l] /= sommawi;
            sigma22[l] /= sommawi;
            sigma12[l] /= sommawi;
        }
    }
}


std::vector<std::vector<double>> pesiU(const int taglia, const std::vector<double> &ascisse,
    const std::vector<double> &ordinate, const std::vector<double> &muascisse, const std::vector<double> &muordinate,
    const std::vector<double> &inverse11, const std::vector<double> &inverse22, const std::vector<double> &inverse12,
    const std::vector<double> &nu, const int i) {
    std::vector<std::vector<double>> uig(taglia, std::vector<double>(i));
    for (int l = 0; l < taglia; l++) {
        const double ascissa = ascisse[l];
        const double ordinata = ordinate[l];
        for (int m = 0; m < i; m++) {
            const double differenzaascisse = ascissa-muascisse[m];
            const double differenzaordinate = ordinata-muordinate[m];
            const double mahalanobis = inverse11[m]*differenzaascisse*differenzaascisse+
                inverse12[m]*differenzaascisse*differenzaordinate+
                inverse22[m]*differenzaordinate*differenzaordinate;
            const double gradi = nu[m];
            uig[l][m] = (gradi+2.0)/(gradi+mahalanobis);
        }
    }
    return uig;
}


void aggiornagradi(std::vector<double> &nu, const std::vector<std::vector<double>> &pesi,
    const std::vector<std::vector<double>> &uig, const int taglia, const int i) {
    std::vector<double> sommewi(i, 0.0);
    for (int l = 0; l < taglia; l++) {
        for (int m = 0; m < i; m++) {
            sommewi[m] += pesi[l][m];
        }
    }
    // Per trovare la stima aggiornata dei gradi di libertà, dobbiamo risolvere (con un metodo
    // iterativo) l'equazione -\psi(\nu/2)+log(\nu/2)+1+(1/\nu)\sum_iw_{ig}(logu_{ig}-u_{ig})+
    // \psi((\nu+2)/2)-log((\nu+2)/2)=0. Useremo il metodo di Newton, dunque l'equazione di
    // aggiornamento sarà x_t=x_{t-1}-f(x_{t-1})/f'(x_{t-1}).
    std::vector<double> sommeru(i, 0.0);
    for (int l = 0; l < taglia; l++) {
        const std::vector<double> rigapesi = pesi[l];
        const std::vector<double> rigau = uig[l];
        for (int m = 0; m < i; m++) {
            double u = rigau[m];
            if (u < 0.000001){u += 0.000001;}
            sommeru[m] += rigapesi[m]*(std::log(u)-u);
        }
    }
    for (int l = 0; l < i; l++){
        double grado = nu[l];
        if (grado <= 0.0){grado = 0.001;}
        const double gradoiniziale = grado;
        double grado1 = gradoiniziale/2.0+1.0;
        double digamma1 = 0.0;
        while (grado1 < 6.0) {
            digamma1 -= 1.0/grado1;
            grado1++;
        }
        const double g2 = grado1*grado1;
        const double g4 = g2*g2;
        const double g8 = g4*g4;
        digamma1 += std::log(grado1)-1.0/(2.0*grado1)-1.0/(12.0*g2)+1.0/(120.0*g4)-1.0/(252.0*g4*g2)+
            1.0/(240.0*g8)-1.0/(132.0*g8*g2);
        const double costantenumeratore = digamma1+1.0+sommeru[l]/sommewi[l];
        for (int m = 0; m < 20; m++) {
            double grado2 = grado/2.0;
            double digamma2 = 0.0;
            double trigamma = 0.0;
            while (grado2 < 6.0) {
                digamma2 -= 1.0/grado2;
                trigamma += 1.0/(grado2*grado2);
                grado2++;
            }
            const double gg2 = grado2*grado2;
            const double gg4 = gg2*gg2;
            const double gg8 = gg4*gg4;
            digamma2 += std::log(grado2)-1.0/(2.0*grado2)-1.0/(12.0*gg2)+1.0/(120.0*gg4)-
                1.0/(252.0*gg4*gg2)+1.0/(240.0*gg8)-1.0/(132.0*gg8*gg2);
            trigamma += 1.0/grado2+1.0/(2.0*gg2)+1.0/(6.0*gg2*grado2)-1.0/(30.0*gg4*grado2)+
                1.0/(42.0*gg4*gg2*grado2)-1.0/(30.0*gg8*grado2)+10.0/(132.0*gg8*gg2*grado2);
            const double logaritmo = std::log(grado/(gradoiniziale+2.0));
            const double reciproco = 1.0/grado;
            grado -= (costantenumeratore-digamma2+logaritmo)/(reciproco-0.5*trigamma);
            if (grado <= 0.0){grado = 0.001;}
            if (grado > 200.0){grado = 200.0;}
        }
        nu[l] = grado;
    }
}


void EM(const bool normale, const int iterazioniEM, const int taglia, const int i,
    std::vector<std::vector<double>> &logverosimiglianzepigreci, std::vector<double> &pigreci,
    std::vector<double> &muascisse, std::vector<double> &muordinate, std::vector<double> &sigma11,
    std::vector<double> &sigma22, std::vector<double> &sigma12, std::vector<double> &determinanti,
    std::vector<double> &inverse11, std::vector<double> &inverse22, std::vector<double> &inverse12, std::vector<double> &nu,
    std::vector<std::vector<double>> &pesi, const std::vector<double> &ascisse, const std::vector<double> &ordinate,
    const double piminimo, std::mt19937 &generatore, double &logverosimiglianzacorrente, const double sogliaconvergenza) {
    std::uniform_real_distribution<> inizializzatore(-1.0, 1.0);
    // Prepariamo un array che contenga le logverosimiglianze delle due iterazioni precedenti, in
    // modo da poter usare l'accelerazione di Aitken e monitorare la convergenza.
    std::array<double, 2> logverosimiglianzeprecedenti = {0.0, 0.0};
    // Se usiamo misture di gaussiane:
    if (normale) {
        std::vector<std::vector<double>> uigvuoti(taglia, std::vector<double>(i, 0.0));
        // A ogni iterazione dell'algoritmo EM:
        for (int k = 0; k < iterazioniEM; k++) {
            aggiornapesi(taglia, logverosimiglianzepigreci, pesi, i);
            aggiornaparametri(i, taglia, pesi, muascisse, muordinate, sigma11, sigma22, sigma12,
                ascisse, ordinate, pigreci, piminimo, generatore, uigvuoti, normale);
            // Delle matrici di covarianze calcoliamo anche le inverse.
            // Usiamo come sempre la regolarizzazione di Tikhonov.
            inverti(sigma11, sigma22, sigma12, determinanti, inverse11, inverse22, inverse12, i);
            // Adesso calcoliamo il valore attuale della logverosimiglianza.
            logverosimiglianzeprecedenti = {logverosimiglianzeprecedenti[1], logverosimiglianzacorrente};
            logverosimiglianzacorrente = logverosimiglia(i, taglia, pigreci, determinanti, muascisse, muordinate,
                inverse11, inverse22, inverse12, nu, ascisse, ordinate, logverosimiglianzepigreci, normale);
            // Dalla seconda iterazione in poi, utilizziamo l'accelerazione di Aitken
            // per controllare la convergenza. Se la regola d'arresto |l_\infty-l_{r-1}|<\epsilon
            // si rivela vera, allora interrompiamo l'algoritmo EM.
            if (k != 0) {
                const double precedente = logverosimiglianzeprecedenti[1];
                if (precedente-logverosimiglianzeprecedenti[0] < 0.000001){break;}
			    double accelerazione = (logverosimiglianzacorrente - precedente) /
			        (precedente - logverosimiglianzeprecedenti[0]);
			    if (accelerazione == 1.0) {accelerazione -= 0.000001;}
                const double linfinito = precedente + 1.0 / (1.0 - accelerazione) *
                    (logverosimiglianzacorrente - precedente);
                if (std::abs(linfinito - precedente) < sogliaconvergenza) { break; }
            }
        }
        // Se invece usiamo misture di t di Student:
    } else {
        // Per ogni iterazione dell'algoritmo EM:
        for (int k = 0; k < iterazioniEM; k++) {
            aggiornapesi(taglia, logverosimiglianzepigreci, pesi, i);
            // Adesso calcoliamo anche i pesi u_{ig}, che ricordiamo essere
            // (\nu_p+p)/(\nu_p+distanza di Mahalanobis di x_i da \mu_g e \Sigma_g).
            std::vector<std::vector<double>> uig = pesiU(taglia, ascisse, ordinate, muascisse, muordinate,
                inverse11, inverse22, inverse12, nu, i);
            // Adesso possiamo aggiornare le stime dei parametri.
            aggiornaparametri(i, taglia, pesi, muascisse, muordinate, sigma11, sigma22, sigma12,
                ascisse, ordinate, pigreci, piminimo, generatore, uig, normale);
            // Inoltre calcoliamo le inverse e i determinanti di tali matrici.
            inverti(sigma11, sigma22, sigma12, determinanti, inverse11, inverse22, inverse12, i);
            aggiornagradi(nu, pesi, uig, taglia, i);
            // Ora calcoliamo la logverosimiglianza.
            logverosimiglianzeprecedenti = {logverosimiglianzeprecedenti[1], logverosimiglianzacorrente};
            logverosimiglianzacorrente = logverosimiglia(i, taglia, pigreci, determinanti, muascisse, muordinate,
                inverse11, inverse22, inverse12, nu, ascisse, ordinate, logverosimiglianzepigreci, normale);
            // Dalla seconda iterazione in poi controlliamo la convergenza con l'accelerazione di Aitken.
            if (k != 0) {
                const double precedente = logverosimiglianzeprecedenti[1];
                if (precedente-logverosimiglianzeprecedenti[0] < 0.000001){break;}
			    double accelerazione = (logverosimiglianzacorrente - precedente)/
			        (precedente - logverosimiglianzeprecedenti[0]);
			    if (accelerazione == 1.0) {accelerazione -= 0.000001;}
                const double linfinito = precedente + 1 / (1 - accelerazione) *
                    (logverosimiglianzacorrente - precedente);
                if (std::abs(linfinito - precedente) < sogliaconvergenza) { break; }
            }
        }
    }
}


void SEM(const bool normale, const int iterazioniEM, const int taglia, const int i,
    std::vector<std::vector<double>> &logverosimiglianzepigreci, std::vector<std::vector<double>> &pesi,
    std::vector<double> &pigreci, std::vector<double> &muascisse, std::vector<double> &muordinate,
    std::vector<double> &sigma11, std::vector<double> &sigma22, std::vector<double> &sigma12, std::vector<double> &nu,
    const std::vector<double> &ascisse, const std::vector<double> &ordinate, std::mt19937 &generatore,
    std::vector<double> &determinanti, std::vector<double> &inverse11, std::vector<double> &inverse22,
    std::vector<double> &inverse12, const double piminimo, double &logverosimiglianzacorrente,
    double &logverosimiglianzavincente, std::vector<double> &pigrecivincenti, std::vector<double> &muascissevincenti,
    std::vector<double> &muordinatevincenti, std::vector<double> &sigma11vincenti, std::vector<double> &sigma22vincenti,
    std::vector<double> &sigma12vincenti, std::vector<double> &nuvincenti,
    std::vector<std::vector<double>> &pesivincenti, const double sogliacontrollo){
    std::uniform_real_distribution<> inizializzatore(-1.0, 1.0);
    // Se usiamo misture di gaussiane:
    if (normale) {
        // Per ogni iterazione dell'algoritmo SEM:
        for (int k = 0; k < iterazioniEM; k++) {
            // Iniziamo calcolando i pesi w_{ig}.
            aggiornapesi(taglia, logverosimiglianzepigreci, pesi, i);
            // L'algoritmo SEM differisce dall'EM perché una volta calcolati i pesi w_{ig},
            // si estraggono dei z^{pse}_i per ogni unità da una multinomiale
            // con probabilità w_{ig}. Da questi si ricavano degli pseudopesi \delta_{ig}
            // che sono semplicemente pari a 1 se z_i=g e nulli altrimenti.
            std::vector<std::vector<double>> pseudi(taglia, std::vector<double>(i, 0.0));
            for (int l = 0; l < taglia; l++) {
                const std::vector<double> probabilita = pesi[l];
                // Sei sicuro che escano pesi non tutti nulli e tutti non negativi?
                std::discrete_distribution<> multinomiale(probabilita.begin(), probabilita.end());
                const int z = multinomiale(generatore);
                pseudi[l][z] = 1.0;
            }
            std::vector<std::vector<double>> uigvuoti(taglia, std::vector<double>(i, 0.0));
            aggiornaparametri(i, taglia, pesi, muascisse, muordinate, sigma11, sigma22, sigma12, ascisse,
                ordinate, pigreci, piminimo, generatore, uigvuoti, normale);
            inverti(sigma11, sigma22, sigma12, determinanti, inverse11, inverse22, inverse12, i);
            // Se abbiamo passato un quarto delle iterazioni, dopo aver calcolato il valore della
            // logverosimiglianza vediamo se abbiamo superato il record precedente.
            logverosimiglianzacorrente = logverosimiglia(i, taglia, pigreci, determinanti, muascisse,
                muordinate, inverse11, inverse22, inverse12, nu, ascisse, ordinate, logverosimiglianzepigreci, normal);
            if (k > sogliacontrollo && logverosimiglianzacorrente > logverosimiglianzavincente){
                logverosimiglianzavincente = logverosimiglianzacorrente;
                pigrecivincenti = pigreci;
                muascissevincenti = muascisse;
                muordinatevincenti = muordinate;
                sigma11vincenti = sigma11;
                sigma22vincenti = sigma22;
                sigma12vincenti = sigma12;
                pesivincenti = pesi;
            }
            // Nel caso dell'algoritmo SEM, non controlliamo la convergenza:
            // ricordiamo che l'algoritmo SEM non converge puntualmente, ma invece genera
            // una sequenza di stime dei parametri che sono una catena di Markov con distribuzione
            // centrata sullo stimatore di massima verosimiglianza dei parametri stessi.
        }
        // Se invece usiamo misture di t di Student:
    } else {
        // Per ogni iterazione dell'algoritmo SEM:
        for (int k = 0; k < iterazioniEM; k++) {
            // Per prima cosa sfruttiamo le logverosimiglianze calcolate all'iterazione
            // precedente per trovare i pesi w_{ig}.
            aggiornapesi(taglia, logverosimiglianzepigreci, pesi, i);
            // L'algoritmo SEM differisce dall'EM perché una volta calcolati i pesi w_{ig},
            // si estraggono dei z^{pse}_i per ogni unità da una multinomiale
            // con probabilità w_{ig}. Da questi si ricavano degli pseudopesi \delta_{ig}
            // che sono semplicemente pari a 1 se z_i=g e nulli altrimenti.
            std::vector<std::vector<double>> pseudi(taglia, std::vector<double>(i, 0.0));
            for (int l = 0; l < taglia; l++) {
                const std::vector<double> probabilita = pesi[l];
                std::discrete_distribution<> multinomiale(probabilita.begin(), probabilita.end());
                const int z = multinomiale(generatore);
                pseudi[l][z] = 1.0;
            }
            std::vector<std::vector<double>> uig = pesiU(taglia, ascisse, ordinate, muascisse, muordinate,
                inverse11, inverse22, inverse12, nu, i);
            aggiornaparametri(i, taglia, pesi, muascisse, muordinate, sigma11, sigma22, sigma12, ascisse,
                ordinate, pigreci, piminimo, generatore, uig, normale);
            aggiornagradi(nu, pesi, uig, taglia, i);
            logverosimiglianzacorrente = logverosimiglia(i, taglia, pigreci, determinanti, muascisse,
                muordinate, inverse11, inverse22, inverse12, nu, ascisse, ordinate, logverosimiglianzepigreci, normale);
            // Se siamo a oltre un quarto delle iterazioni, dopo aver calcolato il valore della
            // logverosimiglianza vediamo se supera il record attuale.
            if (k > sogliacontrollo && logverosimiglianzacorrente > logverosimiglianzavincente){
                logverosimiglianzavincente = logverosimiglianzacorrente;
                pigrecivincenti = pigreci;
                muascissevincenti = muascisse;
                muordinatevincenti = muordinate;
                sigma11vincenti = sigma11;
                sigma22vincenti = sigma22;
                sigma12vincenti = sigma12;
                nuvincenti = nu;
                pesivincenti = pesi;
            }
            // Nel caso dell'algoritmo SEM, non controlliamo la convergenza:
            // ricordiamo che l'algoritmo SEM non converge puntualmente, ma invece genera
            // una sequenza di stime dei parametri che sono una catena di Markov con distribuzione
            // centrata sullo stimatore di massima verosimiglianza dei parametri stessi.
        }
    }
}


void CEM(const bool normale, const int iterazioniEM, const int taglia, const int i,
    std::vector<std::vector<double>> &logverosimiglianzepigreci, std::vector<double> &pigreci,
    std::vector<double> &muascisse, std::vector<double> &muordinate, std::vector<double> &sigma11,
    std::vector<double> &sigma22, std::vector<double> &sigma12, std::vector<double> &determinanti,
    std::vector<double> &inverse11, std::vector<double> &inverse22, std::vector<double> &inverse12, std::vector<double> &nu,
    std::vector<std::vector<double>> &pesi, const std::vector<double> &ascisse, const std::vector<double> &ordinate,
    const double piminimo, std::mt19937 &generatore, double &logverosimiglianzacorrente, const double sogliaconvergenza) {
    std::uniform_real_distribution<> inizializzatore(-1.0, 1.0);
    // Prepariamo un array che contenga le logverosimiglianze delle due iterazioni precedenti,
    // in modo da poter usare l'accelerazione di Aitken e monitorare la convergenza.
    std::array<double, 2> logverosimiglianzeprecedenti = {0.0, 0.0};
    // Se usiamo misture di gaussiane:
    if (normale) {
        // Per ogni iterazione dell'algoritmo CEM:
        for (int k = 0; k < iterazioniEM; k++) {
            // Per prima cosa sfruttiamo le logverosimiglianze calcolate all'iterazione
            // precedente per trovare i pesi w_{ig}.
            aggiornapesi(taglia, logverosimiglianzepigreci, pesi, i);
            // La differenza dell'algoritmo CEM rispetto all'EM è che dopo aver aggiornato
            // i pesi w_{ig}, si genera una partizione dei dati assegnando ogni osservazione i
            // al cluster g per cui w_{ig} è massimo; in seguito, quando ciascun parametro
            // viene aggiornato, nelle formule vengono utilizzati solo i dati che sono stati
            // assegnati al cluster corrispondente.
            std::vector<std::vector<double>> ascissepartizione(i, std::vector<double>(taglia));
            std::vector<std::vector<double>> ordinatepartizione(i, std::vector<double>(taglia));
            std::vector<int> tagliepartizione(i, 0);
            for (int l = 0; l < taglia; l++) {
                double pesomaggiore = pesi[l][0];
                int clusterunita = 0;
                for (int m = 1; m < i; m++) {
                    if (pesi[l][m] > pesomaggiore) {
                        pesomaggiore = pesi[l][m];
                        clusterunita = m;
                    }
                }
                const int indice = tagliepartizione[clusterunita];
                ascissepartizione[clusterunita][indice] = ascisse[l];
                ordinatepartizione[clusterunita][indice] = ordinate[l];
                tagliepartizione[clusterunita]++;
            }
            // Per aggiornare le stime dei parametri, nel caso del CEM, non è necessario
            // sommare i pesi: infatti, questi scompaiono dalle equazioni di aggiornamento
            // e le loro somme sono sostituite da n_g.
            // Ora aggiorniamo la stima dei parametri.
            for (int l = 0; l < i; l++) {
                const int tagliapartizione = tagliepartizione[l];
                if (tagliapartizione > 1){
                    const std::vector<double> ascissecorrenti = ascissepartizione[l];
                    const std::vector<double> ordinatecorrenti = ordinatepartizione[l];
                    double mediaascissecorrente = 0.0;
                    double mediaordinatecorrente = 0.0;
                    for (int m = 0; m < tagliapartizione; m++){
                        mediaascissecorrente += ascissecorrenti[m];
                        mediaordinatecorrente += ordinatecorrenti[m];
                    }
                    pigreci[l] = static_cast<double>(tagliapartizione)/taglia;
                    if (pigreci[l] < piminimo){pigreci[l] = piminimo;}
                    mediaascissecorrente /= tagliapartizione;
                    mediaordinatecorrente /= tagliapartizione;
                    muascisse[l] = mediaascissecorrente;
                    muordinate[l] = mediaordinatecorrente;
                    double varianzaascisse = 0.0;
                    double varianzaordinate = 0.0;
                    double covarianza = 0.0;
                    for (int m = 0; m < tagliapartizione; m++){
                        const double differenzaascisse = ascissecorrenti[m]-mediaascissecorrente;
                        const double differenzaordinate = ordinatecorrenti[m]-mediaordinatecorrente;
                        varianzaascisse += differenzaascisse*differenzaascisse;
                        varianzaordinate += differenzaordinate*differenzaordinate;
                        covarianza += differenzaascisse*differenzaordinate;
                    }
                    if (tagliapartizione != 1) {
                        varianzaascisse /= tagliapartizione-1;
                        varianzaordinate /= tagliapartizione-1;
                        covarianza /= tagliapartizione-1;
                    }
                    const double prodottovarianze = std::sqrt(varianzaascisse*varianzaordinate);
                    if (covarianza < -prodottovarianze || covarianza > prodottovarianze) {
                        covarianza = std::copysign(prodottovarianze, covarianza);
                    }
                    double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
                    if (determinante < 0.000001) {
                        varianzaascisse += 0.001;
                        varianzaordinate += 0.001;
                        determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
                    }
                    inverse11[l] = varianzaordinate/determinante;
                    inverse22[l] = varianzaascisse/determinante;
                    inverse12[l] = -covarianza/determinante;
                    sigma11[l] = varianzaascisse;
                    sigma22[l] = varianzaordinate;
                    sigma12[l] = covarianza;
                    determinanti[l] = determinante;
                } else {
                    pigreci[l] = piminimo;
                    muascisse[l] = inizializzatore(generatore);
                    muordinate[l] = inizializzatore(generatore);
                    sigma11[l] = 0.01;
                    sigma22[l] = 0.01;
                    sigma12[l] = 0.0;
                    inverse11[l] = 100.0;
                    inverse22[l] = 100.0;
                    inverse12[l] = 0.0;
                    determinanti[l] = 0.0001;
                }
            }
            // Adesso calcoliamo il valore attuale della logverosimiglianza.
            logverosimiglianzeprecedenti = {logverosimiglianzeprecedenti[1], logverosimiglianzacorrente};
            logverosimiglianzacorrente = logverosimiglia(i, taglia, pigreci, determinanti, muascisse,
                muordinate, inverse11, inverse22, inverse12, nu, ascisse, ordinate, logverosimiglianzepigreci, normale);
            // Dalla seconda iterazione in poi, utilizziamo l'accelerazione di Aitken
            // per controllare la convergenza. Se la regola d'arresto |l_\infty-l_{r-1}|<\epsilon
            // si rivela vera, allora interrompiamo l'algoritmo EM.
            if (k != 0) {
                const double precedente = logverosimiglianzeprecedenti[1];
                if (precedente-logverosimiglianzeprecedenti[0] < 0.000001){break;}
                double accelerazione = (logverosimiglianzacorrente - precedente) /
                    (precedente - logverosimiglianzeprecedenti[0]);
                if (accelerazione == 1.0) {accelerazione -= 0.000001;}
                const double linfinito = precedente + 1.0 / (1.0 - accelerazione) *
                    (logverosimiglianzacorrente - precedente);
                if (std::abs(linfinito - precedente) < sogliaconvergenza) { break; }
            }
        }
        // Se invece usiamo misture di t di Student:
    } else {
        for (int k = 0; k < iterazioniEM; k++) {
            aggiornapesi(taglia, logverosimiglianzepigreci, pesi, i);
            // Adesso calcoliamo anche i pesi u_{ig}, che ricordiamo essere
            // (\nu_p+p)/(\nu_p+distanza di Mahalanobis di x_i da \mu_g e \Sigma_g).
            std::vector<std::vector<double>> uig(taglia, std::vector<double>(i));
            for (int l = 0; l < taglia; l++) {
                const double ascissa = ascisse[l];
                const double ordinata = ordinate[l];
                for (int m = 0; m < i; m++) {
                    const double differenzaascisse = ascissa-muascisse[m];
                    const double differenzaordinate = ordinata-muordinate[m];
                    const double mahalanobis = inverse11[m]*differenzaascisse*differenzaascisse+
                        inverse12[m]*differenzaascisse*differenzaordinate+
                        inverse22[m]*differenzaordinate*differenzaordinate;
                    const double gradi = nu[m];
                    uig[l][m] = (gradi+2.0)/(gradi+mahalanobis);
                }
            }
            // La differenza dell'algoritmo CEM rispetto all'EM è che dopo aver aggiornato
            // i pesi w_{ig}, si genera una partizione dei dati assegnando ogni osservazione i
            // al cluster g per cui w_{ig} è massimo; in seguito, quando ciascun parametro
            // viene aggiornato, nelle formule vengono utilizzati solo i dati che sono stati
            // assegnati al cluster corrispondente.
            std::vector<std::vector<double>> ascissepartizione(i, std::vector<double>(taglia));
            std::vector<std::vector<double>> ordinatepartizione(i, std::vector<double>(taglia));
            std::vector<std::vector<double>> upartizione(i, std::vector<double>(taglia));
            std::vector<int> tagliepartizione(i, 0);
            for (int l = 0; l < taglia; l++) {
                double pesomaggiore = pesi[l][0];
                int clusterunita = 0;
                for (int m = 1; m < i; m++) {
                    if (pesi[l][m] > pesomaggiore) {
                        pesomaggiore = pesi[l][m];
                        clusterunita = m;
                    }
                }
                const int indice = tagliepartizione[clusterunita];
                ascissepartizione[clusterunita][indice] = ascisse[l];
                ordinatepartizione[clusterunita][indice] = ordinate[l];
                upartizione[clusterunita][indice] = uig[l][indice];
                tagliepartizione[clusterunita]++;
            }
            // Ora aggiorniamo la stima dei parametri.
            for (int l = 0; l < i; l++) {
                const int tagliapartizione = tagliepartizione[l];
                if (tagliapartizione != 0){
                    const std::vector<double> ascissecorrenti = ascissepartizione[l];
                    const std::vector<double> ordinatecorrenti = ordinatepartizione[l];
                    const std::vector<double> ucorrenti = upartizione[l];
                    double mediaascissecorrente = 0.0;
                    double mediaordinatecorrente = 0.0;
                    double denominatore = 0.0;
                    for (int m = 0; m < tagliapartizione; m++) {
                        const double u = ucorrenti[m];
                        mediaascissecorrente += u*ascissecorrenti[m];
                        mediaordinatecorrente += u*ordinatecorrenti[m];
                        denominatore += u;
                    }
                    pigreci[l] = static_cast<double>(tagliapartizione)/taglia;
                    if (pigreci[l] < piminimo){pigreci[l] = piminimo;}
                    if (denominatore >= 0.000001 && tagliapartizione != 1) {
                        mediaascissecorrente /= denominatore;
                        mediaordinatecorrente /= denominatore;
                        muascisse[l] = mediaascissecorrente;
                        muordinate[l] = mediaordinatecorrente;
                        double varianzaascisse = 0.0;
                        double varianzaordinate = 0.0;
                        double covarianza = 0.0;
                        for (int m = 0; m < tagliapartizione; m++) {
                            const double u = ucorrenti[m];
                            const double differenzaascisse = ascissecorrenti[m]-mediaascissecorrente;
                            const double differenzaordinate = ordinatecorrenti[m]-mediaordinatecorrente;
                            varianzaascisse += u*differenzaascisse*differenzaascisse;
                            varianzaordinate += u*differenzaordinate*differenzaordinate;
                            covarianza += u*differenzaascisse*differenzaordinate;
                        }
                        varianzaascisse /= tagliapartizione;
                        varianzaordinate /= tagliapartizione;
                        covarianza /= tagliapartizione;
                        const double prodottovarianze = std::sqrt(varianzaascisse*varianzaordinate);
                        if (covarianza < -prodottovarianze || covarianza > prodottovarianze) {
                            covarianza = std::copysign(prodottovarianze, covarianza);
                        }
                        double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
                        if (determinante < 0.000001) {
                            varianzaascisse += 0.001;
                            varianzaordinate += 0.001;
                            determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
                        }
                        inverse11[l] = varianzaordinate/determinante;
                        inverse22[l] = varianzaascisse/determinante;
                        inverse12[l] = -covarianza/determinante;
                        sigma11[l] = varianzaascisse;
                        sigma22[l] = varianzaordinate;
                        sigma12[l] = covarianza;
                        determinanti[l] = determinante;
                    } else {
                        muascisse[l] = inizializzatore(generatore);
                        muordinate[l] = inizializzatore(generatore);
                        sigma11[l] = 0.01;
                        sigma22[l] = 0.01;
                        sigma12[l] = 0.0;
                        inverse11[l] = 100.0;
                        inverse22[l] = 100.0;
                        inverse12[l] = 0.0;
                        determinanti[l] = 0.0001;
                    }
                    double grado = nu[l];
                    if (grado <= 0.0){grado = 0.001;}
                    const double gradoiniziale = grado;
                    double grado1 = gradoiniziale/2.0+1.0;
                    double digamma1 = 0.0;
                    while (grado1 < 6.0) {
                        digamma1 -= 1.0/grado1;
                        grado1++;
                    }
                    const double g2 = grado1*grado1;
                    const double g4 = g2*g2;
                    const double g8 = g4*g4;
                    digamma1 += std::log(grado1)-1.0/(2.0*grado1)-1.0/(12.0*g2)+1.0/(120.0*g4)-1.0/(252.0*g4*g2)+
                        1.0/(240.0*g8)-1.0/(132.0*g8*g2);
                    double sommaru = 0.0;
                    for (int m = 0; m < tagliapartizione; m++){
                        double u = ucorrenti[m];
                        if (u < 0.000001){u += 0.000001;}
                        sommaru += std::log(u)-u;
                    }
                    const double costantenumeratore = digamma1+1.0+sommaru/tagliapartizione;
                    for (int m = 0; m < 20; m++) {
                        double grado2 = grado/2.0;
                        double digamma2 = 0.0;
                        double trigamma = 0.0;
                        while (grado2 < 6.0) {
                            digamma2 -= 1.0/grado2;
                            trigamma += 1.0/(grado2*grado2);
                            grado2++;
                        }
                        const double gg2 = grado2*grado2;
                        const double gg4 = gg2*gg2;
                        const double gg8 = gg4*gg4;
                        digamma2 += std::log(grado2)-1.0/(2.0*grado2)-1.0/(12.0*gg2)+1.0/(120.0*gg4)-
                            1.0/(252.0*gg4*gg2)+1.0/(240.0*gg8)-1.0/(132.0*gg8*gg2);
                        trigamma += 1.0/grado2+1.0/(2.0*gg2)+1.0/(6.0*gg2*grado2)-1.0/(30.0*gg4*grado2)+
                            1.0/(42.0*gg4*gg2*grado2)-1.0/(30.0*gg8*grado2)+10.0/(132.0*gg8*gg2*grado2);
                        const double logaritmo = std::log(grado/(gradoiniziale+2.0));
                        const double reciproco = 1.0/grado;
                        grado -= (costantenumeratore-digamma2+logaritmo)/(reciproco-0.5*trigamma);
                        if (grado <= 0.0){grado = 0.001;}
                        if (grado > 200.0){grado = 200.0;}
                    }
                    nu[l] = grado;
                } else {
                    pigreci[l] = piminimo;
                    muascisse[l] = inizializzatore(generatore);
                    muordinate[l] = inizializzatore(generatore);
                    sigma11[l] = 0.01;
                    sigma22[l] = 0.01;
                    sigma12[l] = 0.0;
                    inverse11[l] = 100.0;
                    inverse22[l] = 100.0;
                    inverse12[l] = 0.0;
                }
            }
            // Ora calcoliamo la logverosimiglianza.
            logverosimiglianzeprecedenti = {logverosimiglianzeprecedenti[1], logverosimiglianzacorrente};
            logverosimiglianzacorrente = logverosimiglia(i, taglia, pigreci, determinanti, muascisse,
                muordinate, inverse11, inverse22, inverse12, nu, ascisse, ordinate, logverosimiglianzepigreci, normale);
            // Dalla seconda iterazione in poi controlliamo la convergenza con l'accelerazione di Aitken.
            if (k != 0) {
                const double precedente = logverosimiglianzeprecedenti[1];
                if (precedente-logverosimiglianzeprecedenti[0] < 0.000001){break;}
                double accelerazione = (logverosimiglianzacorrente - precedente) /
                    (precedente - logverosimiglianzeprecedenti[0]);
                if (accelerazione == 1.0) {accelerazione -= 0.000001;}
                const double linfinito = precedente + 1.0 / (1.0 - accelerazione) *
                    (logverosimiglianzacorrente - precedente);
                if (std::abs(linfinito - precedente) < sogliaconvergenza) { break; }
            }
        }
    }
}


std::vector<int> kmedoidi(std::mt19937 &generatore, const int taglia, const std::vector<double> &ascisse,
    const std::vector<double> &ordinate, const int i, const double minimoascisse, const double massimoascisse,
    const double minimoordinate, const double massimoordinate, const int iterazionikmeans) {
    std::uniform_int_distribution<> uniformemedoidi(0, taglia-1);
    // Iniziamo estraendo dei medoidi a caso, facendo attenzione a non estrarre
    // due volte lo stesso medoide.
    std::vector<int> medoidi(i);
    medoidi[0] = uniformemedoidi(generatore);
    for (int k = 1; k < i; k++) {
        int nuovomedoide = uniformemedoidi(generatore);
        bool uguali = true;
        while (uguali) {
            uguali = false;
            for (int l = 0; l < k; l++) {
                if (nuovomedoide == medoidi[l]) {
                    nuovomedoide = uniformemedoidi(generatore);
                    uguali = true;
                    break;
                }
            }
        }
        medoidi[k] = nuovomedoide;
    }
    // Ora associamo ogni osservazione al medoide più vicino.
    std::vector<int> clusterkmedoidi(taglia);
    double costototale = 0.0;
    for (int k = 0; k < taglia; k++) {
        int medoideattuale = medoidi[0];
        if (k == medoideattuale) {
            clusterkmedoidi[k] = 0;
            continue;
        }
        int medoideosservazione = 0;
        const double ascissaosservazione = ascisse[k];
        const double ordinataosservazione = ordinate[k];
        double ascissamedoide = ascisse[medoideattuale];
        double ordinatamedoide = ordinate[medoideattuale];
        double differenzaascisse = ascissamedoide - ascissaosservazione;
        double differenzaordinate = ordinatamedoide - ordinataosservazione;
        double distanza = std::sqrt(differenzaascisse * differenzaascisse + differenzaordinate * differenzaordinate);
        for (int l = 1; l < i; l++) {
            medoideattuale = medoidi[l];
            ascissamedoide = ascisse[medoideattuale];
            ordinatamedoide = ordinate[medoideattuale];
            differenzaascisse = ascissamedoide - ascissaosservazione;
            differenzaordinate = ordinatamedoide - ordinataosservazione;
            const double nuovadistanza = std::sqrt(differenzaascisse * differenzaascisse +
                                                   differenzaordinate * differenzaordinate);
            if (nuovadistanza < distanza) {
                distanza = nuovadistanza;
                medoideosservazione = l;
            }
        }
        clusterkmedoidi[k] = medoideosservazione;
        costototale += distanza;
    }
    // Adesso siamo pronti a implementare i k-medoidi, usando l'algoritmo PAM.
    for (int k = 0; k < iterazionikmeans; k++) {
        std::vector<int> medoidiprecedenti = medoidi;
        for (int l = 0; l < i; l++) {
            for (int m = 0; m < taglia; m++) {
                bool traimedoidi = false;
                for (int n = 0; n < i; n++) {
                    if (m == medoidi[n]) {
                        traimedoidi = true;
                        break;
                    }
                }
                if (traimedoidi) { continue; }
                double nuovocosto = 0.0;
                std::vector<int> nuovimedoidi = medoidi;
                nuovimedoidi[l] = m;
                std::vector<int> nuovicluster(taglia);
                for (int n = 0; n < taglia; n++) {
                    const int primomedoide = nuovimedoidi[0];
                    const double ascissa = ascisse[n];
                    const double ordinata = ordinate[n];
                    const double distanzaprimeascisse = ascissa - ascisse[primomedoide];
                    const double distanzaprimeordinate = ordinata - ordinate[primomedoide];
                    double distanzaminima = std::sqrt(distanzaprimeascisse * distanzaprimeascisse +
                                                      distanzaprimeordinate * distanzaprimeordinate);
                    int medoidecandidato = 0;
                    for (int o = 1; o < i; o++) {
                        const int altromedoide = nuovimedoidi[o];
                        const double distanzaaltreascisse = ascissa - ascisse[altromedoide];
                        const double distanzaaltreordinate = ordinata - ordinate[altromedoide];
                        const double altradistanza = std::sqrt(distanzaaltreascisse * distanzaaltreascisse +
                                                               distanzaaltreordinate * distanzaaltreordinate);
                        if (altradistanza < distanzaminima) {
                            distanzaminima = altradistanza;
                            medoidecandidato = o;
                        }
                    }
                    nuovicluster[n] = medoidecandidato;
                    nuovocosto += distanzaminima;
                }
                if (nuovocosto < costototale) {
                    costototale = nuovocosto;
                    clusterkmedoidi = nuovicluster;
                    medoidi = nuovimedoidi;
                }
            }
        }
        bool convergenza = true;
        for (int l = 0; l < i; l++) {
            if (medoidi[l] != medoidiprecedenti[l]) {
                convergenza = false;
                break;
            }
        }
        if (convergenza) { break; }
    }
    return clusterkmedoidi;
}


// Funzione che contiene il cuore dell'algoritmo di clustering basato sul modello.
// Di seguito sono dati i significati dei parametri. Notare che devono essere tutti specificati
// anche se non vengono usati.
// Normale: "true" se si vogliono usare misture di gaussiane, "false" se si vogliono usare
// misture di t di Student.
// I: numero di cluster.
// Iterazioniinizializzazione: numero di volte in cui è ripetuto
// l'algoritmo di inizializzazione (k-medie o k-medoidi) prima di fissare i parametri iniziali.
// Taglia: numero di osservazioni nel campione.
// Inizializzazione: una stringa tra "kmeans", "kmedoidi" e "casuale", che indica l'algoritmo
// di inizializzazione dei parametri.
// Ascisse e ordinate: il campione, passato con le variabili in due vettori separati.
// Iterazionikmeans: numero massimo di volte in cui vengono aggiornati i centroidi nell'algoritmo
// k-means o k-medoidi. Tale numero può non essere raggiunto se si arriva prima a convergenza.
// Algoritmo: algoritmo per la stima dei parametri della mistura, una stringa tra "EM", "SEM" o "CEM".
// IterazioniEM: numero massimo di iterazioni dell'algoritmo EM, SEM o CEM. Tale numero può non essere
// raggiunto se viene soddisfatta prima una regola di arresto.
// Logverosimiglianza: riferimento alla variabile nella funzione madre che immagazzina la logverosimiglianza.
// Cluster: riferimento a un vettore che contiene l'indice del cluster a cui appartiene ogni unità.
// Sogliaconvergenza: numero \epsilon che determina la potenza della regola di arresto dell'algoritmo EM.
// Pigrecofinali: riferimento a un vettore di parametri \pi_g.
// Muascissefinali: riferimento a un vettore di parametri \mu_{1g}.
// Muordinatefinali: riferimento a un vettore di parametri \mu_{2g}.
// Sigma11finali: riferimento a un vettore di parametri \sigma^2_1.
// Sigma22finali: riferimento a un vettore di parametri \sigma^2_2.
// Sigma12finali: riferimento a un vettore di parametri \sigma_{12}.
// Nufinali: riferimento a un vettore di parametri \nu_g.
// Pesifinali: riferimento a un vettore di vettori di pesi w_{ig}.
// La logverosimiglianza, i cluster, i parametri finali e i pesi finali sono l'output dell'algoritmo,
// e vengono calcolati dalla funzione, che poi li aggiorna per riferimento.
void veroclustering(const bool normale, const int i, const int iterazioniinizializzazione,
    const int taglia, const char inizializzazione, const std::vector<double> &ascisse,
    const std::vector<double> &ordinate,
    const int iterazionikmeans, const char algoritmo, const int iterazioniEM,
    double &logverosimiglianza, std::vector<int> &clusterfinali, const double sogliaconvergenza,
    std::vector<double> &pigrecifinali, std::vector<double> &muascissefinali, std::vector<double> &muordinatefinali,
    std::vector<double> &sigma11finali, std::vector<double> &sigma22finali, std::vector<double> &sigma12finali,
    std::vector<double> &nufinali, std::vector<std::vector<double>> &pesifinali) {
    if (ascisse.size() != taglia){throw std::runtime_error("ASCISSE");}
    if (ordinate.size() != taglia){throw std::runtime_error("ORDINATE");}
    if (clusterfinali.size() != taglia){throw std::runtime_error("CLUSTERFINALI");}
    if (pigrecifinali.size() != i) {throw std::runtime_error("PIGRECIFINALI");}
    if (muascissefinali.size() != i){throw std::runtime_error("MUASCISSEFINALI");}
    if (muordinatefinali.size() != i){throw std::runtime_error("MUORDINATEFINALI");}
    if (sigma11finali.size() != i){throw std::runtime_error("SIGMA11FINALI");}
    if (sigma22finali.size() != i){throw std::runtime_error("SIGMA22FINALI");}
    if (sigma12finali.size() != i){throw std::runtime_error("SIGMA12FINALI");}
    if (nufinali.size() != i){throw std::runtime_error("NUFINALI");}
    if (pesifinali.size() != taglia){throw std::runtime_error("PESIFINALI");}
    if (pesifinali[0].size() != i){throw std::runtime_error("PESIFINALI");}
    const double piminimo = 2.0/taglia;
    // Prima di inizializzare i parametri, vediamo in che sottoinsieme del piano cartesiano
    // è limitato il campione: troviamo i minimi e i massimi valori registrati delle due coordinate.
    double minimoascisse = ascisse[0];
    double minimoordinate = ordinate[0];
    double massimoascisse = ascisse[0];
    double massimoordinate = ordinate[0];
    for (int j = 1; j < taglia; j++) {
        const double ascissa = ascisse[j];
        const double ordinata = ordinate[j];
        if (ascissa < minimoascisse) {
            minimoascisse = ascissa;
        }
        if (ascissa > massimoascisse) {
            massimoascisse = ascissa;
        }
        if (ordinata < minimoordinate) {
            minimoordinate = ordinata;
        }
        if (ordinata > massimoordinate) {
            massimoordinate = ordinata;
        }
    }
    // Prepariamo i generatori di numeri casuali.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::uniform_real_distribution<> inizializzatore(-1.0, 1.0);
    // Se abbiamo scelto di inizializzare con le k-medie:
    if (inizializzazione == 'K') {
        // Inizializziamo un vettore che conterrà la classificazione delle unità secondo il k-means.
        std::vector<int> clustercorrenti(taglia);
        std::vector<int> taglieclustercorrenti(i, 0);
        double silhouettemediamassima = -1.1;
        // Per ogni iterazione dell'algoritmo di inizializzazione:
        for (int j = 0; j < iterazioniinizializzazione; j++) {
            std::vector<int> clusterkmeans = kmeans(generatore, taglia, ascisse, ordinate, i, minimoascisse,
                massimoascisse, minimoordinate, massimoordinate, iterazionikmeans);
            // Ora calcoliamo la silhouette media della partizione così ottenuta. Se è minore del record
            // attuale, la salviamo come candidata all'inizializzazione dell'algoritmo EM.
            std::vector<int> tagliecluster(i, 0);
            for (int k = 0; k < taglia; k++) {
                tagliecluster[clusterkmeans[k]]++;
            }
            double silhouettemedia = silhouette(taglia, clusterkmeans, tagliecluster, ascisse, ordinate, i);
            if (silhouettemedia > silhouettemediamassima) {
                silhouettemediamassima = silhouettemedia;
                clustercorrenti = clusterkmeans;
                taglieclustercorrenti = tagliecluster;
            }
        }
        // Ora possiamo ricavare dalla partizione ottenuta i valori iniziali dei parametri.
        std::vector<double> pigreci(i);
        std::vector<double> muascisse(i, 0.0);
        std::vector<double> muordinate(i, 0.0);
        std::vector<double> sigma11(i, 0.0);
        std::vector<double> sigma22(i, 0.0);
        std::vector<double> sigma12(i, 0.0);
        // Inizializziamo in ogni caso i gradi di libertà; nel caso abbiamo scelto di usare misture
        // di gaussiane, semplicemente non saranno usati.
        std::vector<double> nu(i, 5.0);
        inizializza(taglia, piminimo, generatore, clustercorrenti, taglieclustercorrenti, ascisse, ordinate,
            muascisse, muordinate, pigreci, sigma11, sigma22, sigma12, i);
        // Per evitare problemi di matrici singolari, applichiamo la regolarizzazione di Tikhonov.
        // Contemporaneamente, per accelerarci successivamente, salviamo i determinanti delle
        // matrici di covarianza dei cluster e le loro inverse.
        std::vector<double> determinanti(i);
        std::vector<double> inverse11(i);
        std::vector<double> inverse22(i);
        std::vector<double> inverse12(i);
        inverti(sigma11, sigma22, sigma12, determinanti, inverse11, inverse22, inverse12, i);
        // Calcoliamo il valore della logverosimiglianza (delle gaussiane o delle t di Student)
        // con la partizione attuale. Inoltre, salviamo i valori individuali delle logverosimiglianze
        // per gli specifici i e g in modo da riusarli dopo.
        std::vector<std::vector<double>> logverosimiglianzepigreci(taglia, std::vector<double>(i, 0.0));
        double logverosimiglianzacorrente = logverosimiglia(i, taglia, pigreci, determinanti, muascisse, muordinate,
            inverse11, inverse22, inverse12, nu, ascisse, ordinate, logverosimiglianzepigreci, normale);
        // Inizializziamo il vettore di pesi w_{ig} e poi avviamo l'algoritmo iterativo di stima dei parametri.
        std::vector<std::vector<double>> pesi(taglia, std::vector<double>(i, 0.0));
        if (algoritmo == 'E') {
            EM(normale, iterazioniEM, taglia, i, logverosimiglianzepigreci, pigreci, muascisse,
                muordinate, sigma11, sigma22, sigma12, determinanti, inverse11, inverse22,
                inverse12, nu, pesi, ascisse, ordinate, piminimo, generatore, logverosimiglianzacorrente,
                sogliaconvergenza);
            // Adesso generiamo la partizione delle unità in base ai parametri ottenuti;
            // ricordiamo che ogni unità viene assegnata al cluster g per cui w_{ig} è massimo.
            for (int k = 0; k < taglia; k++) {
                const std::vector<double> rigapesi = pesi[k];
                double pesomaggiore = rigapesi[0];
                int cluster = 0;
                for (int l = 1; l < i; l++) {
                    if (rigapesi[l] > pesomaggiore){
                        pesomaggiore = rigapesi[l];
                        cluster = l;
                    }
                }
                clusterfinali[k] = cluster;
            }
            // Inoltre, memorizziamo al di fuori della funzione le stime dei parametri ottenute.
            pigrecifinali = pigreci;
            muascissefinali = muascisse;
            muordinatefinali = muordinate;
            sigma11finali = sigma11;
            sigma22finali = sigma22;
            sigma12finali = sigma12;
            nufinali = nu;
            pesifinali = pesi;
            logverosimiglianza = logverosimiglianzacorrente;
        }
        if (algoritmo == 'S') {
            // Notare che date le caratteristiche dell'algoritmo SEM, non ha senso controllare la
            // convergenza nel modo classico dell'algoritmo EM. Ciò che faremo invece è: dopo
            // iterazioniEM/4 iterazioni, controlleremo di volta in volta la logverosimiglianza,
            // e se sarà superiore al record precedentemente registrato, porremo le stime ottenute
            // in quell'iterazione come candidate a essere quelle finali a meno che non vengano
            // battute da un'altra logverosimiglianza, fino all'esaurimento delle iterazioni.
            // Come valore iniziale del "record" della logverosimiglianza useremo quella ottenuta
            // dopo l'algoritmo di inizializzazione.
            const int sogliacontrollo = iterazioniEM/4;
            double logverosimiglianzavincente = logverosimiglianzacorrente;
            std::vector<double> pigrecivincenti = pigreci;
            std::vector<double> muascissevincenti = muascisse;
            std::vector<double> muordinatevincenti = muordinate;
            std::vector<double> sigma11vincenti = sigma11;
            std::vector<double> sigma22vincenti = sigma22;
            std::vector<double> sigma12vincenti = sigma12;
            std::vector<double> nuvincenti = nu;
            std::vector<std::vector<double>> pesivincenti(taglia, pigreci);
            SEM(normale, iterazioniEM, taglia, i, logverosimiglianzepigreci, pesi, pigreci, muascisse,
                muordinate, sigma11, sigma22, sigma12, nu, ascisse, ordinate, generatore, determinanti,
                inverse11, inverse22, inverse12, piminimo, logverosimiglianzacorrente, logverosimiglianzavincente,
                pigrecivincenti, muascissevincenti, muordinatevincenti, sigma11vincenti, sigma22vincenti,
                sigma12vincenti, nuvincenti, pesivincenti, sogliacontrollo);
            // Adesso generiamo la partizione delle unità in base ai parametri ottenuti;
            // ricordiamo che ogni unità viene assegnata al cluster g per cui w_{ig} è massimo.
            for (int k = 0; k < taglia; k++) {
                const std::vector<double> rigapesi = pesivincenti[k];
                double pesomaggiore = rigapesi[0];
                int cluster = 0;
                for (int l = 1; l < i; l++) {
                    if (rigapesi[l] > pesomaggiore){
                        pesomaggiore = rigapesi[l];
                        cluster = l;
                    }
                }
                clusterfinali[k] = cluster;
            }
            // Inoltre, memorizziamo al di fuori della funzione le stime dei parametri ottenute.
            pigrecifinali = pigrecivincenti;
            muascissefinali = muascissevincenti;
            muordinatefinali = muordinatevincenti;
            sigma11finali = sigma11vincenti;
            sigma22finali = sigma22vincenti;
            sigma12finali = sigma12vincenti;
            nufinali = nuvincenti;
            pesifinali = pesivincenti;
            logverosimiglianza = logverosimiglianzavincente;
        }
        if (algoritmo == 'C') {
            CEM(normale, iterazioniEM, taglia, i, logverosimiglianzepigreci, pigreci, muascisse,
                muordinate, sigma11, sigma22, sigma12, determinanti, inverse11, inverse22,
                inverse12, nu, pesi, ascisse, ordinate, piminimo, generatore, logverosimiglianzacorrente,
                sogliaconvergenza);
            // Adesso generiamo la partizione delle unità in base ai parametri ottenuti;
            // ricordiamo che ogni unità viene assegnata al cluster g per cui w_{ig} è massimo.
            for (int k = 0; k < taglia; k++) {
                const std::vector<double> rigapesi = pesi[k];
                double pesomaggiore = rigapesi[0];
                int cluster = 0;
                for (int l = 1; l < i; l++) {
                    if (rigapesi[l] > pesomaggiore){
                        pesomaggiore = rigapesi[l];
                        cluster = l;
                    }
                }
                clusterfinali[k] = cluster;
            }
            // Inoltre, memorizziamo al di fuori della funzione le stime dei parametri ottenute.
            pigrecifinali = pigreci;
            muascissefinali = muascisse;
            muordinatefinali = muordinate;
            sigma11finali = sigma11;
            sigma22finali = sigma22;
            sigma12finali = sigma12;
            nufinali = nu;
            pesifinali = pesi;
            logverosimiglianza = logverosimiglianzacorrente;
        }
    }
    if (inizializzazione == 'D') {
        // Inizializziamo un vettore che conterrà la classificazione delle unità secondi i k-medoidi.
        std::vector<int> clustercorrenti(taglia);
        std::vector<int> tagliecluster(i, 0);
        double silhouettemassima = -1.1;
        // Per ogni iterazione dell'algoritmo di inizializzazione:
        for (int j = 0; j < iterazioniinizializzazione; j++) {
            std::vector<int> clusterkmedoidi = kmedoidi(generatore, taglia, ascisse, ordinate, i, minimoascisse,
                massimoascisse, minimoordinate, massimoordinate, iterazionikmeans);
            std::vector<int> taglieclustercorrenti(taglia);
            for (int k = 0; k < taglia; k++){
                taglieclustercorrenti[clusterkmedoidi[k]]++;
            }
            double silhouettemedia = silhouette(taglia, clusterkmedoidi, tagliecluster, ascisse, ordinate, i);
            if (silhouettemedia > silhouettemassima) {
                silhouettemassima = silhouettemedia;
                clustercorrenti = clusterkmedoidi;
                tagliecluster = taglieclustercorrenti;
            }
        }
        // Ora possiamo ricavare dalla partizione ottenuta i valori iniziali dei parametri.
        std::vector<double> pigreci(i);
        std::vector<double> muascisse(i, 0.0);
        std::vector<double> muordinate(i, 0.0);
        std::vector<double> sigma11(i, 0.0);
        std::vector<double> sigma22(i, 0.0);
        std::vector<double> sigma12(i, 0.0);
        // Inizializziamo in ogni caso i gradi di libertà; nel caso abbiamo scelto di usare misture
        // di gaussiane, semplicemente non saranno usati.
        std::vector<double> nu(i, 5.0);
        inizializza(taglia, piminimo, generatore, clustercorrenti, taglieclustercorrenti, ascisse, ordinate,
            muascisse, muordinate, pigreci, sigma11, sigma22, sigma12, i);
        // Per evitare problemi di matrici singolari, applichiamo la regolarizzazione di Tikhonov.
        // Contemporaneamente, per accelerarci successivamente, salviamo i determinanti delle
        // matrici di covarianza dei cluster e le loro inverse.
        std::vector<double> determinanti(i);
        std::vector<double> inverse11(i);
        std::vector<double> inverse22(i);
        std::vector<double> inverse12(i);
        inverti(sigma11, sigma22, sigma12, determinanti, inverse11, inverse22, inverse12, i);
        // Calcoliamo il valore della logverosimiglianza (delle gaussiane o delle t di Student)
        // con la partizione attuale. Inoltre, salviamo i valori individuali delle verosimiglianze
        // per gli specifici i e g in modo da riusarli dopo.
        std::vector<std::vector<double>> logverosimiglianzepigreci(taglia, std::vector<double>(i, 0.0));
        double logverosimiglianzacorrente = logverosimiglia(i, taglia, pigreci, determinanti, muascisse, muordinate,
            inverse11, inverse22, inverse12, nu, ascisse, ordinate, logverosimiglianzepigreci, normale);
        // Inizializziamo il vettore di pesi w_{ig} e poi avviamo l'algoritmo iterativo di stima dei parametri.
        std::vector<std::vector<double>> pesi(taglia, std::vector<double>(i, 0.0));
        if (algoritmo == 'E') {
            EM(normale, iterazioniEM, taglia, i, logverosimiglianzepigreci, pigreci, muascisse,
                muordinate, sigma11, sigma22, sigma12, determinanti, inverse11, inverse22,
                inverse12, nu, pesi, ascisse, ordinate, piminimo, generatore, logverosimiglianzacorrente,
                sogliaconvergenza);
            // Adesso generiamo la partizione delle unità in base ai parametri ottenuti;
            // ricordiamo che ogni unità viene assegnata al cluster g per cui w_{ig} è massimo.
            for (int k = 0; k < taglia; k++) {
                const std::vector<double> rigapesi = pesi[k];
                double pesomaggiore = rigapesi[0];
                int cluster = 0;
                for (int l = 1; l < i; l++) {
                    if (rigapesi[l] > pesomaggiore){
                        pesomaggiore = rigapesi[l];
                        cluster = l;
                    }
                }
                clusterfinali[k] = cluster;
            }
            // Inoltre, memorizziamo al di fuori della funzione le stime dei parametri ottenute.
            pigrecifinali = pigreci;
            muascissefinali = muascisse;
            muordinatefinali = muordinate;
            sigma11finali = sigma11;
            sigma22finali = sigma22;
            sigma12finali = sigma12;
            nufinali = nu;
            pesifinali = pesi;
            logverosimiglianza = logverosimiglianzacorrente;
        }
        if (algoritmo == 'S') {
            // Notare che date le caratteristiche dell'algoritmo SEM, non ha senso controllare la
            // convergenza nel modo classico dell'algoritmo EM. Ciò che faremo invece è: dopo
            // iterazioniEM/4 iterazioni, controlleremo di volta in volta la logverosimiglianza,
            // e se sarà superiore al record precedentemente registrato, porremo le stime ottenute
            // in quell'iterazione come candidate a essere quelle finali a meno che non vengano
            // battute da un'altra logverosimiglianza, fino all'esaurimento delle iterazioni.
            // Come valore iniziale del "record" della logverosimiglianza useremo quella ottenuta
            // dopo l'algoritmo di inizializzazione.
            const int sogliacontrollo = iterazioniEM/4;
            double logverosimiglianzavincente = logverosimiglianzacorrente;
            std::vector<double> pigrecivincenti = pigreci;
            std::vector<double> muascissevincenti = muascisse;
            std::vector<double> muordinatevincenti = muordinate;
            std::vector<double> sigma11vincenti = sigma11;
            std::vector<double> sigma22vincenti = sigma22;
            std::vector<double> sigma12vincenti = sigma12;
            std::vector<double> nuvincenti = nu;
            std::vector<std::vector<double>> pesivincenti(taglia, pigreci);
            SEM(normale, iterazioniEM, taglia, i, logverosimiglianzepigreci, pesi, pigreci, muascisse,
                muordinate, sigma11, sigma22, sigma12, nu, ascisse, ordinate, generatore, determinanti,
                inverse11, inverse22, inverse12, piminimo, logverosimiglianzacorrente, logverosimiglianzavincente,
                pigrecivincenti, muascissevincenti, muordinatevincenti, sigma11vincenti, sigma22vincenti,
                sigma12vincenti, nuvincenti, pesivincenti, sogliacontrollo);
            // Adesso generiamo la partizione delle unità in base ai parametri ottenuti;
            // ricordiamo che ogni unità viene assegnata al cluster g per cui w_{ig} è massimo.
            for (int k = 0; k < taglia; k++) {
                const std::vector<double> rigapesi = pesivincenti[k];
                double pesomaggiore = rigapesi[0];
                int cluster = 0;
                for (int l = 1; l < i; l++) {
                    if (rigapesi[l] > pesomaggiore){
                        pesomaggiore = rigapesi[l];
                        cluster = l;
                    }
                }
                clusterfinali[k] = cluster;
            }
            // Inoltre, memorizziamo al di fuori della funzione le stime dei parametri ottenute.
            pigrecifinali = pigrecivincenti;
            muascissefinali = muascissevincenti;
            muordinatefinali = muordinatevincenti;
            sigma11finali = sigma11vincenti;
            sigma22finali = sigma22vincenti;
            sigma12finali = sigma12vincenti;
            nufinali = nuvincenti;
            pesifinali = pesivincenti;
            logverosimiglianza = logverosimiglianzavincente;
        }
        if (algoritmo == 'C') {
            CEM(normale, iterazioniEM, taglia, i, logverosimiglianzepigreci, pigreci, muascisse,
                muordinate, sigma11, sigma22, sigma12, determinanti, inverse11, inverse22,
                inverse12, nu, pesi, ascisse, ordinate, piminimo, generatore, logverosimiglianzacorrente,
                sogliaconvergenza);
            // Adesso generiamo la partizione delle unità in base ai parametri ottenuti;
            // ricordiamo che ogni unità viene assegnata al cluster g per cui w_{ig} è massimo.
            for (int k = 0; k < taglia; k++) {
                const std::vector<double> rigapesi = pesi[k];
                double pesomaggiore = rigapesi[0];
                int cluster = 0;
                for (int l = 1; l < i; l++) {
                    if (rigapesi[l] > pesomaggiore){
                        pesomaggiore = rigapesi[l];
                        cluster = l;
                    }
                }
                clusterfinali[k] = cluster;
            }
            // Inoltre, memorizziamo al di fuori della funzione le stime dei parametri ottenute.
            pigrecifinali = pigreci;
            muascissefinali = muascisse;
            muordinatefinali = muordinate;
            sigma11finali = sigma11;
            sigma22finali = sigma22;
            sigma12finali = sigma12;
            nufinali = nu;
            pesifinali = pesi;
            logverosimiglianza = logverosimiglianzacorrente;
        }
    }
    if (inizializzazione == 'C') {
        std::gamma_distribution<> gammavarianze(2.0, 1.0);
        std::uniform_real_distribution<> uniformecorrelazione(0.0, 1.0);
        std::uniform_real_distribution<> uniformeascisse(minimoascisse, massimoascisse);
        std::uniform_real_distribution<> uniformeordinate(minimoordinate, massimoordinate);
        double logverosimiglianzamassima = -std::numeric_limits<double>::infinity();
        // Se scegliamo di inizializzare i parametri casualmente, dopo ogni estrazione applicheremo
        // l'algoritmo EM e alla fine sceglieremo la soluzione con la verosimiglianza più alta.
        for (int j = 0; j < iterazioniinizializzazione; j++) {
            std::vector<double> pigreci(i);
            std::vector<double> muascisse(i);
            std::vector<double> muordinate(i);
            std::vector<double> sigma11(i);
            std::vector<double> sigma22(i);
            std::vector<double> sigma12(i);
            std::vector<double> nu(i, 5.0);
            for (int k = 0; k < i; k++) {
                // \pi_g è l'unico parametro inizializzato deterministicamente, e in particolare
                // sono impostati, per ogni g, a 1/numerocluster.
                pigreci[k] = 1.0/i;
                // \mu_g invece è estratto da un'uniforme con supporto pari al campo di variazione
                // dei dati.
                muascisse[k] = uniformeascisse(generatore);
                muordinate[k] = uniformeordinate(generatore);
                // La diagonale di \Sigma_g è estratta da una Gamma(2, 1); la covarianza è ottenuta
                // estraendo da un'uniforme(0, 1) la correlazione, che poi viene moltiplicata per
                // le deviazioni standard.
                const double varianzaascisse = gammavarianze(generatore);
                const double varianzaordinate = gammavarianze(generatore);
                sigma11[k] = varianzaascisse;
                sigma22[k] = varianzaordinate;
                sigma12[k] = uniformecorrelazione(generatore)*std::sqrt(varianzaascisse*varianzaordinate);
            }
            // Per evitare problemi di matrici singolari, applichiamo la regolarizzazione di Tikhonov.
            // Contemporaneamente, per accelerarci successivamente, salviamo i determinanti delle
            // matrici di covarianza dei cluster e le loro inverse.
            std::vector<double> determinanti(i);
            std::vector<double> inverse11(i);
            std::vector<double> inverse22(i);
            std::vector<double> inverse12(i);
            inverti(sigma11, sigma22, sigma12, determinanti, inverse11, inverse22, inverse12, i);
            // Calcoliamo il valore della logverosimiglianza (delle gaussiane o delle t di Student)
            // con la partizione attuale. Inoltre, salviamo i valori individuali delle verosimiglianze
            // per gli specifici i e g in modo da riusarli dopo.
            std::vector<std::vector<double>> logverosimiglianzepigreci(taglia, std::vector<double>(i, 0.0));
            double logverosimiglianzacorrente = logverosimiglia(i, taglia, pigreci, determinanti, muascisse, muordinate,
            inverse11, inverse22, inverse12, nu, ascisse, ordinate, logverosimiglianzepigreci, normale);
            // Inizializziamo il vettore di pesi w_{ig} e poi avviamo l'algoritmo iterativo di stima dei parametri.
            std::vector<std::vector<double>> pesi(taglia, std::vector<double>(i, 0.0));
            if (algoritmo == 'E') {
                EM(normale, iterazioniEM, taglia, i, logverosimiglianzepigreci, pigreci, muascisse,
                    muordinate, sigma11, sigma22, sigma12, determinanti, inverse11, inverse22,
                    inverse12, nu, pesi, ascisse, ordinate, piminimo, generatore, logverosimiglianzacorrente,
                    sogliaconvergenza);
                // I parametri ottenuti con questo giro dell'algoritmo EM vengono salvati solo
                // se il record precedente della logverosimiglianza viene battuto.
                if (logverosimiglianzacorrente > logverosimiglianzamassima) {
                    logverosimiglianzamassima = logverosimiglianzacorrente;
                    // Adesso generiamo la partizione delle unità in base ai parametri ottenuti;
                    // ricordiamo che ogni unità viene assegnata al cluster g per cui w_{ig} è massimo.
                    for (int k = 0; k < taglia; k++) {
                        const std::vector<double> rigapesi = pesi[k];
                        double pesomaggiore = rigapesi[0];
                        int cluster = 0;
                        for (int l = 1; l < i; l++) {
                            if (rigapesi[l] > pesomaggiore) {
                                pesomaggiore = rigapesi[l];
                                cluster = l;
                            }
                        }
                        clusterfinali[k] = cluster;
                    }
                    // Inoltre, memorizziamo al di fuori della funzione le stime dei parametri ottenute.
                    pigrecifinali = pigreci;
                    muascissefinali = muascisse;
                    muordinatefinali = muordinate;
                    sigma11finali = sigma11;
                    sigma22finali = sigma22;
                    sigma12finali = sigma12;
                    nufinali = nu;
                    pesifinali = pesi;
                    logverosimiglianza = logverosimiglianzacorrente;
                }
            }
            if (algoritmo == 'S') {
                // Notare che date le caratteristiche dell'algoritmo SEM, non ha senso controllare la
                // convergenza nel modo classico dell'algoritmo EM. Ciò che faremo invece è: dopo
                // iterazioniEM/4 iterazioni, controlleremo di volta in volta la logverosimiglianza,
                // e se sarà superiore al record precedentemente registrato, porremo le stime ottenute
                // in quell'iterazione come candidate a essere quelle finali a meno che non vengano
                // battute da un'altra logverosimiglianza, fino all'esaurimento delle iterazioni.
                // Come valore iniziale del "record" della logverosimiglianza useremo quella ottenuta
                // dopo l'algoritmo di inizializzazione.
                const int sogliacontrollo = iterazioniEM / 4;
                double logverosimiglianzavincente = logverosimiglianzacorrente;
                std::vector<double> pigrecivincenti = pigreci;
                std::vector<double> muascissevincenti = muascisse;
                std::vector<double> muordinatevincenti = muordinate;
                std::vector<double> sigma11vincenti = sigma11;
                std::vector<double> sigma22vincenti = sigma22;
                std::vector<double> sigma12vincenti = sigma12;
                std::vector<double> nuvincenti = nu;
                std::vector<std::vector<double> > pesivincenti(taglia, pigreci);
                SEM(normale, iterazioniEM, taglia, i, logverosimiglianzepigreci, pesi, pigreci, muascisse,
                    muordinate, sigma11, sigma22, sigma12, nu, ascisse, ordinate, generatore, determinanti,
                    inverse11, inverse22, inverse12, piminimo, logverosimiglianzacorrente, logverosimiglianzavincente,
                    pigrecivincenti, muascissevincenti, muordinatevincenti, sigma11vincenti, sigma22vincenti,
                    sigma12vincenti, nuvincenti, pesivincenti, sogliacontrollo);
                // I parametri ottenuti con questo giro dell'algoritmo SEM vengono salvati solo
                // se il record precedente della logverosimiglianza viene battuto.
                if (logverosimiglianzacorrente > logverosimiglianzamassima) {
                    logverosimiglianzamassima = logverosimiglianzacorrente;
                    // Adesso generiamo la partizione delle unità in base ai parametri ottenuti;
                    // ricordiamo che ogni unità viene assegnata al cluster g per cui w_{ig} è massimo.
                    for (int k = 0; k < taglia; k++) {
                        const std::vector<double> rigapesi = pesivincenti[k];
                        double pesomaggiore = rigapesi[0];
                        int cluster = 0;
                        for (int l = 1; l < i; l++) {
                            if (rigapesi[l] > pesomaggiore) {
                                pesomaggiore = rigapesi[l];
                                cluster = l;
                            }
                        }
                        clusterfinali[k] = cluster;
                    }
                    // Inoltre, memorizziamo al di fuori della funzione le stime dei parametri ottenute.
                    pigrecifinali = pigrecivincenti;
                    muascissefinali = muascissevincenti;
                    muordinatefinali = muordinatevincenti;
                    sigma11finali = sigma11vincenti;
                    sigma22finali = sigma22vincenti;
                    sigma12finali = sigma12vincenti;
                    nufinali = nuvincenti;
                    pesifinali = pesivincenti;
                    logverosimiglianza = logverosimiglianzacorrente;
                }
            }
            if (algoritmo == 'C') {
                CEM(normale, iterazioniEM, taglia, i, logverosimiglianzepigreci, pigreci, muascisse,
                    muordinate, sigma11, sigma22, sigma12, determinanti, inverse11, inverse22,
                    inverse12, nu, pesi, ascisse, ordinate, piminimo, generatore, logverosimiglianzacorrente,
                    sogliaconvergenza);
                // I parametri ottenuti con questo giro dell'algoritmo CEM vengono salvati solo
                // se il record precedente della logverosimiglianza viene battuto.
                if (logverosimiglianzacorrente > logverosimiglianzamassima) {
                    logverosimiglianzamassima = logverosimiglianzacorrente;
                    // Adesso generiamo la partizione delle unità in base ai parametri ottenuti;
                    // ricordiamo che ogni unità viene assegnata al cluster g per cui w_{ig} è massimo.
                    for (int k = 0; k < taglia; k++) {
                        const std::vector<double> rigapesi = pesi[k];
                        double pesomaggiore = rigapesi[0];
                        int cluster = 0;
                        for (int l = 1; l < i; l++) {
                            if (rigapesi[l] > pesomaggiore) {
                                pesomaggiore = rigapesi[l];
                                cluster = l;
                            }
                        }
                        clusterfinali[k] = cluster;
                    }
                    // Inoltre, memorizziamo al di fuori della funzione le stime dei parametri ottenute.
                    pigrecifinali = pigreci;
                    muascissefinali = muascisse;
                    muordinatefinali = muordinate;
                    sigma11finali = sigma11;
                    sigma22finali = sigma22;
                    sigma12finali = sigma12;
                    nufinali = nu;
                    pesifinali = pesi;
                    logverosimiglianza = logverosimiglianzacorrente;
                }
            }
        }
    }
}


// Funzione che prepara l'applicazione del clustering, usando i criteri informativi per scegliere
// il numero di cluster. Di seguito la lista di argomenti da passare:
// Minimo: numero minimo di cluster da considerare.
// Massimo: numero massimo di cluster da considerare.
// Taglia: numero di osservazioni nel dataset.
// Normale: true se si vogliono usare misture di gaussiane, false se si vogliono usare misture di t di Student.
// Iterazioniinizializzazione: numero di volte in cui va ripetuto l'intero algoritmo di inizializzazione
// dei parametri (k-means, k-medoidi o casuale).
// Inizializzazione: una stringa tra "kmeans", "kmedoidi" e "casuale", che indica come vengono
// inizializzati i parametri.
// Ascisse: la lista di ascisse di ogni osservazione.
// Ordinate: la lista di ordinate di ogni osservazione.
// Iterazionikmeans: numero di iterazioni che un singolo tentativo dell'algoritmo k-means o k-medoidi
// può fare prima di doversi fermare anche se non ha raggiunto la convergenza.
// Algoritmo: una stringa tra "EM", "SEM" e "CEM", che indica con quale algoritmo vengono cercati
// i parametri.
// IterazioniEM: numero di iterazioni dell'algoritmo EM, SEM o CEM dopo cui ci si deve fermare
// anche se non si raggiunge la convergenza.
// Sogliaconvergenza: numero \epsilon che determina la regola di arresto per l'algoritmo EM.
// Criterioscelto: una stringa tra "AIC", "BIC" e "ICL", che indica il criterio informativo usato.
// Criterio: un riferimento al valore numerico del criterio nella funzione madre.
// Clusterfinali: un riferimento al vettore che contiene gli indici dei cluster per ogni osservazione.
void clusteringconcriterio(const int minimo, const int massimo, const int taglia, const bool normale,
    const int iterazioniinizializzazione, const char inizializzazione,
    const std::vector<double> &ascisse, const std::vector<double> &ordinate,
    const int iterazionikmeans, const char algoritmo, const int iterazioniEM, const double sogliaconvergenza,
    const char criterioscelto, double &criterio, std::vector<int> &clusterfinali, const bool stampa) {
    // Per ogni numero di cluster possibile:
    for (int i = minimo; i < massimo; i++) {
        // Si dichiarano le variabili dei parametri da passare nella funzione.
        double logverosimiglianza = 0.0;
        std::vector<int> cluster(taglia);
        std::vector<double> pigrecifinali(i);
        std::vector<double> muascissefinali(i);
        std::vector<double> muordinatefinali(i);
        std::vector<double> sigma11finali(i);
        std::vector<double> sigma22finali(i);
        std::vector<double> sigma12finali(i);
        std::vector<double> nufinali(i);
        std::vector<std::vector<double>> pesifinali(taglia, std::vector<double>(i));
        // Viene eseguito il clustering.
        veroclustering(normale, i, iterazioniinizializzazione, taglia, inizializzazione, ascisse, ordinate,
                       iterazionikmeans, algoritmo, iterazioniEM, logverosimiglianza, cluster, sogliaconvergenza,
                       pigrecifinali, muascissefinali, muordinatefinali, sigma11finali, sigma22finali,
                       sigma12finali, nufinali, pesifinali);
        // Calcoliamo il numero di parametri: sono 6k per le misture di gaussiane e 7k per le misture di t di Student.
        int numeroparametri;
        if (normale){numeroparametri = 6*i-1;} else {numeroparametri = 7*i-1;}
        // A questo punto il criterio viene applicato: per prima cosa viene calcolato, e se batte il
        // record precedente, viene sostituito a questo, e inoltre vengono salvati i cluster ottenuti
        // come soluzione migliore.
        if (criterioscelto == 'A') {
            const double criteriocorrente = -2.0 * logverosimiglianza + 2.0 * numeroparametri;
            if (criteriocorrente < criterio) {
                criterio = criteriocorrente;
                clusterfinali = cluster;
            }
            if (stampa){
            for (int gruppo = 0; gruppo < i; gruppo++){
                std::cout << "Con " << i << " componenti, la componente " << gruppo << " ha gradi di libertà " << nufinali[gruppo] << "\n" << std::flush;
            }
            std::cout << i << " componenti e AIC: " << criteriocorrente << "\n" << std::flush;
        }
        }
        if (criterioscelto == 'B') {
            const double criteriocorrente = -2.0 * logverosimiglianza + numeroparametri * std::log(taglia);
            if (criteriocorrente < criterio) {
                criterio = criteriocorrente;
                clusterfinali = cluster;
            }
            if (stampa){
            for (int gruppo = 0; gruppo < i; gruppo++){
                std::cout << "Con " << i << " componenti, la componente " << gruppo << " ha gradi di libertà " << nufinali[gruppo] << "\n" << std::flush;
            }
            std::cout << i << " componenti e BIC: " << criteriocorrente << "\n" << std::flush;
        }
        }
        if (criterioscelto == 'I') {
            double criteriocorrente = -2.0 * logverosimiglianza + numeroparametri * std::log(taglia);
            for (int j = 0; j < taglia; j++) {
                for (int k = 0; k < i; k++) {
                    const double peso = pesifinali[j][k];
                    if (peso > 0.0){criteriocorrente += 2.0 * peso * std::log(peso);}
                }
            }
            if (criteriocorrente < criterio) {
                criterio = criteriocorrente;
                clusterfinali = cluster;
            }
        }
    }
}


// Funzione che esegue il clustering usando un approccio bootstrap per selezionare il numero di componenti.
// I parametri sono:
// Minimo e massimo: due interi che indicano che intervallo del vettore statisticheempiriche nella
// funzione clustering viene occupato dal thread corrente.
// Taglia: numerosità del campione.
// Normale: "true" se vanno usate misture di gaussiane, "false" se vanno usate misture di t di Student.
// I: numero di componenti corrente.
// Pigreci: un riferimento a un vettore che contiene i parametri \pi_g. Stessa cosa per muascisse,
// muordinate, sigma11, sigma22, sigma12 e nu.
// Iterazioniinizializzazione: numero di volte che va ripetuto l'algoritmo di inizializzazione
// dei parametri eseguito prima dell'algoritmo EM nel clustering.
// Inizializzazione: una stringa tra "kmeans", "kmedoidi" e "casuale", che sceglie l'algoritmo di inizializzazione.
// Algoritmo: una stringa tra "EM", "SEM" e "CEM", che determina con quale algoritmo vengono aggiornati
// i parametri dei cluster.
// IterazioniEM: numero di iterazioni dell'algoritmo di aggiornamento dei parametri.
// Sogliaconvergenza: numero \epsilon che viene usato nella regola di arresto.
// Statisticheempiriche: un riferimento al vettore di statistiche empiriche che deve essere riempito.
void clusteringconbootstrap(const int minimo, const int massimo, const int taglia, const bool normale,
    const int i, std::vector<double> &pigreci, std::vector<double> &muascisse, std::vector<double> &muordinate,
    std::vector<double> &sigma11, std::vector<double> &sigma22, std::vector<double> &sigma12,
    std::vector<double> &nu, const int iterazioniinizializzazione,
    const char inizializzazione, const int iterazionikmeans, const char algoritmo,
    const int iterazioniEM,
    const double sogliaconvergenza, std::vector<double> &statisticheempiriche) {
    // Prepariamo il generatore di numeri casuali.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    // Per ogni statistica empirica da calcolare:
    for (int j = minimo; j < massimo; j++) {
        // Prepariamo due vettori che conterranno il campione bootstrap.
        std::vector<double> ascissebootstrap(taglia);
        std::vector<double> ordinatebootstrap(taglia);
        // Prepariamo due vettori che conterranno una versione non ancora pronta del campione.
        std::vector<double> ascissegrezze(taglia);
        std::vector<double> ordinategrezze(taglia);
        // Se usavamo le misture di gaussiane:
        if (normale) {
            // Estraiamo un campione da una normale standard.
            std::normal_distribution<> gaussiana(0.0, 1.0);
            for (int k = 0; k < taglia; k++) {
                ascissegrezze[k] = gaussiana(generatore);
                ordinategrezze[k] = gaussiana(generatore);
            }
            // Estraiamo anche gli indici delle componenti per ogni osservazione da una categorica.
            std::discrete_distribution<> estrattorecomponenti(pigreci.begin(), pigreci.end());
            // Adattiamo ogni osservazione alla sua componente aggiungendone la media e facendo in modo
            // che la matrice di covarianza sia quella trovata.
            for (int k = 0; k < taglia; k++) {
                const int componente = estrattorecomponenti(generatore);
                const double cholesky11 = std::sqrt(sigma11[componente]);
                const double cholesky12 = sigma12[componente] / cholesky11;
                const double cholesky22 = std::sqrt(sigma22[componente] - cholesky12 * cholesky12);
                ascissebootstrap[k] = muascisse[componente]+cholesky11*ascissegrezze[k];
                ordinatebootstrap[k] = muordinate[componente]+cholesky12*ascissegrezze[k]+cholesky22*ordinategrezze[k];
            }
            // Se invece usavamo misture di t di Student:
        } else {
            // Iniziamo di nuovo estraendo un campione da una normale standard.
            std::normal_distribution<> gaussiana(0.0, 1.0);
            for (int k = 0; k < taglia; k++) {
                ascissegrezze[k] = gaussiana(generatore);
                ordinategrezze[k] = gaussiana(generatore);
            }
            // Di nuovo estraiamo a quale componente appartiene ciascun'osservazione da una categorica.
            std::discrete_distribution<> estrattorecomponenti(pigreci.begin(), pigreci.end());
            // Riapplichiamo delle trasformazioni per rendere le osservazioni conformi alle medie
            // e matrici di covarianze della propria componente. Dopo di che, rendiamo le osservazioni
            // delle t di Student con \nu gradi di libertà.
            std::vector<std::gamma_distribution<double>> chiquadro;
            chiquadro.reserve(i-1);
            for (double v : nu){
                if (v/2.0 <= 0.2){chiquadro.emplace_back(0.2, 2.0);} else {chiquadro.emplace_back(v/2.0, 2.0);}
            }
            for (int k = 0; k < taglia; k++) {
                const int componente = estrattorecomponenti(generatore);
                const double cholesky11 = std::sqrt(sigma11[componente]);
                const double cholesky12 = sigma12[componente] / cholesky11;
                const double cholesky22 = std::sqrt(sigma22[componente] - cholesky12 * cholesky12);
                const double ascissanormale = cholesky11 * ascissegrezze[k];
                const double ordinatanormale = cholesky12*ascissegrezze[k]+cholesky22*ordinategrezze[k];
                if (nu[componente] <= 0.0){throw std::runtime_error("Un grado di libertà è non positivo");}
                const double denominatore = std::sqrt(chiquadro[componente](generatore) / nu[componente]);
                ascissebootstrap[k] = ascissanormale/denominatore+muascisse[componente];
                ordinatebootstrap[k] = ordinatanormale/denominatore+muordinate[componente];
            }
        }
        // Adesso partendo dal campione bootstrap, che è stato estratto sotto l'ipotesi nulla,
        // ricalcoliamo la statistica -2log\lambda empirica.
        double logverosimiglianzabootstrap1;
        std::vector<int> clusterbootstrap1(taglia);
        std::vector<double> pigrecibootstrap1(i-1);
        std::vector<double> muascissebootstrap1(i-1);
        std::vector<double> muordinatebootstrap1(i-1);
        std::vector<double> sigma11bootstrap1(i-1);
        std::vector<double> sigma22bootstrap1(i-1);
        std::vector<double> sigma12bootstrap1(i-1);
        std::vector<double> nubootstrap1(i-1);
        std::vector<std::vector<double>> pesibootstrap1(taglia, std::vector<double>(i - 1));
        veroclustering(normale, i - 1, iterazioniinizializzazione, taglia, inizializzazione,
                       ascissebootstrap, ordinatebootstrap, iterazionikmeans, algoritmo, iterazioniEM,
                       logverosimiglianzabootstrap1,
                       clusterbootstrap1, sogliaconvergenza, pigrecibootstrap1, muascissebootstrap1,
                       muordinatebootstrap1, sigma11bootstrap1, sigma22bootstrap1,
                       sigma12bootstrap1, nubootstrap1, pesibootstrap1);
        double logverosimiglianzabootstrap2;
        std::vector<int> clusterbootstrap2(taglia);
        std::vector<double> pigrecibootstrap2(i);
        std::vector<double> muascisse2(i);
        std::vector<double> muordinate2(i);
        std::vector<double> sigma11bootstrap2(i);
        std::vector<double> sigma22bootstrap2(i);
        std::vector<double> sigma12bootstrap2(i);
        std::vector<double> nubootstrap2(i);
        std::vector<std::vector<double>> pesibootstrap2(taglia, std::vector<double>(i));
        veroclustering(normale, i, iterazioniinizializzazione, taglia, inizializzazione,
                       ascissebootstrap, ordinatebootstrap, iterazionikmeans, algoritmo, iterazioniEM,
                       logverosimiglianzabootstrap2,
                       clusterbootstrap2, sogliaconvergenza, pigrecibootstrap2, muascisse2,
                       muordinate2, sigma11bootstrap2, sigma22bootstrap2, sigma12bootstrap2,
                       nubootstrap2, pesibootstrap2);
        statisticheempiriche[j] = -2.0 * (logverosimiglianzabootstrap1 - logverosimiglianzabootstrap2);
    }
}


// Funzione che applica una procedura di doppio bootstrap per selezionare il numero di cluster. I parametri sono:
// Minimo e massimo: estremi dell'intervallo all'interno di statisticheempiriche che deve essere
// riempito dal thread attuale.
// Taglia: numerosità del campione.
// Normale: "true" se usiamo misture di gaussiane e "false" se usiamo misture di t di Student.
// I: numero di cluster attualmente testato come ipotesi alternativa.
// Pigreci: un riferimento al vettore dei parametri \pi_g ottenuti clusterizzando il campione
// con i-1 componenti. Stessa cosa per muascisse, muordiante, sigma11, sigma22, sigma12 e nu.
// Iterazioniinizializzazione: i parametri da passare a veroclustering. Stessa cosa per inizializzazione,
// iterazionikmeans, algoritmo, iterazioniEM e sogliaconvergenza.
// Statisticheempiriche: un riferimento al vettore delle statistiche empiriche del primo strato bootstrap
// da riempire.
// Tutteinterne: un riferimento al vettore di vettori per le statistiche del secondo strato bootstrap.
// Bootstrapinterni: numerosità dei vettori interni di tutteinterne.
void clusteringcondoppiobootstrap(const int minimo, const int massimo, const int taglia, const bool normale,
    const int i, std::vector<double> &pigreci, std::vector<double> &muascisse, std::vector<double> &muordinate,
    std::vector<double> &sigma11, std::vector<double> &sigma22, std::vector<double> &sigma12,
    std::vector<double> &nu, const int iterazioniinizializzazione,
    const char inizializzazione, const int iterazionikmeans, const char algoritmo, const int iterazioniEM,
    const double sogliaconvergenza, std::vector<double> &statisticheempiriche,
    std::vector<std::vector<double>> &tutteinterne,
    const int bootstrapinterni) {
    // Prepariamo il generatore di numeri casuali.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    // Per ogni statistica da calcolare:
    for (int j = minimo; j < massimo; j++) {
        // Generiamo un campione bootstrap da una normale o una t di Student, in base a quanto indicato.
        // Il funzionamento è simile alla funzione clusteringconbootstrap.
        std::vector<double> ascissebootstrap(taglia);
        std::vector<double> ordinatebootstrap(taglia);
        std::vector<double> ascissegrezze(taglia);
        std::vector<double> ordinategrezze(taglia);
        // Se usavamo le misture di gaussiane:
        if (normale) {
            // Estraiamo un campione da una normale standard.
            std::normal_distribution<> gaussiana(0.0, 1.0);
            for (int k = 0; k < taglia; k++) {
                ascissegrezze[k] = gaussiana(generatore);
                ordinategrezze[k] = gaussiana(generatore);
            }
            // Estraiamo anche gli indici delle componenti per ogni osservazione da una categorica.
            std::discrete_distribution<> estrattorecomponenti(pigreci.begin(), pigreci.end());
            // Adattiamo ogni osservazione alla sua componente aggiungendone la media e facendo in modo
            // che la matrice di covarianza sia quella trovata.
            for (int k = 0; k < taglia; k++) {
                const int componente = estrattorecomponenti(generatore);
                const double cholesky11 = std::sqrt(sigma11[componente]);
                const double cholesky12 = sigma12[componente] / cholesky11;
                const double cholesky22 = std::sqrt(sigma22[componente] - cholesky12 * cholesky12);
                ascissebootstrap[k] = muascisse[componente]+cholesky11*ascissegrezze[k];
                ordinatebootstrap[k] = muordinate[componente]+cholesky12*ascissegrezze[k]+cholesky22*ordinategrezze[k];
            }
            // Se invece usavamo misture di t di Student:
        } else {
            // Iniziamo di nuovo estraendo un campione da una normale standard.
            std::normal_distribution<> gaussiana(0.0, 1.0);
            for (int k = 0; k < taglia; k++) {
                ascissegrezze[k] = gaussiana(generatore);
                ordinategrezze[k] = gaussiana(generatore);
            }
            // Di nuovo estraiamo a quale componente appartiene ciascun'osservazione da una categorica.
            std::discrete_distribution<> estrattorecomponenti(pigreci.begin(), pigreci.end());
            // Riapplichiamo delle trasformazioni per rendere le osservazioni conformi alle medie
            // e matrici di covarianze della propria componente. Dopo di che, rendiamo le osservazioni
            // delle t di Student con \nu gradi di libertà.
            std::vector<std::gamma_distribution<double>> chiquadro;
            chiquadro.reserve(i-1);
            for (double v : nu){
                if (v/2.0 <= 0.2){chiquadro.emplace_back(0.2, 2.0);} else {chiquadro.emplace_back(v/2.0, 2.0);}
            }
            for (int k = 0; k < taglia; k++) {
                const int componente = estrattorecomponenti(generatore);
                const double cholesky11 = std::sqrt(sigma11[componente]);
                const double cholesky12 = sigma12[componente] / cholesky11;
                const double cholesky22 = std::sqrt(sigma22[componente] - cholesky12 * cholesky12);
                const double ascissanormale = cholesky11 * ascissegrezze[k];
                const double ordinatanormale = cholesky12*ascissegrezze[k]+cholesky22*ordinategrezze[k];
                if (nu[componente] <= 0.0){throw std::runtime_error("Un grado di libertà è non positivo");}
                const double denominatore = std::sqrt(chiquadro[componente](generatore) / nu[componente]);
                ascissebootstrap[k] = ascissanormale/denominatore+muascisse[componente];
                ordinatebootstrap[k] = ordinatanormale/denominatore+muordinate[componente];
            }
        }
        // Adesso operiamo il primo strato bootstrap, calcolando -2log\lambda per il campione estratto.
        double logverosimiglianzabootstrap1;
        std::vector<int> clusterbootstrap1(taglia);
        std::vector<double> pigreci1(i-1);
        std::vector<double> muascisse1(i-1);
        std::vector<double> muordinate1(i-1);
        std::vector<double> sigma111(i-1);
        std::vector<double> sigma221(i-1);
        std::vector<double> sigma121(i-1);
        std::vector<double> nu1(i-1);
        std::vector<std::vector<double>> pesibootstrap1(taglia, std::vector<double>(i - 1));
        veroclustering(normale, i - 1, iterazioniinizializzazione, taglia, inizializzazione,
                       ascissebootstrap, ordinatebootstrap, iterazionikmeans, algoritmo, iterazioniEM,
                       logverosimiglianzabootstrap1,
                       clusterbootstrap1, sogliaconvergenza, pigreci1, muascisse1, muordinate1,
                       sigma111, sigma221, sigma121, nu1, pesibootstrap1);
        double logverosimiglianzabootstrap2;
        std::vector<int> clusterbootstrap2(taglia);
        std::vector<double> pigreci2(i);
        std::vector<double> muascisse2(i);
        std::vector<double> muordinate2(i);
        std::vector<double> sigma112(i);
        std::vector<double> sigma222(i);
        std::vector<double> sigma122(i);
        std::vector<double> nu2(i);
        std::vector<std::vector<double> > pesibootstrap2(taglia, std::vector<double>(i));
        veroclustering(normale, i, iterazioniinizializzazione, taglia, inizializzazione,
                       ascissebootstrap, ordinatebootstrap, iterazionikmeans, algoritmo, iterazioniEM,
                       logverosimiglianzabootstrap2,
                       clusterbootstrap2, sogliaconvergenza, pigreci2, muascisse2, muordinate2,
                       sigma112, sigma222, sigma122, nu2, pesibootstrap2);
        statisticheempiriche[j] = -2.0 * (logverosimiglianzabootstrap1 - logverosimiglianzabootstrap2);
        // Ora dobbiamo eseguire il secondo strato bootstrap, in cui estraiamo un certo numero di
        // campioni bootstrap interni usando i parametri ottenuti sotto i-1 componenti; fatto questo
        // ricalcoliamo per ciascuno un'altra statistica -2log\lambda, che serviranno per aggiustare
        // quella del primo strato bootstrap, mettendole nel vettore tutteinterne.
        std::vector<double> statisticheinterne(bootstrapinterni);
        for (int k = 0; k < bootstrapinterni; k++) {
            std::vector<double> ascisseinterne(taglia);
            std::vector<double> ordinateinterne(taglia);
            std::vector<double> ascisseinternegrezze(taglia);
            std::vector<double> ordinateinternegrezze(taglia);
            // Se usavamo le misture di gaussiane:
            if (normale) {
                // Estraiamo un campione da una normale standard.
                std::normal_distribution<> gaussiana(0.0, 1.0);
                for (int l = 0; l < taglia; l++) {
                    ascisseinternegrezze[l] = gaussiana(generatore);
                    ordinateinternegrezze[l] = gaussiana(generatore);
                }
                // Estraiamo anche gli indici delle componenti per ogni osservazione da una categorica.
                std::discrete_distribution<> estrattorecomponenti(pigreci1.begin(), pigreci1.end());
                // Adattiamo ogni osservazione alla sua componente aggiungendone la media e facendo in modo
                // che la matrice di covarianza sia quella trovata.
                for (int l = 0; l < taglia; l++) {
                    const int componente = estrattorecomponenti(generatore);
                    const double cholesky11 = std::sqrt(sigma111[componente]);
                    const double cholesky12 = sigma121[componente] / cholesky11;
                    const double cholesky22 = std::sqrt(sigma221[componente] - cholesky12 * cholesky12);
                    ascisseinterne[l] = muascisse1[componente]+cholesky11*ascisseinternegrezze[l];
                    ordinateinterne[l] = muordinate1[componente]+cholesky12*ascisseinternegrezze[l]+cholesky22*ordinateinternegrezze[l];
                }
                // Se invece usavamo misture di t di Student:
            } else {
                // Iniziamo di nuovo estraendo un campione da una normale standard.
                std::normal_distribution<> gaussiana(0.0, 1.0);
                for (int l = 0; l < taglia; l++) {
                    ascisseinternegrezze[l] = gaussiana(generatore);
                    ordinateinternegrezze[l] = gaussiana(generatore);
                }
                // Di nuovo estraiamo a quale componente appartiene ciascun'osservazione da una categorica.
                std::discrete_distribution<> estrattorecomponenti(pigreci1.begin(), pigreci1.end());
                // Riapplichiamo delle trasformazioni per rendere le osservazioni conformi alle medie
                // e matrici di covarianze della propria componente. Dopo di che, rendiamo le osservazioni
                // delle t di Student con \nu gradi di libertà.
                std::vector<std::gamma_distribution<double>> chiquadro;
                chiquadro.reserve(i-1);
                for (double v : nu1){
                    if (v/2.0 <= 0.2){chiquadro.emplace_back(0.2, 2.0);} else {chiquadro.emplace_back(v/2.0, 2.0);}
                }
                for (int l = 0; l < taglia; l++) {
                    const int componente = estrattorecomponenti(generatore);
                    const double cholesky11 = std::sqrt(sigma111[componente]);
                    const double cholesky12 = sigma121[componente] / cholesky11;
                    const double cholesky22 = std::sqrt(sigma221[componente] - cholesky12 * cholesky12);
                    const double ascissanormale = cholesky11 * ascisseinternegrezze[l];
                    const double ordinatanormale = cholesky12*ascisseinternegrezze[l]+cholesky22*ordinateinternegrezze[l];
                    if (nu1[componente] <= 0.0){throw std::runtime_error("Un grado di libertà è non positivo");}
                    const double denominatore = std::sqrt(chiquadro[componente](generatore) / nu1[componente]);
                    ascisseinterne[l] = ascissanormale/denominatore+muascisse1[componente];
                    ordinateinterne[l] = ordinatanormale/denominatore+muordinate1[componente];
                }
            }
            double logverosimiglianzainterna1;
            std::vector<int> clusterinterni1(taglia);
            std::vector<double> pigreciinterni1(i-1);
            std::vector<double> muascisseinterne1(i-1);
            std::vector<double> muordinateinterne1(i-1);
            std::vector<double> sigma11interne1(i-1);
            std::vector<double> sigma22interne1(i-1);
            std::vector<double> sigma12interne1(i-1);
            std::vector<double> nuinterni1(i-1);
            std::vector<std::vector<double>> pesiinterni1(taglia, std::vector<double>(i - 1));
            veroclustering(normale, i - 1, iterazioniinizializzazione, taglia, inizializzazione,
                           ascisseinterne, ordinateinterne, iterazionikmeans, algoritmo, iterazioniEM,
                           logverosimiglianzainterna1,
                           clusterinterni1, sogliaconvergenza, pigreciinterni1, muascisseinterne1,
                           muordinateinterne1, sigma11interne1, sigma22interne1,
                           sigma12interne1, nuinterni1, pesiinterni1);
            double logverosimiglianzainterna2;
            std::vector<int> clusterinterni2(taglia);
            std::vector<double> pigreciinterni2(i);
            std::vector<double> muascisseinterne2(i);
            std::vector<double> muordinateinterne2(i);
            std::vector<double> sigma11interne2(i);
            std::vector<double> sigma22interne2(i);
            std::vector<double> sigma12interne2(i);
            std::vector<double> nuinterni2(i);
            std::vector<std::vector<double>> pesiinterni2(taglia, std::vector<double>(i));
            veroclustering(normale, i, iterazioniinizializzazione, taglia, inizializzazione,
                           ascisseinterne, ordinateinterne, iterazionikmeans, algoritmo, iterazioniEM,
                           logverosimiglianzainterna2,
                           clusterinterni2, sogliaconvergenza, pigreciinterni2, muascisseinterne2,
                           muordinateinterne2, sigma11interne2, sigma22interne2,
                           sigma12interne2, nuinterni2, pesiinterni2);
            statisticheinterne[k] = -2.0 * (logverosimiglianzainterna1 - logverosimiglianzainterna2);
        }
        tutteinterne[j] = statisticheinterne;
    }
}


void clusteringconcrossvalidation(const int minimo, const int massimo, const int taglia, const int fold,
    const int i, const std::vector<double> &ascisse, const std::vector<double> &ordinate,
    const bool normale,
    const int iterazioniinizializzazione, const char inizializzazione, const int iterazionikmeans,
    const char algoritmo, const int iterazioniEM, const double sogliaconvergenza,
    std::vector<double> &stimecrossvalidate) {
    // Per ogni fold da controllare:
    for (int j = minimo; j < massimo; j++) {
        // Dividiamo il training set dal test set. Le osservazioni la cui posizione diviso il numero
        // di fold restituisce j, l'indice della fold da controllare, sono messe nel test set.
        int tagliatestset;
        if (taglia % fold == 0){tagliatestset = taglia/fold;} else {tagliatestset = taglia/fold+1;}
        std::vector<double> ascissetraining(taglia - tagliatestset);
        std::vector<double> ordinatetraining(taglia-tagliatestset);
        std::vector<double> ascissetest(tagliatestset);
        std::vector<double> ordinatetest(tagliatestset);
        int trainingattuale = 0;
        int testattuale = 0;
        for (int k = 0; k < taglia; k++) {
            if (k % fold == j) {
                ascissetest[testattuale] = ascisse[k];
                ordinatetest[testattuale] = ordinate[k];
                testattuale++;
            } else {
                ascissetraining[testattuale] = ascisse[k];
                ordinatetraining[testattuale] = ordinate[k];
                trainingattuale++;
            }
        }
        // Ricaviamo i parametri dal training set con il folding attuale.
        std::vector<double> pigreci(i);
        std::vector<double> muascisse(i);
        std::vector<double> muordinate(i);
        std::vector<double> sigma11(i);
        std::vector<double> sigma22(i);
        std::vector<double> sigma12(i);
        std::vector<double> nu(i);
        double logverosimiglianza = 0.0;
        std::vector<int> cluster(taglia-tagliatestset);
        std::vector<std::vector<double> > pesi(taglia-tagliatestset, std::vector<double>(i));
        veroclustering(normale, i, iterazioniinizializzazione, taglia-tagliatestset, inizializzazione,
                       ascissetraining, ordinatetraining, iterazionikmeans, algoritmo, iterazioniEM,
                       logverosimiglianza,
                       cluster, sogliaconvergenza, pigreci, muascisse, muordinate, sigma11,
                       sigma22, sigma12, nu, pesi);
        // Calcoliamo i determinanti e le inverse delle matrici di covarianze.
        std::vector<double> determinanti(i);
        std::vector<double> inversa11(i);
        std::vector<double> inversa22(i);
        std::vector<double> inversa12(i);
        for (int k = 0; k < i; k++) {
            const double prodottovarianze = std::sqrt(sigma11[k]*sigma22[k]);
            double covarianza = sigma12[k];
            if (covarianza < -prodottovarianze || covarianza > prodottovarianze){
                covarianza = std::copysign(prodottovarianze, covarianza);
                sigma12[k] = covarianza;
            }
            double determinante = sigma11[k]*sigma22[k]-covarianza*covarianza;
            if (determinante < 0.000001) {
                sigma11[k] += 0.001;
                sigma22[k] += 0.001;
                determinante = sigma11[k]*sigma22[k]-covarianza*covarianza;
            }
            determinanti[k] = determinante;
            inversa11[k] = sigma22[k]/determinante;
            inversa22[k] = sigma11[k]/determinante;
            inversa12[k] = -covarianza/determinante;
        }
        // Adesso calcoliamo il log-predictive score di questa fold.
        double logpredictivescore = 0.0;
        if (normale) {
            std::vector<double> precostanti(i);
            for (int k = 0; k < i; k++){
                precostanti[k] = std::log(pigreci[k])-logduepigreco-std::log(determinanti[k])/2.0;
            }
            for (int k = 0; k < tagliatestset; k++) {
                std::vector<double> esponenti(i);
                double esponentemassimo = -std::numeric_limits<double>::infinity();
                const double ascissa = ascissetest[k];
                const double ordinata = ordinatetest[k];
                for (int l = 0; l < i; l++) {
                    const double distanzaascisse = ascissa-muascisse[l];
                    const double distanzaordinate = ordinata-muordinate[l];
                    const double esponente = precostanti[l]-(inversa11[l]*distanzaascisse*distanzaascisse+
                        2.0*inversa12[l]*distanzaascisse*distanzaordinate+inversa22[l]*distanzaordinate*
                            distanzaordinate)/2.0;
                    esponenti[l] = esponente;
                    if (esponente > esponentemassimo){esponentemassimo = esponente;}
                }
                double sommatoria = 0.0;
                for (int l = 0; l < i; l++){sommatoria += std::exp(esponenti[l]-esponentemassimo);}
                logpredictivescore += esponentemassimo+std::log(sommatoria);
            }
        } else {
            std::vector<double> precostanti(i);
            for (int k = 0; k < i; k++){
                const double grado = nu[k];
                precostanti[k] = std::log(pigreci[k])+std::lgamma(grado/2.0+1.0)-std::log(grado*pigreco)-
                    std::log(determinanti[k])/2.0-std::lgamma(grado/2.0);
            }
            for (int k = 0; k < taglia; k++) {
                std::vector<double> esponenti(i);
                double esponentemassimo = -std::numeric_limits<double>::infinity();
                const double ascissa = ascissetest[k];
                const double ordinata = ordinatetest[k];
                for (int l = 0; l < i; l++) {
                    const double grado = nu[l];
                    if (grado <= 0.0){throw std::runtime_error("Un grado di libertà è non positivo");}
                    const double distanzaascisse = ascissa-muascisse[l];
                    const double distanzaordinate = ordinata-muordinate[l];
                    const double esponente = precostanti[l]-(grado+2.0)/2.0*std::log(1.0+(inversa11[l]*
                        distanzaascisse*distanzaascisse+2.0*inversa12[l]*distanzaascisse*distanzaordinate+
                            inversa22[l]*distanzaordinate*distanzaordinate)/grado);
                    esponenti[l] = esponente;
                    if (esponente > esponentemassimo){esponentemassimo = esponente;}
                }
                double sommatoria = 0.0;
                for (int l = 0; l < i; l++){sommatoria += std::exp(esponenti[l]-esponentemassimo);}
                logpredictivescore += esponentemassimo+std::log(sommatoria);
            }
        }
        stimecrossvalidate[j] = logpredictivescore;
    }
}


// Funzione preparatoria per eseguire il clustering basato sul modello sul dataset.
// I parametri sono:
// Inizializzazione: una stringa tra "kmeans", "kmedoidi" e "casuale" che indica l'algoritmo
// di inizializzazione dei parametri nell'algoritmo EM.
// Taglia: numerosità del campione.
// Ascisse: vettore delle ascisse del dataset su cui fare clustering.
// Ordinate: stessa cosa per le ordinate.
// Iterazionikmeans: numero massimo di iterazioni che devono fare l'algoritmo k-means o k-medoidi.
// Componentimassime: numero massimo di componenti da considerare durante la selezione del modello
// (vengono considerati tutti i numeri interi tra 2 e componentimassime).
// Selezione: una stringa tra "criterio", "bootstrap", "doppiobootstrap" e "crossvalidation" che
// indica il metodo di selezione del numero di componenti del modello mistura.
// Campionibootstrap: numero di iterazioni del (primo strato) bootstrap se si usa il metodo bootstrap
// o doppio bootstrap.
// Alfabootstrap: probabilità di errore di I tipo che si vuole fissare per i test d'ipotesi
// condotti nell'approccio bootstrap o doppio bootstrap.
// Iterazioniinizializzazione: numero di volte che viene tentato l'algoritmo di inizializzazione
// in cerca di quella migliore.
// IterazioniEM: numero massimo di iterazioni dell'algoritmo di aggiornamento dei parametri.
// Normale: "true" se si vogliono usare misture di gaussiane, "false" se si vogliono usare misture di t di Student.
// Algoritmo: una stringa tra "EM", "SEM" e "CEM" che indica con quale algoritmo si aggiornano i parametri.
// Sogliaconvergenza: numero utilizzato per specificare la regola di arresto nell'algoritmo
// di aggiornamento dei parametri.
// Criterioscelto: una stringa tra "AIC", "BIC", "ICL" che indica quale criterio informativo va usato
// se si usa questo metodo di selezione del modello.
// Bootstrapinterni: numero di iterazioni del secondo strato bootstrap nell'approccio di doppio bootstrap.
// Fold: specifica il v quando si utilizza un approccio cross-validation v-fold.
std::vector<int> clustering(const char inizializzazione, const int taglia,
    const std::vector<double> &ascisse, const std::vector<double> &ordinate, const int iterazionikmeans,
    const int componentimassime, const char selezione, const int campionibootstrap,
    const double alfabootstrap, const int iterazioniinizializzazione, const int iterazioniEM,
    const bool normale, const char algoritmo, const double sogliaconvergenza,
    const char criterioscelto, const int bootstrapinterni, int fold) {
    if (inizializzazione != 'K' && inizializzazione != 'D' && inizializzazione != 'C'){throw std::runtime_error("Algoritmo di inizializzazione dei parametri non valido");}
    if (taglia < componentimassime){throw std::runtime_error("Ci sono meno osservazioni che componenti");}
    if (ascisse.size() != taglia || ordinate.size() != taglia){throw std::runtime_error("Il dataset non coincide con la taglia");}
    if (iterazionikmeans < 1){throw std::runtime_error("Le iterazioni devono essere positive");}
    if (componentimassime < 2){throw std::runtime_error("Il clustering deve avere almeno due componenti");}
    if (selezione != 'C' && selezione != 'B' && selezione != 'D' && selezione != 'V' && selezione != 'M'){throw std::runtime_error("Criterio di selezione del modello non valido");}
    if (campionibootstrap < 2){throw std::runtime_error("Il bootstrap deve fare almeno due iterazioni");}
    if (alfabootstrap <= 0.0 || alfabootstrap >= 1.0){throw std::runtime_error("Alfa non valido");}
    if (iterazioniinizializzazione < 1){throw std::runtime_error("Le iterazioni devono essere positive");}
    if (iterazioniEM < 1){throw std::runtime_error("Le iterazioni devono essere positive");}
    if (algoritmo != 'E' && algoritmo != 'S' && algoritmo != 'C'){throw std::runtime_error("Algoritmo di clustering non valido");}
    if (sogliaconvergenza <= 0.0){throw std::runtime_error("La soglia di convergenza deve essere positiva");}
    if (criterioscelto != 'A' && criterioscelto != 'B' && criterioscelto != 'I'){throw std::runtime_error("Criterio informativo non valido");}
    if (bootstrapinterni < 2){throw std::runtime_error("Il bootstrap deve fare almeno due iterazioni");}
    if (fold < 2){throw std::runtime_error("La cross-validation deve fare almeno due fold");}
    if (selezione == 'M') {
        double mediaascisse = 0.0;
        double mediaordinate = 0.0;
        for (int i = 0; i < taglia; i++) {
            mediaascisse += ascisse[i];
            mediaordinate += ordinate[i];
        }
        mediaascisse /= taglia;
        mediaordinate /= taglia;
        double deviazioneascisse = 0.0;
        double deviazioneordinate = 0.0;
        for (int i = 0; i < taglia; i++){
            const double differenzaascisse = ascisse[i]-mediaascisse;
            const double differenzaordinate = ordinate[i]-mediaordinate;
            deviazioneascisse += differenzaascisse*differenzaascisse;
            deviazioneordinate += differenzaordinate*differenzaordinate;
        }
        deviazioneascisse /= taglia-1;
        deviazioneordinate /= taglia-1;
        const double bandwidthascisse = 1.06*std::sqrt(deviazioneascisse)/std::pow(taglia, 0.2);
        const double bandwidthordinate = 1.06*std::sqrt(deviazioneordinate)/std::pow(taglia, 0.2);
        const double frazionegaussiana = 1.0/std::sqrt(2.0*pigreco);
        const double frazionemediatrice = 1.0/(taglia*bandwidthascisse*bandwidthordinate);
        const double frazioneintera = frazionegaussiana*frazionemediatrice;
        std::array<std::array<double, 100>, 100> griglia;
        for (int i = 0; i < 100; i++) {
            const double ascissacella = (2.0*i-99.0)/100.0;
            for (int j = 0; j < 100; j++){
                const double ordinatacella = (2.0*j-99.0)/100.0;
                double kernel = 0.0;
                for (int k = 0; k < taglia; k++) {
                    const double primoargomento = (ascissacella-ascisse[k])/bandwidthascisse;
                    const double secondoargomento = (ordinatacella-ordinate[k])/bandwidthordinate;
                    kernel += std::exp(-(primoargomento*primoargomento+secondoargomento*secondoargomento)/2.0);
                }
                griglia[i][j] = frazioneintera*kernel;
            }
        }
        int numeropicchi = 0;
        if (griglia[0][0] > griglia[1][0] && griglia[0][0] > griglia[0][1] && griglia[0][0] > griglia[1][1]){numeropicchi++;}
        if (griglia[99][0] > griglia[98][0] && griglia[99][0] > griglia[99][1] && griglia[99][0] > griglia[98][1]){numeropicchi++;}
        if (griglia[0][99] > griglia[0][98] && griglia[0][99] > griglia[1][99] && griglia[0][99] > griglia[1][98]){numeropicchi++;}
        if (griglia[99][99] > griglia[98][99] && griglia[99][99] > griglia[99][98] && griglia[99][99] > griglia[98][98]){numeropicchi++;}
        for (int i = 1; i < 100; i++) {
            const double cella1 = griglia[0][i];
            if (cella1 > griglia[0][i-1] && cella1 > griglia[0][i+1] && cella1 > griglia[1][i-1] && cella1 > griglia[1][i] && cella1 > griglia[1][i+1]){numeropicchi++;}
            const double cella2 = griglia[i][0];
            if (cella2 > griglia[i-1][0] && cella2 > griglia[i+1][0] && cella2 > griglia[i-1][1] && cella2 > griglia[i][1] && cella2 > griglia[i+1][1]){numeropicchi++;}
            const double cella3 = griglia[99][i];
            if (cella3 > griglia[99][i-1] && cella3 > griglia[99][i+1] && cella3 > griglia[98][i-1] && cella3 > griglia[98][i] && cella3 > griglia[98][i+1]){numeropicchi++;}
            const double cella4 = griglia[i][99];
            if (cella4 > griglia[i-1][99] && cella4 > griglia[i+1][99] && cella4 > griglia[i-1][98] && cella4 > griglia[i][98] && cella4 > griglia[i+1][98]){numeropicchi++;}
            for (int j = 1; j < 100; j++) {
                const double cella = griglia[i][j];
                if (cella > griglia[i-1][j-1] && cella > griglia[i-1][j] && cella > griglia[i-1][j+1] && cella > griglia[i][j-1] && cella > griglia[i][j+1] && cella > griglia[i+1][j-1] && cella > griglia[i+1][j] && cella > griglia[i+1][j+1]){numeropicchi++;}
            }
        }
        if (numeropicchi < 2) {
            numeropicchi = 2;
        }
        double logverosimiglianza = 0.0;
        std::vector<int> cluster(taglia);
        std::vector<double> pigreci(numeropicchi);
        std::vector<double> muascisse(numeropicchi);
        std::vector<double> muordinate(numeropicchi);
        std::vector<double> sigma11(numeropicchi);
        std::vector<double> sigma22(numeropicchi);
        std::vector<double> sigma12(numeropicchi);
        std::vector<double> nu(numeropicchi, 5.0);
        std::vector<std::vector<double>> pesi(taglia, std::vector<double>(numeropicchi));
        veroclustering(normale, numeropicchi, iterazioniinizializzazione, taglia, inizializzazione, ascisse, ordinate,
            iterazionikmeans, algoritmo, iterazioniEM, logverosimiglianza, cluster, sogliaconvergenza, pigreci,
            muascisse, muordinate, sigma11, sigma22, sigma12, nu, pesi);
        return cluster;
    }
    if (selezione == 'C') {
        // Prepariamo i thread.
        int numerothread = static_cast<int>(std::thread::hardware_concurrency());
        if (numerothread == 0){numerothread = 4;}
        std::vector<std::thread> thread;
        thread.reserve(numerothread);
        // Assegniamo a ogni thread una parte del numero di componenti da esaminare: in particolare
        // iniziamo dando a tutti parti uguali e distribuendo poi il resto. Quindi se per
        // esempio vogliamo esaminare da 2 a 10 componenti e abbiamo 4 thread, dobbiamo distribuire
        // 9 numeri: allora diamo inizialmente int(9/4)=2 numeri a ogni thread, e poi assegniamo al primo
        // thread l'ultimo numero rimanente. Quindi i thread hanno {3, 2, 2, 2} numeri di componenti, e
        // in particolare il primo thread esaminerà il caso di 2, 3 e 4 componenti, il secondo di 5 e 6, il
        // terzo di 7 e 8 e il quarto di 9 e 10.
        std::vector<int> iterazioniathread(numerothread, (componentimassime-1)/numerothread);
        for (int i = 0; i < (componentimassime-1) % numerothread; i++) {
            iterazioniathread[i]++;
        }
        // Per ogni thread, inizializziamo un vettore che contenga la partizione e un valore numerico
        // che ricordi il record attuale del criterio informativo, che quindi parte da infinito.
        // Se viene trovata una partizione che batte il record attuale, questa viene salvata.
        std::vector<std::vector<int>> candidaticlusterfinali(numerothread);
        std::vector<double> candidaticriterio(numerothread, std::numeric_limits<double>::infinity());
        std::vector<std::exception_ptr> thread_exceptions(numerothread);
        int inizio = 2;
        for (int i = 0; i < numerothread; i++) {
            // Assegniamo a ogni thread i numeri k che gli spettano.
            int fine = inizio + iterazioniathread[i];
            thread.emplace_back([&, i, inizio, fine](){
                try {
                    clusteringconcriterio(inizio, fine, taglia, normale, iterazioniinizializzazione,
                        inizializzazione, ascisse, ordinate, iterazionikmeans, algoritmo, iterazioniEM, sogliaconvergenza,
                        criterioscelto, std::ref(candidaticriterio[i]),
                        std::ref(candidaticlusterfinali[i]));
                } catch(...) {
                    thread_exceptions[i] = std::current_exception();
                }
            });
            inizio = fine;
        }
        int it = 0;
        // Aspettiamo che i thread finiscano.
        for (auto& t : thread) {
            try {
                if (t.joinable()) {
                    t.join();
                } else {
                    std::cout << "Thread numero " << it << " inattaccabile\n" << std::flush;
                }
            } catch (const std::system_error& e) {
                std::cerr << "System_error sollevato col thread " << it << ": " << e.what() << "\n" << std::flush;
            } catch (const std::exception& e) {
                std::cerr << "Errore generico sollevato col thread " << it << ": " << e.what() << "\n" << std::flush;
            } catch (...) {
                std::cerr << "Eccezione ignota col thread " << it << "\n" << std::flush;
            }
            it++;
        }
        for (int i = 0; i < numerothread; i++){
            if (thread_exceptions[i]){
                try {
                    std::rethrow_exception(thread_exceptions[i]);
                } catch (const std::exception& e) {
                    std::cerr << "Il worker " << i << " ha lanciato: " << e.what() << "\n";
                } catch (...) {
                    std::cerr << "Il worker " << i << " ha lanciato un'eccezione fuori da std\n";
                }
            }
        }
        // Troviamo tra le soluzioni di tutti i thread quella col criterio minimo.
        if (numerothread == 1) {
            return candidaticlusterfinali[0];
        } else {
            std::vector<int> clusterfinali = candidaticlusterfinali[0];
            double criterio = candidaticriterio[0];
            for (int i = 1; i < numerothread; i++){
                if (candidaticriterio[i] < criterio) {
                    clusterfinali = candidaticlusterfinali[i];
                    criterio = candidaticriterio[i];
                }
            }
            return clusterfinali;
        }
    }
    if (selezione == 'B') {
        // Prepariamo i thread.
        int numerothread = static_cast<int>(std::thread::hardware_concurrency());
        if (numerothread == 0){numerothread = 4;}
        std::vector<std::thread> thread;
        thread.reserve(numerothread);
        std::vector<std::exception_ptr> thread_exceptions(numerothread);
        // Contrariamente al caso precedente in cui usavamo i criteri informativi, stavolta i thread
        // verranno utilizzati per calcolare le statistiche bootstrap. Dunque, anche se con un meccanismo
        // simile a prima, assegniamo a ciascuno il numero di campioni bootstrap da estrarre.
        std::vector<int> iterazioniathread(numerothread, campionibootstrap/numerothread);
        for (int i = 0; i < campionibootstrap % numerothread; i++) {
            iterazioniathread[i]++;
        }
        // Iniziamo dichiarando un vettore che conterrà la partizione migliore.
        std::vector<int> clusterfinali(taglia);
        // Cominciamo effettuando il clustering con due componenti, in modo da averne il valore della
        // logverosimiglianza e i parametri.
        double logverosimiglianza1 = 0.0;
        std::vector<int> cluster1(taglia, 0);
        std::vector<double> pigreci1(2);
        std::vector<double> muascisse1(2);
        std::vector<double> muordinate1(2);
        std::vector<double> sigma111(2);
        std::vector<double> sigma221(2);
        std::vector<double> sigma121(2);
        std::vector<double> nu1(2);
        std::vector<std::vector<double>> pesifinali1(taglia, std::vector<double>(2));
        veroclustering(normale, 2, iterazioniinizializzazione, taglia, inizializzazione, ascisse, ordinate,
            iterazionikmeans, algoritmo, iterazioniEM, logverosimiglianza1, cluster1, sogliaconvergenza,
            pigreci1, muascisse1, muordinate1, sigma111, sigma221, sigma121, nu1, pesifinali1);
        // Chiaramente, se componentimassime è 2, non c'è altro da fare: restituiamo la soluzione
        // appena ottenuta.
        if (componentimassime == 2){return cluster1;}
        // Per ogni successivo k possibile:
        for (int i = 3; i <= componentimassime; i++) {
            // Fittiamo il modello con i componenti (che è l'ipotesi alternativa).
            double logverosimiglianza2 = 0.0;
            std::vector<int> cluster2(taglia);
            std::vector<double> pigreci2(i);
            std::vector<double> muascisse2(i);
            std::vector<double> muordinate2(i);
            std::vector<double> sigma112(i);
            std::vector<double> sigma222(i);
            std::vector<double> sigma122(i);
            std::vector<double> nu2(i);
            std::vector<std::vector<double>> pesifinali2(taglia, std::vector<double>(i));
            veroclustering(normale, i, iterazioniinizializzazione, taglia, inizializzazione, ascisse, ordinate,
                iterazionikmeans, algoritmo, iterazioniEM, logverosimiglianza2, cluster2, sogliaconvergenza,
                pigreci2, muascisse2, muordinate2, sigma112, sigma222, sigma122, nu2, pesifinali2);
            // Adesso calcoliamo -2log\lambda.
            const double statisticaoriginale = -2.0*(logverosimiglianza1-logverosimiglianza2);
            // Adesso generiamo la distribuzione bootstrap delle statistiche -2log\lambda.
            std::vector<double> statisticheempiriche(campionibootstrap);
            // A ogni thread è assegnato un segmento del vettore statisticheempiriche, che deve riempire.
            int inizio = 0;
            for (int j = 0; j < numerothread; j++) {
                int fine = inizio + iterazioniathread[j];
                thread.emplace_back([&, i, inizio, fine](){
                    try {
                        clusteringconbootstrap(inizio, fine, taglia, normale, i,
                            std::ref(pigreci1), std::ref(muascisse1), std::ref(muordinate1),
                            std::ref(sigma111), std::ref(sigma221), std::ref(sigma121), std::ref(nu1),
                            iterazioniinizializzazione, inizializzazione, iterazionikmeans, algoritmo,
                            iterazioniEM, sogliaconvergenza, std::ref(statisticheempiriche));
                    } catch(...) {
                        thread_exceptions[j] = std::current_exception();
                    }
                });
                inizio = fine;
            }
            int it = 0;
            // Aspettiamo che i thread finiscano.
            for (auto& t : thread) {
                try {
                    if (t.joinable()) {
                        t.join();
                    } else {
                        std::cout << "Thread numero " << it << " inattaccabile\n" << std::flush;
                    }
                } catch (const std::system_error& e) {
                    std::cerr << "System_error sollevato col thread " << it << ": " << e.what() << "\n" << std::flush;
                } catch (const std::exception& e) {
                    std::cerr << "Errore generico sollevato col thread " << it << ": " << e.what() << "\n" << std::flush;
                } catch (...) {
                    std::cerr << "Eccezione ignota col thread " << it << "\n" << std::flush;
                }
                it++;
            }
            thread.clear();
            for (int j = 0; j < numerothread; j++){
                if (thread_exceptions[j]){
                    try {
                        std::rethrow_exception(thread_exceptions[j]);
                    } catch (const std::exception& e) {
                        std::cerr << "Il worker " << j << " ha lanciato: " << e.what() << "\n";
                    } catch (...) {
                        std::cerr << "Il worker " << j << " ha lanciato un'eccezione fuori da std\n";
                    }
                }
            }
            // Calcoliamo il quantile 1-alfa della distribuzione empirica delle statistiche bootstrap.
            std::sort(statisticheempiriche.begin(), statisticheempiriche.end());
            const double posizione = (1.0-alfabootstrap)*(campionibootstrap-1);
            const int sotto = static_cast<int>(std::floor(posizione));
            const int sopra = static_cast<int>(std::ceil(posizione));
            const double peso = posizione-sotto;
            const double sogliacritica = (1.0-peso)*statisticheempiriche[sotto]+peso*statisticheempiriche[sopra];
            // Se la statistica -2log\lambda è minore del quantile così ottenuto, allora accettiamo
            // l'ipotesi nulla e restituiamo i cluster così ottenuti.
            if (statisticaoriginale <= sogliacritica) {
                clusterfinali = cluster1;
                break;
            }
            // Se abbiamo raggiunto l'ultimo test possibile allora semplicemente decidiamo
            // che il numero di componenti è componentimassime.
            if (i == componentimassime) {
                clusterfinali = cluster2;
                break;
            }
            // Altrimenti passiamo a testare l'ipotesi nulla che G=i.
            logverosimiglianza1 = logverosimiglianza2;
            cluster1 = cluster2;
            pigreci1 = pigreci2;
            muascisse1 = muascisse2;
            muordinate1 = muordinate2;
            sigma111 = sigma112;
            sigma221 = sigma222;
            sigma121 = sigma122;
            nu1 = nu2;
            pesifinali1 = pesifinali2;
        }
        return clusterfinali;
    }
    if (selezione == 'D') {
        // Prepariamo i thread.
        int numerothread = static_cast<int>(std::thread::hardware_concurrency());
        if (numerothread == 0){numerothread = 4;}
        std::vector<std::thread> thread;
        thread.reserve(numerothread);
        std::vector<std::exception_ptr> thread_exceptions(numerothread);
        // Dividiamo le iterazioni come nel caso del bootstrap singolo.
        std::vector<int> iterazioniathread(numerothread, campionibootstrap/numerothread);
        for (int i = 0; i < campionibootstrap % numerothread; i++) {
            iterazioniathread[i]++;
        }
        // Prepariamo un vettore che conterrà la partizione migliore ottenuta.
        std::vector<int> clusterfinali(taglia, 0);
        // Cominciamo effettuando il clustering con due componenti, in modo da averne il valore della
        // logverosimiglianza e i parametri.
        double logverosimiglianza1 = 0.0;
        std::vector<int> cluster1(taglia, 0);
        std::vector<double> pigreci1(2);
        std::vector<double> muascisse1(2);
        std::vector<double> muordinate1(2);
        std::vector<double> sigma111(2);
        std::vector<double> sigma221(2);
        std::vector<double> sigma121(2);
        std::vector<double> nu1(2);
        std::vector<std::vector<double>> pesifinali1(taglia, std::vector<double>(2));
        veroclustering(normale, 2, iterazioniinizializzazione, taglia, inizializzazione, ascisse, ordinate,
            iterazionikmeans, algoritmo, iterazioniEM, logverosimiglianza1, cluster1, sogliaconvergenza,
            pigreci1, muascisse1, muordinate1, sigma111, sigma221, sigma121, nu1, pesifinali1);
        // Chiaramente, se componentimassime è 2, non c'è altro da fare: restituiamo la soluzione
        // appena ottenuta.
        if (componentimassime == 2){return cluster1;}
        // Per ogni successivo k possibile:
        for (int i = 3; i <= componentimassime; i++) {
            // Troviamo la logverosimiglianza, la partizione e i parametri sotto l'ipotesi alternativa.
            double logverosimiglianza2 = 0.0;
            std::vector<int> cluster2(taglia);
            std::vector<double> pigreci2(i);
            std::vector<double> muascisse2(i);
            std::vector<double> muordinate2(i);
            std::vector<double> sigma112(i);
            std::vector<double> sigma222(i);
            std::vector<double> sigma122(i);
            std::vector<double> nu2(i);
            std::vector<std::vector<double>> pesifinali2(taglia, std::vector<double>(i));
            veroclustering(normale, i, iterazioniinizializzazione, taglia, inizializzazione, ascisse, ordinate,
                iterazionikmeans, algoritmo, iterazioniEM, logverosimiglianza2, cluster2, sogliaconvergenza,
                pigreci2, muascisse2, muordinate2, sigma112, sigma222, sigma122, nu2, pesifinali2);
            // Adesso calcoliamo il valore di -2log\lambda per il campione originale.
            const double statisticaoriginale = -2.0*(logverosimiglianza1-logverosimiglianza2);
            // Per applicare una procedura di doppio bootstrap, dobbiamo ricavare una serie di statistiche
            // empiriche dal primo strato bootstrap, e per ognuna di esse un'altra serie dal secondo stato
            // bootstrap da usare come aggiustamento.
            std::vector<double> statisticheempiriche(campionibootstrap);
            std::vector<std::vector<double>> tutteinterne(campionibootstrap, std::vector<double>(bootstrapinterni));
            // Facciamo partire i thread che calcolano le statistiche bootstrap e aspettiamo finiscano.
            int inizio = 0;
            for (int j = 0; j < numerothread; j++) {
                int fine = inizio + iterazioniathread[j];
                thread.emplace_back([&, i, inizio, fine](){
                    try {
                        clusteringcondoppiobootstrap(inizio, fine, taglia, normale, i, std::ref(pigreci1),
                        std::ref(muascisse1), std::ref(muordinate1), std::ref(sigma111), std::ref(sigma221), std::ref(sigma121), std::ref(nu1),
                        iterazioniinizializzazione, inizializzazione, iterazionikmeans, algoritmo,
                        iterazioniEM, sogliaconvergenza, std::ref(statisticheempiriche), std::ref(tutteinterne), bootstrapinterni);
                    } catch(...) {
                        thread_exceptions[j] = std::current_exception();
                    }
                });
                inizio = fine;
            }
            int it = 0;
            // Aspettiamo che i thread finiscano.
            for (auto& t : thread) {
                try {
                    if (t.joinable()) {
                        t.join();
                    } else {
                        std::cout << "Thread numero " << it << " inattaccabile\n" << std::flush;
                    }
                } catch (const std::system_error& e) {
                    std::cerr << "System_error sollevato col thread " << it << ": " << e.what() << "\n" << std::flush;
                } catch (const std::exception& e) {
                    std::cerr << "Errore generico sollevato col thread " << it << ": " << e.what() << "\n" << std::flush;
                } catch (...) {
                    std::cerr << "Eccezione ignota col thread " << it << "\n" << std::flush;
                }
                it++;
            }
            thread.clear();
            for (int j = 0; j < numerothread; j++){
                if (thread_exceptions[j]){
                    try {
                        std::rethrow_exception(thread_exceptions[j]);
                    } catch (const std::exception& e) {
                        std::cerr << "Il worker " << j << " ha lanciato: " << e.what() << "\n";
                    } catch (...) {
                        std::cerr << "Il worker " << j << " ha lanciato un'eccezione fuori da std\n";
                    }
                }
            }
            // Usiamo le statistiche del secondo strato bootstrap per correggere quelle del primo.
            double mediastatisticheesterne = 0.0;
            for (int j = 0; j < campionibootstrap; j++) {
                mediastatisticheesterne += statisticheempiriche[j];
            }
            mediastatisticheesterne /= campionibootstrap;
            for (int j = 0; j < campionibootstrap; j++) {
                double mediainterna = 0.0;
                for (int k = 0; k < bootstrapinterni; k++) {
                    mediainterna += tutteinterne[j][k];
                }
                mediainterna /= bootstrapinterni;
                statisticheempiriche[j] -= mediainterna-mediastatisticheesterne;
            }
            // Adesso finalmente calcoliamo il quantile della distribuzione empirica di -2log\lambda
            // ed eseguiamo il test d'ipotesi.
            std::sort(statisticheempiriche.begin(), statisticheempiriche.end());
            const double posizione = (1.0-alfabootstrap)*(campionibootstrap-1);
            const int sotto = static_cast<int>(std::floor(posizione));
            const int sopra = static_cast<int>(std::ceil(posizione));
            const double peso = posizione-sotto;
            const double sogliacritica = (1.0-peso)*statisticheempiriche[sotto]+peso*statisticheempiriche[sopra];
            if (statisticaoriginale <= sogliacritica) {
                clusterfinali = cluster1;
                break;
            }
            logverosimiglianza1 = logverosimiglianza2;
            cluster1 = cluster2;
            pigreci1 = pigreci2;
            muascisse1 = muascisse2;
            muordinate1 = muordinate2;
            sigma111 = sigma112;
            sigma221 = sigma222;
            sigma121 = sigma122;
            nu1 = nu2;
            pesifinali1 = pesifinali2;
            if (i == componentimassime) {
                clusterfinali = cluster1;
            }
        }
        return clusterfinali;
    }
    if (selezione == 'V') {
        // Se è stato scelto un numero di fold troppo alto, maggiore delle unità nel campione,
        // lo cambiamo in modo da applicare di fatto una leave-one-out cross-validation.
        if (fold > taglia) {
            fold = taglia;
        }
        // Prepariamo i thread e assegniamo a ciascuno il numero di fold che esamineranno.
        int numerothread = static_cast<int>(std::thread::hardware_concurrency());
        if (numerothread == 0){numerothread = 4;}
        std::vector<std::thread> thread;
        thread.reserve(numerothread);
        std::vector<std::exception_ptr> thread_exceptions(numerothread);
        std::vector<int> iterazioniathread(numerothread, fold/numerothread);
        for (int i = 0; i < fold % numerothread; i++) {
            iterazioniathread[i]++;
        }
        // Dichiariamo il vettore che conterrà le stime della verosimiglianza cross-validate.
        std::vector<double> stimecrossvalidate(componentimassime-1);
        // Per ogni numero di componenti:
        for (int i = 2; i <= componentimassime; i++) {
            // Dichiariamo una lista dei log-predictive score per ogni fold, che poi medieremo.
            std::vector<double> listalogpredictivescore(fold);
            // Ora riempiamola.
            int inizio = 0;
            for (int j = 0; j < numerothread; j++) {
                int fine = inizio + iterazioniathread[j];
                thread.emplace_back([&, i, inizio, fine](){
                    try {
                        clusteringconcrossvalidation(inizio, fine, taglia, fold, i,
                        ascisse, ordinate, normale, iterazioniinizializzazione, inizializzazione, iterazionikmeans, algoritmo,
                        iterazioniEM, sogliaconvergenza, std::ref(listalogpredictivescore));
                    } catch(...) {
                        thread_exceptions[j] = std::current_exception();
                    }
                });
                inizio = fine;
            }
            int it = 0;
            // Aspettiamo che i thread finiscano.
            for (auto& t : thread) {
                try {
                    if (t.joinable()) {
                        t.join();
                    } else {
                        std::cout << "Thread numero " << it << " inattaccabile\n" << std::flush;
                    }
                } catch (const std::system_error& e) {
                    std::cerr << "System_error sollevato col thread " << it << ": " << e.what() << "\n" << std::flush;
                } catch (const std::exception& e) {
                    std::cerr << "Errore generico sollevato col thread " << it << ": " << e.what() << "\n" << std::flush;
                } catch (...) {
                    std::cerr << "Eccezione ignota col thread " << it << "\n" << std::flush;
                }
                it++;
            }
            thread.clear();
            for (int j = 0; j < numerothread; j++){
                if (thread_exceptions[j]){
                    try {
                        std::rethrow_exception(thread_exceptions[j]);
                    } catch (const std::exception& e) {
                        std::cerr << "Il worker " << j << " ha lanciato: " << e.what() << "\n";
                    } catch (...) {
                        std::cerr << "Il worker " << j << " ha lanciato un'eccezione fuori da std\n";
                    }
                }
            }
            double stimacrossvalidata = 0.0;
            for (int j = 0; j < fold; j++) {
                stimacrossvalidata += listalogpredictivescore[j];
            }
            stimacrossvalidata /= fold;
            stimecrossvalidate[i-2] = stimacrossvalidata;
        }
        // Troviamo il numero ottimale di componenti cercando quella con la stima cross-validata maggiore.
        int componenti = 2;
        double massimastima = stimecrossvalidate[0];
        for (int i = 1; i < componentimassime-1; i++) {
            if (stimecrossvalidate[i] > massimastima) {
                componenti = i+2;
                massimastima = stimecrossvalidate[i];
            }
        }
        // A questo punto possiamo semplicemente eseguire il clustering col numero di componenti trovato.
        double logverosimiglianzavera = 0.0;
        std::vector<int> clusterveri(taglia);
        std::vector<double> pigreciveri(componenti);
        std::vector<double> muascissevere(componenti);
        std::vector<double> muordinatevere(componenti);
        std::vector<double> sigma11vere(componenti);
        std::vector<double> sigma22vere(componenti);
        std::vector<double> sigma12vere(componenti);
        std::vector<double> nuveri(componenti);
        std::vector<std::vector<double>> pesiveri(taglia, std::vector<double>(componenti));
        veroclustering(normale, componenti, iterazioniinizializzazione, taglia, inizializzazione,
            ascisse, ordinate, iterazionikmeans, algoritmo, iterazioniEM, logverosimiglianzavera,
            clusterveri, sogliaconvergenza, pigreciveri, muascissevere, muordinatevere,
            sigma11vere, sigma22vere, sigma12vere, nuveri, pesiveri);
        return clusterveri;
    }
    return std::vector<int>(taglia, 0);
}


// Funziona che esegue effettivamente i calcoli per stimare con metodo Monte Carlo la probabilità
// di errore di II tipo di un t-test univariato sulle norme.
int calcolabetatestnorme(const int taglia, const bool maggiore, const bool disuguale, const int iterazioni,
    const double media, const double distanza, const double varianza, const double soglia) {
    // Inizializziamo il contatore di quante volte l'ipotesi nulla viene accettata.
    int accettazioni = 0;
    // Prepariamo il generatore di numeri casuali.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    // Se l'ipotesi alternativa è H_1:\mu>\mu_0:
    if (maggiore) {
        std::normal_distribution<> normale(media+distanza, std::sqrt(varianza));
        for (int i = 0; i < iterazioni; i++) {
            // Estraiamo un campione sotto l'ipotesi alternativa e contemporaneamente calcoliamone la media.
            std::vector<double> campione(taglia);
            double mediacampione = 0.0;
            for (int j = 0; j < taglia; j++) {
                const double elemento = normale(generatore);
                campione[j] = elemento;
                mediacampione += elemento;
            }
            mediacampione /= taglia;
            // Ora calcoliamone la varianza.
            double varianzacampione = 0.0;
            for (int j = 0; j < taglia; j++) {
                const double differenza = campione[j]-mediacampione;
                varianzacampione += differenza*differenza;
            }
            varianzacampione /= taglia-1;
            // Eseguiamo il classico t-test.
            if (varianzacampione < 0.000001 && mediacampione == media){accettazioni++; continue;}
            if (varianzacampione < 0.000001 && mediacampione != media){continue;}
            const double statisticatest = (mediacampione-media)/std::sqrt(varianzacampione/taglia);
            if (statisticatest < soglia) {
                accettazioni++;
            }
        }
        // Facciamo una cosa simile se l'ipotesi alternativa è H_1:\mu\neq\mu_0.
    } else if (disuguale) {
        std::normal_distribution<> normale(media+distanza, std::sqrt(varianza));
        for (int i = 0; i < iterazioni; i++) {
            std::vector<double> campione(taglia);
            double mediacampione = 0.0;
            for (int j = 0; j < taglia; j++) {
                const double elemento = normale(generatore);
                campione[j] = elemento;
                mediacampione += elemento;
            }
            mediacampione /= taglia;
            double varianzacampione = 0.0;
            for (int j = 0; j < taglia; j++) {
                const double differenza = campione[j]-mediacampione;
                varianzacampione += differenza*differenza;
            }
            varianzacampione /= taglia-1;
            if (varianzacampione < 0.000001 && mediacampione == media){accettazioni++; continue;}
            if (varianzacampione < 0.000001 && mediacampione != media){continue;}
            const double statisticatest = (mediacampione-media)/std::sqrt(varianzacampione/taglia);
            if (std::abs(statisticatest) < soglia) {
                accettazioni++;
            }
        }
        // Stessa cosa per H_1:\mu<\mu_0.
    } else {
        std::normal_distribution<> normale(media-distanza, std::sqrt(varianza));
        for (int i = 0; i < iterazioni; i++) {
            std::vector<double> campione(taglia);
            double mediacampione = 0.0;
            for (int j = 0; j < taglia; j++) {
                const double elemento = normale(generatore);
                campione[j] = elemento;
                mediacampione += elemento;
            }
            mediacampione /= taglia;
            double varianzacampione = 0.0;
            for (int j = 0; j < taglia; j++) {
                const double differenza = campione[j]-mediacampione;
                varianzacampione += differenza*differenza;
            }
            varianzacampione /= taglia-1;
            if (varianzacampione < 0.000001 && mediacampione == media){accettazioni++; continue;}
            if (varianzacampione < 0.000001 && mediacampione != media){continue;}
            const double statisticatest = (mediacampione-media)/std::sqrt(varianzacampione/taglia);
            if (statisticatest > soglia) {
                accettazioni++;
            }
        }
    }
    return accettazioni;
}


// Funzione preparatoria per calcolare la probabilità di errore di II tipo di un semplice t-test
// univariato. I parametri sono:
// Taglia: numerosità del campione.
// Maggiore: "true" se l'ipotesi alternativa ha il segno >, altrimenti "false".
// Disuguale: stessa cosa per il segno \neq.
// Iterazioni: numero di iterazioni Monte Carlo.
// Media: valore di \mu_0 nell'ipotesi nulla.
// Distanza: quando vengono estratti campioni sotto l'ipotesi alternativa, la media della distribuzione
// sarà mu_0+distanza se il segno è > o \neq, altrimenti sarà \mu_0-distanza.
// Varianza: varianza che avranno i campioni Monte Carlo.
// Soglia: valore critico oltre cui si rifiuta l'ipotesi nulla, da calcolare al di fuori della funzione.
double betatestnorme(const int taglia, const bool maggiore, const bool disuguale, const int iterazioni,
    const double media, const double distanza, const double varianza, const double soglia) {
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (iterazioni < 1){throw std::runtime_error("Le iterazioni devono essere positive");}
    // Prepariamo i thread come al solito.
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    std::vector<int> accettazionithread(numerothread);
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcolabetatestnorme(taglia, maggiore, disuguale, volteperthread, media,
                distanza, varianza, soglia);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazionithread]() {
        accettazionithread[numerothread-1] =  calcolabetatestnorme(taglia, maggiore, disuguale, volteperthread+iterazionirestanti, media,
                distanza, varianza, soglia);
    });
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return static_cast<double>(accettazioni)/iterazioni;
}


// Funzione che esegue effettivamente i calcoli per betatestvarianze.
// Il suo funzionamento è analogo a calcolabetatestnorme.
int calcolabetatestvarianze(const int taglia, const bool maggiore, const bool disuguale,
    const int iterazioni, const double distanza, const double soglia, const double secondasoglia,
    const double ipotesinulla){
    int accettazioni = 0;
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    if (maggiore) {
        std::normal_distribution<> normale(0.0, std::sqrt(ipotesinulla+distanza));
        for (int i = 0; i < iterazioni; i++) {
            std::vector<double> campione(taglia);
            double mediacampione = 0.0;
            for (int j = 0; j < taglia; j++) {
                const double elemento = normale(generatore);
                campione[j] = elemento;
                mediacampione += elemento;
            }
            mediacampione /= taglia;
            double varianzacampione = 0.0;
            for (int j = 0; j < taglia; j++) {
                const double differenza = campione[j]-mediacampione;
                varianzacampione += differenza*differenza;
            }
            varianzacampione /= taglia-1;
            const double statisticatest = (taglia-1)*varianzacampione/ipotesinulla;
            if (statisticatest < soglia){
                accettazioni++;
            }
        }
    } else if (disuguale) {
        std::normal_distribution<> normale(0.0, std::sqrt(ipotesinulla+distanza));
        for (int i = 0; i < iterazioni; i++) {
            std::vector<double> campione(taglia);
            double mediacampione = 0.0;
            for (int j = 0; j < taglia; j++) {
                const double elemento = normale(generatore);
                campione[j] = elemento;
                mediacampione += elemento;
            }
            mediacampione /= taglia;
            double varianzacampione = 0.0;
            for (int j = 0; j < taglia; j++) {
                const double differenza = campione[j]-mediacampione;
                varianzacampione += differenza*differenza;
            }
            varianzacampione /= taglia-1;
            const double statisticatest = (taglia-1)*varianzacampione/ipotesinulla;
            if (soglia < statisticatest && statisticatest < secondasoglia) {
                accettazioni++;
            }
        }
    } else {
        std::normal_distribution<> normale(0.0, std::sqrt(ipotesinulla-distanza));
        for (int i = 0; i < iterazioni; i++) {
            std::vector<double> campione(taglia);
            double mediacampione = 0.0;
            for (int j = 0; j < taglia; j++) {
                const double elemento = normale(generatore);
                campione[j] = elemento;
                mediacampione += elemento;
            }
            mediacampione /= taglia;
            double varianzacampione = 0.0;
            for (int j = 0; j < taglia; j++) {
                const double differenza = campione[j]-mediacampione;
                varianzacampione += differenza*differenza;
            }
            varianzacampione /= taglia-1;
            const double statisticatest = (taglia-1)*varianzacampione/ipotesinulla;
            if (statisticatest > soglia) {
                accettazioni++;
            }
        }
    }
    return accettazioni;
}


// Funzione preparatoria per calcolare la probabilità di errore di II tipo del test d'ipotesi sulla
// varianza della popolazione. I parametri sono:
// Iterazioni: numero di iterazioni Monte Carlo.
// Taglia: numerosità del campione.
// Maggiore: "true" se l'ipotesi alternativa è H_1:\sigma^2>\sigma^2_0 e "false" altrimenti.
// Disuguale: stessa cosa per il segno \neq.
// Distanza: se maggiore o disuguale sono uguali, la varianza della popolazione sotto l'ipotesi alternativa
// nei campioni Monte Carlo è posta pari a ipotesinulla+distanza, altrimenti a ipotesinulla-distanza.
// Soglia: valore critico oltre cui si rifiuta l'ipotesi nulla nel caso di ipotesi alternativa unilaterale;
// se l'ipotesi alternativa è bilaterale, allora questo parametro è l'estremo sinistro dell'intervallo
// per la statistica test in cui si accetta l'ipotesi nulla. Allo stesso modo, seconda soglia è l'estremo destro.
// Ipotesinulla: valore di \sigma^2_0.
double betatestvarianze(const int iterazioni, const int taglia, const bool maggiore, const bool disuguale,
    const double distanza, const double soglia, const double secondasoglia, const double ipotesinulla) {
    if (iterazioni < 1){throw std::runtime_error("Le iterazioni devono essere positive");}
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (ipotesinulla <= 0.0){throw std::runtime_error("La varianza sotto l'ipotesi nulla deve essere positiva");}
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    std::vector<int> accettazionithread(numerothread);
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcolabetatestvarianze(taglia, maggiore, disuguale, volteperthread,
                distanza, soglia, secondasoglia, ipotesinulla);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazionithread]() {
        accettazionithread[numerothread-1] = calcolabetatestvarianze(taglia, maggiore, disuguale, volteperthread+iterazionirestanti,
                distanza, soglia, secondasoglia, ipotesinulla);
    });
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return static_cast<double>(accettazioni)/iterazioni;
}


// Funzione che prende un dataset di tiri di frecce con etichette corrispondenti a ogni freccia specifica
// e determina per ognuna se è difettosa (assumendo che tutte le altre non in esame siano sane) applicando
// un test di Hotelling a due campioni. I parametri:
// Ascisse e ordinate: due vettori che contengono il dataset.
// Frecce: vettore che contiene gli indici corrispondenti alle frecce per ogni osservazione.
// Taglia: numerosità del campione.
// Soglia: valore critico oltre cui si rifiuta l'ipotesi nulla.
// Numerofrecce: il numero di frecce fisicamente distinte tirate nell'allenamento del dataset.
std::vector<CoseTest> freccedifettose(const std::vector<double> &ascisse, const std::vector<double> &ordinate,
    const std::vector<int> &frecce, const int taglia, const double alfa, const int numerofrecce) {
    if (ascisse.size() != taglia || ordinate.size() != taglia || frecce.size() != taglia){throw std::runtime_error("Il dataset non corrisponde alla taglia");}
    if (numerofrecce < 2){throw std::runtime_error("Devono esserci almeno due frecce distinte");}
    // Dichiariamo il vettore di risultati dei test.
    std::vector<CoseTest> risultati(numerofrecce);
    fisher_f_distribution<> distribuzione(2, taglia-2);
    // Per ognuno degli indici che contrassegnano le frecce:
    for (int i = 0; i < numerofrecce; i++) {
        // Separiamo nel campione le osservazioni corrispondenti alla freccia di cui vogliamo
        // verificare la difettosità e tutte le altre; calcoliamo i vettori medi di questi due sottocampioni.
        double mediaascissedifettande = 0.0;
        double mediaordinatedifettande = 0.0;
        double mediaascissealtre = 0.0;
        double mediaordinatealtre = 0.0;
        int numerodifettande = 0;
        for (int j = 0; j < taglia; j++) {
            if (frecce[j] == i) {
                mediaascissedifettande += ascisse[j];
                mediaordinatedifettande += ordinate[j];
                numerodifettande++;
            } else {
                mediaascissealtre += ascisse[j];
                mediaordinatealtre += ordinate[j];
            }
        }
        const int numeroaltre = taglia-numerodifettande;
        if (numerodifettande == 0){throw std::runtime_error("C'è una freccia che non esiste");}
        if (numeroaltre == 0){throw std::runtime_error("C'è una freccia che non esiste");}
        mediaascissedifettande /= numerodifettande;
        mediaordinatedifettande /= numerodifettande;
        mediaascissealtre /= numeroaltre;
        mediaordinatealtre /= numeroaltre;
        // Adesso, sempre per i due sottocampioni, calcoliamo le matrici di covarianze.
        double varianzaascissedifettande = 0.0;
        double varianzaordinatedifettande = 0.0;
        double covarianzadifettande = 0.0;
        double varianzaascissealtre = 0.0;
        double varianzaordinatealtre = 0.0;
        double covarianzaaltre = 0.0;
        for (int j = 0; j < taglia; j++) {
            if (frecce[j] == i) {
                const double differenzaascisse = ascisse[j]-mediaascissedifettande;
                const double differenzaordinate = ordinate[j]-mediaordinatedifettande;
                varianzaascissedifettande += differenzaascisse*differenzaascisse;
                varianzaordinatedifettande += differenzaordinate*differenzaordinate;
                covarianzadifettande += differenzaascisse*differenzaordinate;
            } else {
                const double differenzaascisse = ascisse[j]-mediaascissealtre;
                const double differenzaordinate = ordinate[j]-mediaordinatealtre;
                varianzaascissealtre += differenzaascisse*differenzaascisse;
                varianzaordinatealtre += differenzaordinate*differenzaordinate;
                covarianzaaltre += differenzaascisse*differenzaordinate;
            }
        }
        if (numerodifettande == 1){risultati[i] = false; continue;}
        if (numerodifettande == 1){risultati[i] = false; continue;}
        varianzaascissedifettande /= numerodifettande-1;
        varianzaordinatedifettande /= numerodifettande-1;
        covarianzadifettande /= numerodifettande-1;
        varianzaascissealtre /= numeroaltre-1;
        varianzaordinatealtre /= numeroaltre-1;
        covarianzaaltre /= numeroaltre-1;
        // Adesso calcoliamo la statistica test per un test di Hotelling a due campioni.
        const int n1 = numerodifettande-1;
        const int n2 = numeroaltre-1;
        const double denominatore = n1+n2;
        double varianzaascissepooled = (n1*varianzaascissedifettande+n2*varianzaascissealtre)/denominatore;
        double varianzaordinatepooled = (n1*varianzaordinatedifettande+n2*varianzaordinatealtre)/denominatore;
        double covarianzapooled = (n1*covarianzadifettande+n2*covarianzaaltre)/denominatore;
        if (std::abs(covarianzapooled) > std::sqrt(varianzaascissepooled*varianzaordinatepooled)){covarianzapooled = std::sqrt(varianzaascissepooled*varianzaordinatepooled);}
        double determinantepooled = varianzaascissepooled*varianzaordinatepooled-covarianzapooled*covarianzapooled;
        if (determinantepooled < 0.000001){
            varianzaascissepooled += 0.001;
            varianzaordinatepooled += 0.001;
            determinantepooled = varianzaascissepooled*varianzaordinatepooled-covarianzapooled*covarianzapooled;
        }
        const double inversapooled11 = varianzaordinatepooled/determinantepooled;
        const double inversapooled22 = varianzaascissepooled/determinantepooled;
        const double inversapooled12 = -covarianzapooled/determinantepooled;
        const double differenzamedieascisse = mediaascissedifettande-mediaascissealtre;
        const double differenzamedieordinate = mediaordinatedifettande-mediaordinatealtre;
        const double formaquadratica = inversapooled11*differenzamedieascisse*differenzamedieascisse+
            inversapooled22*differenzamedieordinate*differenzamedieordinate+2.0*inversapooled12*
                differenzamedieordinate*differenzamedieordinate;
        const double statisticagrezza = (n1+1)*(n2+1)/static_cast<double>(n1+n2+2)*formaquadratica;
        const double statisticatest = 0.5*(n1+n2-1)/static_cast<double>(n1+n2)*statisticagrezza;
        // Se rifiutiamo l'ipotesi nulla, ossia superiamo il valore critico, che le medie dei due gruppi
        // siano uguali, possiamo concludere che la freccia sia difettosa (dunque "true").
        risultati[i].statistica = statisticatest;
        risultati[i].pvalue = 1.0-cdf(distribuzione, statisticatest);
        risultati[i].accettazione = risultati[i].pvalue >= alfa;
    }
    return risultati;
}


// Funzione che esegue i calcoli per betahotellingduecampioni.
int calcolabetahotellingduecampioni(const int iterazioni, const double varianzaascisse, const double varianzaordinate,
    const double covarianza, const int taglia, const double distanza, const int numerofrecce, const double soglia) {
    // Inizializziamo il contatore di volte in cui l'ipotesi nulla è accettata e prepariamo
    // il generatore di numeri casuali.
    int accettazioni = 0;
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normale(0.0, 1.0);
    // Calcoliamo la decomposizione di Cholesky della matrice di covarianze.
    const double cholesky11 = std::sqrt(varianzaascisse);
    const double cholesky12 = covarianza/cholesky11;
    if (varianzaordinate-cholesky12*cholesky12 <= 0.0){throw std::runtime_error("Co/varianze specificate non valide");}
    const double cholesky22 = std::sqrt(varianzaordinate-cholesky12*cholesky12);
    // Facciamo l'assunzione che ogni freccia distinta venga tirata un numero uguale di volte.
    const int difettande = taglia/numerofrecce;
    const int altre = taglia-difettande;
    // Per ogni iterazione:
    for (int i = 0; i < iterazioni; i++) {
        // Estraiamo un campione da una normale a media nulla e con la matrice di covarianze specificata;
        // contemporaneamente calcoliamone il vettore medio.
        std::vector<double> ascissedifettande(difettande);
        std::vector<double> ordinatedifettande(difettande);
        double mediaascissedifettande = 0.0;
        double mediaordinatedifettande = 0.0;
        for (int j = 0; j < difettande; j++) {
            const double ascissa = normale(generatore);
            const double ordinata = normale(generatore);
            mediaascissedifettande += cholesky11*ascissa;
            mediaordinatedifettande += cholesky12*ascissa+cholesky22*ordinata;
            ascissedifettande[j] = cholesky11*ascissa;
            ordinatedifettande[j] = cholesky12*ascissa+cholesky22*ordinata;
        }
        // Ed estraiamone un altro più grande con una media diversa nelle ascisse
        // ma la stessa matrice di covarianze.
        std::vector<double> ascissealtre(altre);
        std::vector<double> ordinatealtre(altre);
        double mediaascissealtre = 0.0;
        double mediaordinatealtre = 0.0;
        for (int j = 0; j < altre; j++){
            const double ascissa = normale(generatore);
            const double ordinata = normale(generatore);
            mediaascissealtre += distanza+cholesky11*ascissa;
            mediaordinatealtre += cholesky12*ascissa+cholesky22*ordinata;
            ascissealtre[j] = distanza+cholesky11*ascissa;
            ordinatealtre[j] = cholesky12*ascissa+cholesky22*ordinata;
        }
        mediaascissedifettande /= taglia;
        mediaordinatedifettande /= taglia;
        mediaascissealtre /= taglia;
        mediaordinatealtre /= taglia;
        // Ora calcoliamo le varianze e le covarianze.
        double varianzaascissedifettande = 0.0;
        double varianzaordinatedifettande = 0.0;
        double covarianzadifettande = 0.0;
        for (int j = 0; j < difettande; j++) {
            const double differenzaascisse = ascissedifettande[j]-mediaascissedifettande;
            const double differenzaordinate = ordinatedifettande[j]-mediaordinatedifettande;
            varianzaascissedifettande += differenzaascisse*differenzaascisse;
            varianzaordinatedifettande += differenzaordinate*differenzaordinate;
            covarianzadifettande += differenzaascisse*differenzaordinate;
        }
        varianzaascissedifettande /= difettande-1;
        varianzaordinatedifettande /= difettande-1;
        covarianzadifettande /= difettande-1;
        double varianzaascissealtre = 0.0;
        double varianzaordinatealtre = 0.0;
        double covarianzaaltre = 0.0;
        for (int j = 0; j < altre; j++){
            const double differenzaascisse = ascissealtre[j]-mediaascissealtre;
            const double differenzaordinate = ordinatealtre[j]-mediaordinatealtre;
            varianzaascissealtre += differenzaascisse*differenzaascisse;
            varianzaordinatealtre += differenzaordinate*differenzaordinate;
            covarianzaaltre += differenzaascisse*differenzaordinate;
        }
        varianzaascissealtre /= altre-1;
        varianzaordinatealtre /= altre-1;
        covarianzaaltre /= altre-1;
        // Ora calcoliamo la statistica test e registriamo se l'ipotesi nulla viene accettata.
        const int n1 = difettande-1;
        const int n2 = altre-1;
        const double denominatore = n1+n2;
        double varianzaascissepooled = (n1*varianzaascissedifettande+n2*varianzaascissealtre)/denominatore;
        double varianzaordinatepooled = (n1*varianzaordinatedifettande+n2*varianzaordinatealtre)/denominatore;
        double covarianzapooled = (n1*covarianzadifettande+n2*covarianzaaltre)/denominatore;
        if (std::abs(covarianzapooled) > std::sqrt(varianzaascissepooled*varianzaordinatepooled)){covarianzapooled = std::sqrt(varianzaascissepooled*varianzaordinatepooled);}
        double determinantepooled = varianzaascissepooled*varianzaordinatepooled-covarianzapooled*covarianzapooled;
        if (determinantepooled < 0.000001){
            varianzaascissepooled += 0.001;
            varianzaordinatepooled += 0.001;
            determinantepooled = varianzaascissepooled*varianzaordinatepooled-covarianzapooled*covarianzapooled;
        }
        const double inversapooled11 = varianzaordinatepooled/determinantepooled;
        const double inversapooled22 = varianzaascissepooled/determinantepooled;
        const double inversapooled12 = -covarianzapooled/determinantepooled;
        const double differenzamedieascisse = mediaascissedifettande-mediaascissealtre;
        const double differenzamedieordinate = mediaordinatedifettande-mediaordinatealtre;
        const double formaquadratica = inversapooled11*differenzamedieascisse*differenzamedieascisse+
            inversapooled22*differenzamedieordinate*differenzamedieordinate+2.0*inversapooled12*
                differenzamedieordinate*differenzamedieordinate;
        const double statisticagrezza = (n1+1)*(n2+1)/static_cast<double>(n1+n2+2)*formaquadratica;
        const double statisticatest = 0.5*(n1+n2-1)/static_cast<double>(n1+n2)*statisticagrezza;
        if (statisticatest < soglia) {
            accettazioni++;
        }
    }
    return accettazioni;
}


// Funzione preparatoria per calcolare la probabilità di errore di II tipo del test di Hotelling
// a due campioni. I parametri sono:
// Iterazioni: numero di iterazioni Monte Carlo.
// Varianzaascisse, varianzaordinate e covarianza: varianze delle due variabili e covarianza che devono
// avere i campioni generati.
// Taglia: numerosità dei campioni.
// Distanza: i due campioni generati nel metodo Monte Carlo avranno medie diverse; uno avrà media nulla
// e l'altro media delle ascisse pari a distanza e media delle ordinate nulla.
// Numerofrecce: numero distinto di frecce che si assumono tirate; determina la grandezza dei campioni,
// poiché quello a media nulla avrà taglia/numerofrecce osservazioni e l'altro taglia-taglia/numerofrecce.
// Soglia: valore critico oltre cui si rifiuta l'ipotesi nulla, da trovare fuori dalla funzione.
double betahotellingduecampioni(const int iterazioni, const double varianzaascisse, const double varianzaordinate,
    const double covarianza, const int taglia, const double distanza, const int numerofrecce, const double soglia) {
    if (iterazioni < 1 || varianzaascisse <= 0.0 || varianzaordinate <= 0.0){throw std::runtime_error("Errore!");}
    if (varianzaascisse*varianzaordinate-covarianza*covarianza <= 0.0 || taglia < 3 || numerofrecce < 2 || taglia/numerofrecce < 2){throw std::runtime_error("Errore!");}
    // Praticamente identica a tutte le altre funzioni che fanno metodi Monte Carlo abbiate pietà
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    std::vector<int> accettazionithread(numerothread);
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcolabetahotellingduecampioni(volteperthread, varianzaascisse, varianzaordinate, covarianza,
                taglia, distanza, numerofrecce, soglia);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazionithread]() {
        accettazionithread[numerothread-1] = calcolabetahotellingduecampioni(volteperthread+iterazionirestanti, varianzaascisse, varianzaordinate,
            covarianza, taglia, distanza, numerofrecce, soglia);
    });
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return static_cast<double>(accettazioni)/iterazioni;
}


// Funzione che esegue effettivamente betavarianzeduecampioni.
int calcolabetavarianzeduecampioni(const int iterazioni, const double varianza, const int taglia,
    const int numerofrecce, const double soglia) {
    // Prepariamo i generatori di numeri casuali.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normale1(0.0, std::sqrt(varianza));
    std::normal_distribution<> normale2(0.0, 1.0);
    int accettazioni = 0;
    // Calcoliamo le taglie dei due campioni.
    const int prime = taglia/numerofrecce;
    const int seconde = taglia-taglia/numerofrecce;
    // Per ogni iterazione:
    for (int i = 0; i < iterazioni; i++) {
        // Estraiamo un primo campione piccolo da una normale standard, calcolandone contemporaneamente
        // la media.
        std::vector<double> campione1(taglia);
        double media1 = 0.0;
        for (int j = 0; j < prime; j++) {
            const double elemento1 = normale1(generatore);
            campione1[j] = elemento1;
            media1 += elemento1;
        }
        media1 /= prime;
        // Facciamo lo stesso per un secondo campione più grande da una normale a media nulla e con la
        // varianza specificata.
        std::vector<double> campione2(taglia);
        double media2 = 0.0;
        for (int j = 0; j < seconde; j++){
            const double elemento2 = normale2(generatore);
            campione2[j] = elemento2;
            media2 += elemento2;
        }
        media2 /= seconde;
        // Ora calcoliamo le varianze.
        double varianza1 = 0.0;
        for (int j = 0; j < prime; j++) {
            const double differenza = campione1[j]-media1;
            varianza1 += differenza*differenza;
        }
        varianza1 /= prime-1;
        double varianza2 = 0.0;
        for (int j = 0; j < seconde; j++){
            const double differenza = campione2[j]-media2;
            varianza2 += differenza*differenza;
        }
        varianza2 /= seconde-1;
        // Ora eseguiamo il test.
        if (varianza1/varianza2 < soglia){accettazioni++;}
    }
    return accettazioni;
}


// Funzione preparatoria per calcolare la probabilità di errore di II tipo del test per vedere
// se due campioni hanno la stessa varianza. I parametri:
// Iterazioni: numero di iterazioni bootstrap.
// Varianza: a ogni iterazione Monte Carlo, un campione è estratto da una normale standard e l'altro
// da una normale a media nulla con varianza pari a "varianza".
// Taglia: n_1+n_2.
// Numerofrecce: numero di frecce distinte che si suppongono essere state tirate; determinano
// la dimensione relativa dei due campioni.
// Soglia: valore critico oltre cui si rifiuta l'ipotesi nulla.
double betavarianzeduecampioni(const int iterazioni, const double varianza, const int taglia, const int numerofrecce,
    const double soglia) {
    if (iterazioni < 1){throw std::runtime_error("Le iterazioni devono essere positive");}
    if (varianza <= 0.0){throw std::runtime_error("Varianza specificata non valida");}
    if (numerofrecce < 2){throw std::runtime_error("Devono esserci almeno due frecce distinte");}
    if (taglia/numerofrecce < 2){return 0.0;}
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    std::vector<int> accettazionithread(numerothread);
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcolabetavarianzeduecampioni(volteperthread, varianza,
                taglia, numerofrecce, soglia);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazionithread]() {
        accettazionithread[numerothread-1] = calcolabetavarianzeduecampioni(volteperthread+iterazionirestanti, varianza,
            taglia, numerofrecce, soglia);
    });
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return static_cast<double>(accettazioni)/iterazioni;
}


// Funzione che implementa betarayleigh.
int calcolabetarayleigh(const int iterazioni, const double concentrazione, const int taglia, const double soglia) {
    // Prepariamo i generatori di numeri casuali:
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::uniform_real_distribution<> uniforme(0.0, 1.0);
    int accettazioni = 0;
    // Calcoliamo il parametro r da \kappa.
    const double tau = 1.0+std::sqrt(1.0+concentrazione*concentrazione);
    const double rho = (tau-std::sqrt(2.0*tau))/(2.0*concentrazione);
    const double r = (1.0+rho*rho)/(2.0*rho);
    // Per ogni iterazione:
    for (int i = 0; i < iterazioni; i++) {
        // Estraiamo un campione da una von Mises con \mu=0 e il \kappa specificato.
        // Contemporaneamente, calcoleremo \bar{C} e \bar{S}.
        std::vector<double> campione(taglia);
        int indice = 0;
        double sommacoseni = 0.0;
        double sommaseni = 0.0;
        while (indice < taglia) {
            const double u1 = uniforme(generatore);
            const double z = std::cos(pigreco*u1);
            if (r+z == 0.0){continue;}
            const double f = (1.0+r*z)/(r+z);
            const double c = concentrazione*(r-f);
            if (c <= 0.0){continue;}
            const double u2 = uniforme(generatore);
            if (c*(2.0-c)-u2 > 0.0) {
                const double u3 = uniforme(generatore);
                sommacoseni += f;
                sommaseni += (u3 > 0.5 ? +1 : -1) * std::sqrt(1.0 - f*f);
                indice++;
            } else if (std::log(c/u2)+1.0-c >= 0.0) {
                const double u3 = uniforme(generatore);
                sommacoseni += f;
                sommaseni += (u3 > 0.5 ? +1 : -1) * std::sqrt(1.0 - f*f);
                indice++;
            }
        }
        sommacoseni /= taglia;
        sommaseni /= taglia;
        // Calcoliamo la statistica di Rayleigh ed eseguiamo il test.
        if (2.0*taglia*std::sqrt(sommacoseni*sommacoseni+sommaseni*sommaseni) <= soglia){accettazioni++;}
    }
    return accettazioni;
}


// Funzione per calcolare la probabilità di errore di II tipo del test di Rayleigh. I parametri sono:
// Iterazioni: numero di iterazioni Monte Carlo.
// Concentrazione: parametro di concetrazione dei campioni Monte Carlo sotto l'ipotesi alternativa,
// che saranno estratti da una Von Mises.
// Taglia: numerosità dei campioni Monte Carlo.
// Soglia: valore critico oltre cui si rifiuta l'ipotesi nulla; va calcolato fuori dalla funzione.
double betarayleigh(const int iterazioni, const double concentrazione, const int taglia, const double soglia) {
    if (iterazioni < 1){throw std::runtime_error("Le iterazioni devono essere positive");}
    if (concentrazione <= 0.0){throw std::runtime_error("Parametro di concentrazione non valido");}
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    std::vector<int> accettazionithread(numerothread);
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcolabetarayleigh(volteperthread, concentrazione, taglia, soglia);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazionithread]() {
        accettazionithread[numerothread-1] = calcolabetarayleigh(volteperthread+iterazionirestanti, concentrazione, taglia, soglia);
    });
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return static_cast<double>(accettazioni)/iterazioni;
}


// Esegue il test di goodness of fit della von Mises. Gli argomenti sono:
// Angoli: vettore con i dati.
// Taglia: numerosità del campione.
// Angolomedio: angolo medio.
// Kappa: stima di massima verosimiglianza del parametro di concentrazione dai dati.
// Soglia: valore critico per rifiutare l'ipotesi nulla, ossia il quantile 1-alfa di una chi quadro
// con 2 gradi di libertà.
bool testvonmises(const std::vector<double> &angoli, const int taglia, const double angolomedio,
    const double kappa, const double soglia) {
    if (angoli.size() != taglia){throw std::runtime_error("Il dataset non corrisponde alla taglia");}
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (kappa <= 0.0){throw std::runtime_error("Parametro di concentrazione non valido");}
    // Calcoliamo subito le funzioni di Bessel che ci serviranno più avanti.
    const double i0 = std::cyl_bessel_i(0.0, kappa);
    const double i1 = std::cyl_bessel_i(1.0, kappa);
    const double i2 = std::cyl_bessel_i(2.0, kappa);
    const double i3 = std::cyl_bessel_i(3.0, kappa);
    const double i4 = std::cyl_bessel_i(4.0, kappa);
    // Ora calcoliamo le statistiche s_c e s_s.
    double sc = 0.0;
    double ss = 0.0;
    for (int i = 0; i < taglia; i++) {
        sc += std::cos(2.0*(angoli[i]-angolomedio));
        ss += std::sin(2.0*(angoli[i]-angolomedio));
    }
    sc -= taglia*i1/i0;
    // E adesso passiamo a v_c e v_s.
    const double numeratorevc = i0*i3+i0*i1-2.0*i1*i2;
    const double pezzonumeratorevs = i0-i3;
    const double vc = (i0*i0+i0*i4-2.0*i2*i2)/(2.0*i0*i0)-numeratorevc*numeratorevc/(2.0*i0*i0*(i0*i0+i0*i1-2.0*i1*i1));
    const double vs = ((i0-i4)*(i0-i2)-pezzonumeratorevs*pezzonumeratorevs)/(2.0*i0*(i0-i2));
    // Ora possiamo eseguire il test.
    std::cout << sc << " " << ss << " " << vc << " " << vs << "\n" << std::flush;
    return sc*sc/vc+ss*ss/vs < soglia;
}


// Funzione che esegue i calcoli per alfaverointervalloangolomedio.
int calcolaalfaverointervalloangolomedio(const int iterazioni, const double kappa, const int taglia,
    const bool facile, const double quantile) {
    // Prepariamo i generatori di numeri casuali, il contatore delle volte in cui l'intervallo di
    // confidenza ottenuto è corretto e alcune costanti utili alla generazione del campione.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::uniform_real_distribution<> uniforme(0.0, 1.0);
    int accettazioni = 0;
    const double tau = 1.0+std::sqrt(1.0+kappa*kappa);
    const double rho = (tau-std::sqrt(2.0*tau))/(2.0*kappa);
    const double r = (1.0+rho*rho)/(2.0*rho);
    // Per ogni iterazione Monte Carlo:
    for (int i = 0; i < iterazioni; i++) {
        // Estraiamo un campione da una von Mises con angolo medio 0; per accelerare, calcoleremo contemporaneamente
        // all'estrazione le somme di coseni e seni, senza memorizzarlo.
        double sommacoseni = 0.0;
        double sommaseni = 0.0;
        int indice = 0;
        while (indice < taglia) {
            const double u1 = uniforme(generatore);
            const double z = std::cos(pigreco*u1);
            if (r+z == 0.0){continue;}
            const double f = (1.0+r*z)/(r+z);
            const double c = kappa*(r-f);
            if (c <= 0.0){continue;}
            const double u2 = uniforme(generatore);
            if (c*(2.0-c)-u2 > 0.0) {
                const double u3 = uniforme(generatore);
                sommacoseni += f;
                sommaseni   += (u3 > 0.5 ? +1 : -1) * std::sqrt(1.0 - f*f);
                indice++;
            } else if (std::log(c/u2)+1.0-c >= 0.0) {
                const double u3 = uniforme(generatore);
                sommacoseni += f;
                sommaseni   += (u3 > 0.5 ? +1 : -1) * std::sqrt(1.0 - f*f);
                indice++;
            }
        }
        sommacoseni /= taglia;
        sommaseni /= taglia;
        // Calcoliamo ora \hat{\theta}, \bar{R} e \hat{\kappa}.
        const double angolomedio = std::atan2(sommaseni, sommacoseni);
        const double risultante = std::sqrt(sommacoseni*sommacoseni+sommaseni*sommaseni);
        double concentrazione;
        if (risultante < 0.53){concentrazione = 2.0*risultante+risultante*risultante*risultante+5.0*risultante*risultante*
            risultante*risultante*risultante/6.0;}
        else if (0.53 <= risultante && risultante < 0.85){concentrazione = -0.4+1.39*risultante+0.43/(1.0-risultante);}
        else {concentrazione = 1.0/(risultante*risultante*risultante-4.0*risultante*risultante+3.0*risultante);}
        if (taglia < 15 && concentrazione < 2.0){concentrazione = concentrazione-2.0/(taglia*concentrazione);}
        if (concentrazione < 0.0){concentrazione = 0.0;}
        // Ora calcoliamo l'intervallo di confidenza per l'angolo medio dell'intervallo;
        // ricordiamo che abbiamo dato la possibilità di usare sia un intervallo di confidenza
        // più semplice ma con \alfa più lontano dal livello nominale, e uno più accurato.
        double distanza;
        if (facile) {
            const double arcocosenando = (1.0-quantile)/(2.0*concentrazione*risultante*taglia);
            if (arcocosenando < -1.0 || arcocosenando > 1.0){accettazioni++; continue;}
            distanza = std::acos(arcocosenando);
        } else if (risultante < 2.0/3.0) {
            if (quantile > 2.0*taglia*risultante*risultante){accettazioni++; continue;}
            const double n2r2 = taglia*taglia*risultante*risultante;
            const double arcocosenando = std::sqrt(2.0*taglia*(2.0*n2r2-taglia*quantile)/(n2r2*(4.0*taglia-quantile)));
            if (arcocosenando < -1.0 || arcocosenando > 1.0){accettazioni++; continue;}
            distanza = std::acos(arcocosenando);
        } else {
            const double n2r2 = taglia*taglia*risultante*risultante;
            const double radicando = taglia*taglia-(taglia*taglia-n2r2)*std::exp(quantile);
            const double arcocosenando = std::sqrt(radicando)/n2r2;
            if (radicando < 0.0 || arcocosenando < -1.0 || arcocosenando > 1.0){accettazioni++; continue;}
            distanza = std::acos(arcocosenando);
        }
        const double basso = angolomedio-distanza;
        const double alto = angolomedio+distanza;
        // Poiché il campione è stato estratto da una von Mises con \mu=0, accettiamo l'ipotesi nulla
        // se l'intervallo contiene 0 (o, per sicurezza, 2\pi o -2\pi).
        if ((basso <= 0.0 && 0.0 <= alto) || (basso <= -2.0*pigreco && -2.0*pigreco <= alto) || (basso <= 2.0*pigreco && 2.0*pigreco <= alto)){accettazioni++;}
    }
    return accettazioni;
}


// Funzione per calcolare il livello di confidenza (o più precisamente l'\alfa) di un intervallo
// di confidenza per l'angolo medio. I parametri sono:
// Iterazioni: numero di iterazioni Monte Carlo.
// Kappa: il \kappa stimato dal campione, che sarà quello usato durante la generazione dei campioni Monte Carlo,
// che saranno estratti da un von Mises(0,\kappa).
// Taglia: numerosità del campione.
// Facile: nell'applicazione Python, si dà la possibilità di calcolare l'intervallo con una formula semplice
// e una più complessa. Usa "true" per usare quella semplice e "false" altrimenti.
// Quantile: il quantile di livello \alfa di una chi quadro con 1 grado di libertà.
double alfaverointervalloangolomedio(const int iterazioni, const double kappa, const int taglia, const bool facile,
    const double quantile) {
    if (iterazioni < 1){throw std::runtime_error("Le iterazioni devono essere positive");}
    if (kappa <= 0.0){throw std::runtime_error("Parametro di concentrazione non valido");}
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    std::vector<int> accettazionithread(numerothread);
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcolaalfaverointervalloangolomedio(volteperthread, kappa, taglia,
                facile, quantile);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazionithread]() {
        accettazionithread[numerothread-1] = calcolaalfaverointervalloangolomedio(volteperthread+iterazionirestanti, kappa, taglia,
            facile, quantile);
    });
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return 1.0-static_cast<double>(accettazioni)/iterazioni;
}


// Funzione che consente di ricavare un intervallo di confidenza bootstrap per \kappa dato un campione
// (non usiamo la formula di Mardia e Jupp poiché è valida solo per \kappa>2; questo metodo è più robusto
// per ogni possibile \kappa). I parametri sono:
// Angoli: un vettore contenente gli angoli del campione.
// Iterazioni: numero di iterazioni bootstrap.
// Alfa: uno meno il livello di confidenza.
// Taglia: numerosità del campione.
std::array<double, 2> intervallokappa(const std::vector<double> &angoli, const int iterazioni, const double alfa,
    const int taglia) {
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    if (angoli.size() != taglia){throw std::runtime_error("Il dataset non corrisponde alla taglia");}
    if (alfa <= 0.0 || alfa >= 1.0){throw std::runtime_error("Alfa non valido");}
    // Prepariamo i generatori di numeri casuali.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::uniform_int_distribution<> scelta(0, taglia-1);
    std::vector<double> stimekappa(iterazioni);
    // Per ogni iterazione bootstrap:
    for (int i = 0; i < iterazioni; i++) {
        // Estraiamo un campione bootstrap. Contemporaneamente, calcoliamo \bar{C} e \bar{S}.
        double sommacoseni = 0.0;
        double sommaseni = 0.0;
        for (int j = 0; j < taglia; j++) {
            const double angolo = angoli[scelta(generatore)];
            sommacoseni += std::cos(angolo);
            sommaseni += std::sin(angolo);
        }
        // A questo punto calcoliamo il \hat{\kappa} del campione, e salviamolo. Notare che dividiamo
        // qui per n in modo da fare un'operazione in meno.
        const double risultante = std::sqrt(sommacoseni*sommacoseni+sommaseni*sommaseni)/taglia;
        double stimakappa;
        if (risultante < 0.53){stimakappa = 2.0*risultante+risultante*risultante*risultante+5.0*risultante*risultante*
            risultante*risultante*risultante/6.0;}
        else if (0.53 <= risultante && risultante < 0.85){stimakappa = -0.4+1.39*risultante+0.43/(1.0-risultante);}
        else {stimakappa = 1.0/(risultante*risultante*risultante-4.0*risultante*risultante+3.0*risultante);}
        if (taglia < 15 && stimakappa < 2.0){stimakappa = stimakappa-2.0/(taglia*stimakappa);}
        if (stimakappa < 0.0){stimakappa = 0.0;}
        stimekappa[i] = stimakappa;
    }
    // Calcoliamo il quantile \alfa/2 e 1-\alfa/2 della distribuzione empirica di \kappa così ottenuta,
    // ottenendo un intervallo.
    std::sort(stimekappa.begin(), stimekappa.end());
    const double punto = iterazioni*alfa/2.0;
    const double peso = punto - std::floor(punto);
    const int primoindice = static_cast<int>(std::floor(punto));
    const int secondoindice = static_cast<int>(std::ceil(iterazioni-punto));
    double primoquantile;
    double secondoquantile;
    if (primoindice < 0){primoquantile = 0.0;} else {primoquantile = stimekappa[primoindice]*(1.0-peso)+stimekappa[primoindice+1]*peso;}
    if (secondoindice >= iterazioni){secondoquantile = std::numeric_limits<double>::infinity();}
    else {secondoquantile = stimekappa[secondoindice]*peso+stimekappa[secondoindice-1]*(1.0-peso);}
    return {primoquantile, secondoquantile};
}


// Funzione che esegue i calcoli per alfaverointervallokappa.
int calcolaalfaverointervallokappa(const int iterazioni, const double alfanominale, const double kappa,
    const int iterazionibootstrap, const int taglia) {
    // Prepariamo i generatori di numeri casuali.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::uniform_real_distribution<> uniforme(0.0, 1.0);
    std::uniform_int_distribution<> scelta(0, taglia-1);
    // Inizializziamo il contatore di volte in cui l'intervallo è corretto, e alcune costanti
    // utili alla generazione dei campioni.
    int accettazioni = 0;
    const double tau = 1.0+std::sqrt(1.0+kappa*kappa);
    const double rho = (tau-std::sqrt(2.0*tau))/(2.0*kappa);
    const double r = (1.0+rho*rho)/(2.0*rho);
    // Per ogni iterazione Monte Carlo:
    for (int i = 0; i < iterazioni; i++) {
        // Estraiamo un campione da una von Mises con \mu=0 e il \kappa specificato. Per semplicità
        // memorizzeremo solo i coseni e seni degli angoli.
        std::vector<double> campionecoseni(taglia);
        std::vector<double> campioneseni(taglia);
        int indice = 0;
        while (indice < taglia) {
            const double u1 = uniforme(generatore);
            const double z = std::cos(pigreco*u1);
            const double f = (1.0+r*z)/(r+z);
            if (r+z == 0.0){continue;}
            const double c = kappa*(r-f);
            if (c <= 0.0){continue;}
            const double u2 = uniforme(generatore);
            if (c*(2.0-c)-u2 > 0.0) {
                const double u3 = uniforme(generatore);
                campionecoseni[indice] = f;
                campioneseni[indice] = (u3 > 0.5 ? +1 : -1) * std::sqrt(1.0 - f*f);
                indice++;
            } else if (std::log(c/u2)+1-c >= 0) {
                const double u3 = uniforme(generatore);
                campionecoseni[indice] = f;
                campioneseni[indice] = (u3 > 0.5 ? +1 : -1) * std::sqrt(1.0 - f*f);
                indice++;
            }
        }
        // Sul campione così ottenuto calcoliamo un intervallo di confidenza bootstrap per \kappa.
        std::vector<double> stimekappa(iterazionibootstrap);
        for (int k = 0; k < iterazionibootstrap; k++) {
            double sommacoseni = 0.0;
            double sommaseni = 0.0;
            for (int j = 0; j < taglia; j++) {
                const int angolo = scelta(generatore);
                sommacoseni += campionecoseni[angolo];
                sommaseni += campioneseni[angolo];
            }
            const double risultante = std::sqrt(sommacoseni*sommacoseni+sommaseni*sommaseni)/taglia;
            double stimakappa;
            if (risultante < 0.53){stimakappa = 2.0*risultante+risultante*risultante*risultante+5.0*risultante*risultante*
                risultante*risultante*risultante/6.0;}
            else if (0.53 <= risultante && risultante < 0.85){stimakappa = -0.4+1.39*risultante+0.43/(1.0-risultante);}
            else {stimakappa = 1.0/(risultante*risultante*risultante-4.0*risultante*risultante+3.0*risultante);}
            if (taglia < 15 && stimakappa < 2.0){stimakappa = stimakappa-2.0/(taglia*stimakappa);}
            if (stimakappa < 0.0){stimakappa = 0.0;}
            stimekappa[k] = stimakappa;
        }
        std::sort(stimekappa.begin(), stimekappa.end());
        const double punto = iterazionibootstrap*alfanominale/2.0;
        const double peso = punto - std::floor(punto);
        const int primoindice = static_cast<int>(std::floor(punto));
        const int secondoindice = static_cast<int>(std::ceil(iterazionibootstrap-punto));
        double primoquantile;
        double secondoquantile;
        if (primoindice < 0){primoquantile = 0.0;} else {primoquantile = stimekappa[primoindice]*(1.0-peso)+stimekappa[primoindice+1]*peso;}
        if (secondoindice >= iterazionibootstrap){secondoquantile = std::numeric_limits<double>::infinity();}
        else {secondoquantile = stimekappa[secondoindice]*peso+stimekappa[secondoindice-1]*(1.0-peso);}
        // Se l'intervallo ottenuto contiene il \kappa specificato all'inizio per la generazione del
        // campione, accettiamo l'ipotesi nulla.
        if (primoquantile <= kappa && kappa <= secondoquantile){accettazioni++;}
    }
    return accettazioni;
}


// Funzione per calcolare il livello di confidenza vero (o più precisamente l'\alfa) di un intervallo
// di confidenza per il parametro \kappa di un campione da una von Mises. I parametri sono:
// Iterazioni: numero di iterazioni Monte Carlo, su cui viene giudicata la bontà della procedura
// di calcolo dell'intervallo.
// Alfanominale: \alfa nominale dell'intervallo bootstrap.
// Kappa: \kappa da usare per la generazione dei campioni.
// Iterazionibootstrap: iterazioni del metodo bootstrap per ricavare l'intervallo da un campione,
// fatte a ogni iterazione Monte Carlo.
// Taglia: numerosità del campione.
double alfaverointervallokappa(const int iterazioni, const double alfanominale, const double kappa,
    const int iterazionibootstrap, const int taglia) {
    if (iterazioni < 1){throw std::runtime_error("Le iterazioni devono essere positive");}
    if (alfanominale <= 0.0 || alfanominale >= 1.0){throw std::runtime_error("Alfa non valido");}
    if (kappa <= 0.0){throw std::runtime_error("Parametro di concentrazione non valido");}
    if (iterazionibootstrap < 1){throw std::runtime_error("Le iterazioni devono essere positive");}
    if (taglia < 2){throw std::runtime_error("Dataset troppo piccolo");}
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    std::vector<int> accettazionithread(numerothread);
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcolaalfaverointervallokappa(volteperthread, alfanominale, kappa,
                iterazionibootstrap, taglia);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazionithread]() {
        accettazionithread[numerothread-1] = calcolaalfaverointervallokappa(volteperthread+iterazionirestanti, alfanominale, kappa,
            iterazionibootstrap, taglia);
    });
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return 1.0-static_cast<double>(accettazioni)/iterazioni;
}


// Funzione che, assumendo un modello \Theta\sim vonMises(\mu,\beta\lambda) - \lambda\sim Gamma(a,b),
// cerca iterativamente i parametri della distribuzione a posteriori tramite inferenza variazionale.
// Le distribuzioni a priori sono vonMises(mu0, \lambda) (dunque \beta_0=1) e Gamma(a0, b0);
// n è la taglia del campione. Viene restituito un array {mu, beta, a, b}.
std::array<double, 4> angolomediobayesiano(const double mu0, const double a0, const double b0,
    const std::vector<double> &angoli, const int n) {
    if (a0 <= 0.0 || b0 <= 0.0){throw std::runtime_error("I parametri della Gamma devono essere positivi");}
    if (angoli.size() != n){throw std::runtime_error("Il dataset non corrisponde alla taglia");}
    if (n < 2){throw std::runtime_error("Dataset troppo piccolo");}
    // Iniziamo calcolando la somma di tutte le osservazioni in R^2.
    double sommacoseni = 0.0;
    double sommaseni = 0.0;
    for (int i = 0; i < n; i++) {
        sommacoseni += std::cos(angoli[i]);
        sommaseni += std::sin(angoli[i]);
    }
    // La stima posteriore di \beta è la norma di \beta_0\mu_0+somma.
    const double ascissaposteriore = std::cos(mu0)+sommacoseni;
    const double ordinataposteriore = std::sin(mu0)+sommaseni;
    const double beta = std::sqrt(ascissaposteriore*ascissaposteriore+ordinataposteriore*ordinataposteriore);
    // La stima di \mu invece è \beta_0\mu_0+somma diviso beta; faremo il calcolo più avanti.
    // Ora partiamo con l'algoritmo di inferenza variazionale: poniamo i parametri a e b pari a quelli priori.
    double a = a0;
    double b = b0;
    // Calcoliamo tutte le quantità necessarie al calcolo dell'ELBO iniziale.
    double lambdabarrato;
    if (a <= 1.0){lambdabarrato = a/b;} else {lambdabarrato = (a-1)/b;}
    double digammaa = 0.0;
    double x = a;
    while (x < 6.0) {
        digammaa -= 1.0/x;
        x++;
    }
    double xx = 1.0/x;
    double xx2 = xx*xx;
    double xx4 = xx2*xx2;
    digammaa += std::log(x) - 0.5 * xx - xx2 / 12.0 + xx4 / 120.0 - xx4 * xx2 / 252.0;
    double logb = std::log(b);
    if (lambdabarrato < 0.000001 || beta*lambdabarrato < 0.000001 || b*lambdabarrato){lambdabarrato += 0.00001;}
    double i0l = std::cyl_bessel_i(0.0, lambdabarrato);
    double i0bl = std::cyl_bessel_i(0.0, beta*lambdabarrato);
    double i1l;
    double i1bl;
    double elbo1 = (a0-1.0)*(digammaa-logb)-b0*a/b-(n+1)*std::log(i0l)+std::log(i0bl)-
        (n+1)*std::cyl_bessel_i(1.0, lambdabarrato)/i0l*(a/b-lambdabarrato)
        +std::cyl_bessel_i(1.0, beta*lambdabarrato)/i0bl*beta*lambdabarrato*
        (std::log(1.0/(b*lambdabarrato))+digammaa)+std::lgamma(a)+(a-1.0)*(digammaa-logb)-a/b;
    // Per ogni iterazione (e ci fermiamo comunque dopo 200):
    for (int iterazione = 0; iterazione < 200; iterazione++){
        // Calcoliamo \bar{\lambda} e poi le funzioni di Bessel necessarie.
        if (a <= 1.0){lambdabarrato = a/b;} else {lambdabarrato = (a-1.0)/b;}
        if (lambdabarrato < 0.000001 || beta*lambdabarrato < 0.000001 || b*lambdabarrato < 0.000001){lambdabarrato += 0.000001;}
        i0l = std::cyl_bessel_i(0.0, lambdabarrato);
        i0bl = std::cyl_bessel_i(0.0, beta*lambdabarrato);
        i1l = std::cyl_bessel_i(1.0, lambdabarrato);
        i1bl = std::cyl_bessel_i(1.0, beta*lambdabarrato);
        // Aggiorniamo la stima di a e b.
        a = a0+beta*lambdabarrato*i1bl/i0bl;
        b = b0+(n+1)*i1l/i0l;
        // Calcoliamo un'approssimazione della funzione digamma di a.
        digammaa = 0.0;
        x = a;
        while (x < 6.0) {
            digammaa -= 1.0/x;
            x++;
        }
        xx = 1.0/x;
        xx2 = xx*xx;
        xx4 = xx2*xx2;
        digammaa += std::log(x) - 0.5 * xx - xx2 / 12.0 + xx4 / 120.0 - xx4 * xx2 / 252.0;
        // A questo punto calcoliamo il nuovo ELBO.
        logb = std::log(b);
        double elbo2 = (a0-1.0)*(digammaa-logb)-b0*a/b-(n+1)*std::log(i0l)+std::log(i0bl)-
            (n+1.0)*i1l/i0l*(a/b-lambdabarrato)+i1bl/i0bl*beta*lambdabarrato*
                (std::log(1.0/(b*lambdabarrato))+digammaa)+std::lgamma(a)+(a-1.0)*(digammaa-logb)-a/b;
        // Se l'ELBO nuovo è sufficientemente vicino al precedente, dichiariamo raggiunta la convergenza.
        // Altrimenti passiamo alla prossima iterazione.
        if (elbo2-elbo1 < 0.0001){break;}
        elbo1 = elbo2;
    }
    // Restituiamo \mu in radianti, \beta, a e b.
    return {std::atan2(ordinataposteriore/beta, ascissaposteriore/beta), beta, a, b};
}
// Sei sicuro del calcolo dell'ELBO?


// Funzione per ricercare i parametri della distribuzione a posteriori di un modello bayesiano
// in cui i dati provengono da una mistura di vonMises(\mu_k, \lambda_k): \mu_k ha distribuzione
// a priori vonMises(m_{0k}, \beta_{0k}\lambda_k) e \lambda_k ha Gamma(a_{0k}, b_{0k}). Si arriva
// a una distribuzione a posteriori per \mu_k che è vonMises(m_k, \beta_k\lambda_k) e per \lambda_k
// Gamma(a_k, b_k); i pesi della mistura \tau_k hanno distribuzione a priori Dirichlet(\underline{\alpha}_0)
// che diventa a posteriori Dirichlet(\underline{\alpha}).
// I parametri vengono restituiti in dei vettori con un elemento per componente; viene inoltre restituito
// un ultimo vettore che contiene come unico elemento il BIC del modello valutato col numero
// specificato di componenti. I parametri della funzione sono:
// Alfa0: parametri a posteriori della distribuzione di Dirichlet sui pesi di mistura.
// Beta_0: ognuno di questi è un parametro che determina la relazione tra \lambda_k la varianza di \mu_k.
// M0: medie a priori della distribuzione dei \mu_k, in radianti.
// A0: parametri a priori di shape delle distribuzioni dei \lambda_k.
// B0: parametri a priori di rate delle distribuzioni dei \lambda_k.
// Componenti: numero di componenti della mistura.
// Angoli: vettore degli angoli del dataset in radianti.
// N: taglia del campione.
std::array<std::vector<double>, 7> misturevonmisesvariazionali(const std::vector<double> &alfa0,
    const std::vector<double> &beta0, const std::vector<double> &m0, const std::vector<double> &a0,
    const std::vector<double> &b0, const int componenti, std::vector<double> &angoli, const int n) {
    if (alfa0.size() != componenti || beta0.size() != componenti || m0.size() != componenti || a0.size() != componenti || b0.size() != componenti){throw std::runtime_error("Numero di parametri errato");}
    if (angoli.size() != n){throw std::runtime_error("Il dataset non corrisponde alla taglia");}
    if (n < componenti){throw std::runtime_error("Ci sono meno osservazioni che componenti");}
    for (int i = 0; i < componenti; i++) {
        if (alfa0[i] <= 0.000001 || beta0[i] <= 0.000001 || a0[i] <= 0.000001 || b0[i] <= 0.000001){throw std::runtime_error("I parametri sono troppo piccoli");}
    }
    // Inizializziamo alcune quantità che ci serviranno più tardi.
    std::vector<double> lambdabarrati(componenti);
    std::vector<double> cosenim0(componenti);
    std::vector<double> senim0(componenti);
    for (int i = 0; i < componenti; i++) {
        lambdabarrati[i] = a0[i]/b0[i];
        cosenim0[i] = std::cos(m0[i]);
        senim0[i] = std::sin(m0[i]);
    }
    // Prepariamo i generatori di numeri casuali.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::uniform_real_distribution<> uniforme(-pigreco, pigreco);
    // Ora dobbiamo inizializzare \xi_{nk}, usando il k-means. Dunque, poniamo
    // dei centroidi iniziali sul cerchio, a caso, in numero pari alle componenti della mistura.
    std::vector<double> centroidi(componenti);
    for (int i = 0; i < componenti; i++) {
        centroidi[i] = uniforme(generatore);
    }
    bool convergenza = false;
    int iterazionecorrente = 0;
    // Poiché ci serviranno più avanti, calcoliamo i coseni e i seni degli angoli una volta per tutte.
    std::vector<double> coseni(n);
    std::vector<double> seni(n);
    for (int i = 0; i < n; i++) {
        coseni[i] = std::cos(angoli[i]);
        seni[i] = std::sin(angoli[i]);
    }
    // Finché non arriviamo alla convergenza, per un massimo di 200 iterazioni:
    while (!convergenza && iterazionecorrente < 200){
        iterazionecorrente++;
        // Associamo ogni unità al centroide più vicino, e contemporaneamente
        // calcoliamo per ogni centroide \bar{C} e \bar{S} in modo da poter fare la media
        // delle unità per ogni centroide: questa sarà il nuovo centroide.
        std::vector<double> cosenicentroidi(componenti);
        std::vector<double> senicentroidi(componenti);
        for (int i = 0; i < n; i++) {
            const double angolo = angoli[i];
            double distanzaminima = std::min(std::abs(angolo-centroidi[0]),
                2.0*pigreco-std::abs(angolo-centroidi[0]));
            int centroide = 0;
            for (int j = 1; j < componenti; j++) {
                double distanza = std::min(std::abs(angolo-centroidi[j]), 2.0*pigreco-std::abs(angolo-centroidi[j]));
                if (distanza < distanzaminima) {
                    distanzaminima = distanza;
                    centroide = j;
                }
            }
            cosenicentroidi[centroide] += coseni[i];
            senicentroidi[centroide] += seni[i];
        }
        std::vector<double> nuovicentroidi(componenti);
        for (int i = 0; i < componenti; i++) {
            nuovicentroidi[i] = std::atan2(senicentroidi[i]/n, cosenicentroidi[i]/n);
        }
        // Se i centroidi ottenuti a quest'iterazione sono tutti distanti massimo 0.01 radianti
        // dai precedenti, dichiariamo la convergenza e smettiamo.
        convergenza = true;
        for (int i = 0; i < componenti; i++) {
            if (std::abs(centroidi[i]-nuovicentroidi[i]) > 0.01) {
                convergenza = false;
                break;
            }
        }
        centroidi = nuovicentroidi;
    }
    // Associamo ogni unità ai centroidi trovati e inizializziamo \xi_{nk} pari a 1 se l'unità n
    // si trova nel cluster k, e a 0 altrimenti.
    std::vector<std::vector<double>> xi(n, std::vector<double>(componenti, 0.0));
    for (int i = 0; i < n; i++) {
        double angolo = angoli[i];
        double distanzaminima = std::min(std::abs(angolo-centroidi[0]),
            2.0*pigreco-std::abs(angolo-centroidi[0]));
        int centroide = 0;
        for (int j = 1; j < componenti; j++) {
            double distanza = std::min(std::abs(angolo-centroidi[j]), 2.0*pigreco-std::abs(angolo-centroidi[j]));
            if (distanza < distanzaminima) {
                distanzaminima = distanza;
                centroide = j;
            }
        }
        xi[i][centroide] = 1.0;
    }
    // Prima di continuare abbiamo bisogno della funzione digamma di \sum_k\alpha_{0k}.
    double sommaalfa0 = 0.0;
    for (int i = 0; i < componenti; i++) {
        sommaalfa0 += alfa0[i];
    }
    const double conservasommaalfa0 = sommaalfa0;
    double digammasommaalfa0 = 0.0;
    while (sommaalfa0 < 6.0) {
        digammasommaalfa0 -= 1.0/sommaalfa0;
        sommaalfa0++;
    }
    const double xx = 1.0/sommaalfa0;
    const double xx2 = xx*xx;
    const double xx4 = xx2*xx2;
    digammasommaalfa0 += std::log(sommaalfa0) - 0.5 * xx - xx2 / 12.0 + xx4 / 120.0 - xx4 * xx2 / 252.0;
    const double log2pi = std::log(2.0*pigreco);
    // Adesso calcoliamo i valori iniziali di \tilde{ln(\rho_{nk})}.
    std::vector<std::vector<double>> lnrhonk(n, std::vector<double>(componenti, 0.0));
    for (int i = 0; i < componenti; i++) {
        double alfa0corrente = alfa0[i];
        double digammaalfa0 = 0.0;
        while (alfa0corrente < 6.0) {
            digammaalfa0 -= 1.0/alfa0corrente;
            alfa0corrente++;
        }
        const double aa = 1.0/alfa0corrente;
        const double aa2 = aa*aa;
        const double aa4 = aa2*aa2;
        digammaalfa0 += std::log(alfa0corrente)-0.5*aa-aa2/12.0+aa4/120.0-aa4*aa2/252.0;
        double lambdacorrente = lambdabarrati[i];
        if (lambdacorrente < 0.000001){lambdacorrente += 0.000001;}
        for (int j = 0; j < n; j++) {
            // Notare che poiché nel primo passo sappiamo che \bar{\lambda}_k=a_k/b_k sempre,
            // i calcoli si semplificano.
            lnrhonk[j][i] = digammaalfa0-digammasommaalfa0-log2pi+lambdacorrente*
                (cosenim0[i]*coseni[j]+senim0[i]*seni[j])-std::log(std::cyl_bessel_i(0.0, lambdacorrente));
        }
    }
    // Calcoliamo l'ELBO iniziale. Qui i calcoli sono semplificati perché le divergenze di
    // Kullback-Leibler sono nulle, essendo le distribuzioni "a posteriori" stimate sono ancora,
    // di fatto, le distribuzioni a priori.
    double elbo1 = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < componenti; j++) {
            const double xicorrente = xi[i][j];
            if (xicorrente <= 0.0){continue;}
            elbo1 += xicorrente*(lnrhonk[i][j]-std::log(xicorrente));
        }
    }
    // Adesso siamo pronti ad avviare davvero l'algoritmo variazionale.
    std::vector<double> alfa(componenti);
    std::vector<double> digammaalfa(componenti);
    std::vector<double> beta(componenti);
    std::vector<double> m(componenti);
    std::vector<double> cosenim(componenti);
    std::vector<double> senim(componenti);
    std::vector<double> a(componenti);
    std::vector<double> b(componenti);
    // Notare che qui gli indici, per comodità, sono scambiati: di fatto è \xi_{kn}.
    std::vector<std::vector<double>> xinuovi(componenti, std::vector<double>(n, 0.0));
    // A ogni iterazione, fino a un massimo di 200:
    for (int iterazione = 0; iterazione < 200; iterazione++) {
        // Calcoliamo per ogni n \sum_ke^{\tilde{ln\rho_{nk}}}.
        std::vector<double> elnrhon(n);
        for (int i = 0; i < n; i++) {
            const std::vector<double>& lnrhoncorrente = lnrhonk[i];
            double lnrhonsingolo = 0.0;
            for (int j = 0; j < componenti; j++) {
                lnrhonsingolo += std::exp(lnrhoncorrente[j]);
            }
            elnrhon[i] = lnrhonsingolo;
        }
        // A questo punto possiamo aggiornare gli \xi_{kn}.
        for (int i = 0; i < n; i++) {
            const std::vector<double>& lnrhonkcorrente = lnrhonk[i];
            double lnrhoncorrente = elnrhon[i];
            for (int j = 0; j < componenti; j++) {
                if (lnrhoncorrente < 0.000001){lnrhoncorrente += 0.000001;}
                xinuovi[j][i] = std::exp(lnrhonkcorrente[j])/lnrhoncorrente;
            }
        }
        // Ora possiamo aggiornare i parametri.
        for (int i = 0; i < componenti; i++) {
            // Iniziamo aggiornando \alpha_k.
            double sommaxi = 0.0;
            for (int j = 0; j < n; j++) {
                sommaxi += xinuovi[i][j];
            }
            alfa[i] = alfa0[i]+sommaxi;
            if (alfa[i] < 0.000001){alfa[i] += 0.000001;}
            // Estraiamo dei parametri che ci servono.
            double lambdacorrente = lambdabarrati[i];
            const double beta0corrente = beta0[i];
            const double cosenom0corrente = cosenim0[i];
            const double senom0corrente = senim0[i];
            // Ora aggiorniamo \beta, e nel mentre calcoliamo \sum_n\xi_{nk} per avvantaggiarci.
            double cosenixi = 0.0;
            double senixi = 0.0;
            double sommexi = 0.0;
            for (int j = 0; j < n; j++) {
                const double xicorrente = xinuovi[i][j];
                cosenixi += xicorrente*coseni[j];
                senixi += xicorrente*seni[j];
                sommexi += xicorrente;
            }
            const double betacoseno = beta0corrente*cosenom0corrente+cosenixi;
            const double betaseno = beta0corrente*senom0corrente+senixi;
            double nuovobeta = std::sqrt(betacoseno*betacoseno+betaseno*betaseno);
            if (nuovobeta < 0.000001){nuovobeta += 0.000001;}
            beta[i] = nuovobeta;
            const double ascissam = betacoseno/nuovobeta;
            const double ordinatam = betaseno/nuovobeta;
            cosenim[i] = ascissam;
            senim[i] = ordinatam;
            m[i] = std::atan2(ascissam, ordinatam);
            // E adesso aggiorniamo a e b.
            if (lambdacorrente < 0.000001 || nuovobeta*lambdacorrente < 0.000001 || beta0corrente*lambdacorrente < 0.000001){lambdacorrente += 0.000001;}
            double nuovoa = a0[i]+std::cyl_bessel_i(1.0, nuovobeta*lambdacorrente)/
                std::cyl_bessel_i(0.0, nuovobeta*lambdacorrente)*nuovobeta*lambdacorrente;
            double nuovob = b0[i]+std::cyl_bessel_i(1.0, lambdacorrente)/std::cyl_bessel_i(0.0, lambdacorrente)*sommexi+
                beta0corrente*std::cyl_bessel_i(1.0, beta0corrente*lambdacorrente)/
                    std::cyl_bessel_i(0.0, beta0corrente*lambdacorrente);
            a[i] = nuovoa;
            if (a[i] < 0.000001){a[i] += 0.000001;}
            b[i] = nuovob;
            if (b[i] < 0.000001){b[i] += 0.000001;}
            // E adesso aggiorniamo \bar{\lambda}_k.
            if (nuovoa > 1.0){lambdabarrati[i] = (nuovoa-1.0)/nuovob;} else {lambdabarrati[i] = nuovoa/nuovob;}
        }
        // Iniziamo a calcolare l'ELBO di questa iterazione. La prima parte è quella dell'equazione 27
        // del paper, ossia \sum_n\sum_k\xi_{nk}(\tilde{ln\rho_{nk}}-\xi_{nk}).
        double elbo2 = 0.0;
        for (int i = 0; i < componenti; i++) {
            for (int j = 0; j < n; j++) {
                double xiattuale = xinuovi[i][j];
                if (xiattuale <= 0.0){continue;}
                elbo2 += xiattuale*(lnrhonk[j][i]-std::log(xiattuale));
            }
        }
        // Ora sottraiamo la divergenza di Kullback-Leibler della distribuzione di Dirichlet,
        // ossia ln\frac{\Gamma(\sum_j\alpha_j)}{\Gamma(\sum_j\alpha_{0j})}-
        // \sum_jln\frac{\Gamma(\alpha_j)}{\Gamma(\alpha_{0j})}-\sum_j(\alpha_j-\alpha_{0j})(\psi(\alpha_j)-
        // \psi(\sum_i\alpha_i)).
        double sommaalfa = 0.0;
        for (int i = 0; i < componenti; i++) {
            sommaalfa += alfa[i];
        }
        const double conservasommaalfa = sommaalfa;
        double digammasommaalfa = 0.0;
        while (sommaalfa < 6.0) {
            digammasommaalfa -= 1.0/sommaalfa;
            sommaalfa++;
        }
        const double aa = 1.0/sommaalfa;
        const double aa2 = aa*aa;
        const double aa4 = aa2*aa2;
        digammasommaalfa += std::log(sommaalfa)-0.5*aa-aa2/12.0+aa4/120.0-aa4*aa2/252.0;
        elbo2 -= std::lgamma(conservasommaalfa)-std::lgamma(conservasommaalfa0);
        for (int i = 0; i < componenti; i++) {
            const double alfacorrente = alfa[i];
            const double alfa0corrente = alfa0[i];
            double alfa_corrente = alfacorrente;
            double digammaalfacorrente = 0.0;
            while (alfa_corrente < 6.0) {
                digammaalfacorrente -= 1.0/alfa_corrente;
                alfa_corrente++;
            }
            const double alfaalfa = 1.0/alfa_corrente;
            const double alfaalfa2 = alfaalfa*alfaalfa;
            const double alfaalfa4 = alfaalfa2*alfaalfa2;
            digammaalfacorrente += std::log(alfa_corrente)-0.5*alfaalfa-alfaalfa2/12.0+alfaalfa4/120.0-
                alfaalfa4*alfaalfa2/252.0;
            digammaalfa[i] = digammaalfacorrente;
            elbo2 += std::lgamma(alfacorrente)-std::lgamma(alfa0corrente)-
                (alfacorrente-alfa0corrente)*(digammaalfacorrente-digammasommaalfa);
        }
        // Adesso, per ogni componente, dobbiamo sottrarre la somma della divergenza di Kullback-Leibler
        // di una Gamma in \lambda_k e quella di una von Mises in \mu_k|\lambda_k; per quest'ultima usiamo
        // l'approssimazione \tilde{g(\lambda)} del paper per stimare la parte intrattabile. In soldoni:
        // KL_{\lambda_k}=a_{0k}ln(b_k/b_{0k})-ln\Gamma(a_k)+ln\Gamma(a_{0k})+(a_k-a_{0k})\psi(a_k)+a_k(b_{0k}/b_k-1);
        // KL_{\mu_k|\lambda_k}=(\beta a/b-\beta_0(a_0/b_0)\mu^T_p\mu_q)I_1(\beta a/b)/I_0(\beta a/b)-
        // (a_0-1)ln(a/b)-b_0a/b-nlnI_0(\bar{\lambda})-lnI_0(\beta_0\bar{\lambda})+lnI_0(\beta\bar{\lambda})-
        // nI_1(\bar{\lambda})/I_0(\bar{\lambda})(a/b-\bar{\lambda})-I_1(\beta_0\bar{\lambda})/
        // I_0(\beta_0\bar{\lambda})(\beta_0a/b-\beta_0\bar{\lambda})+I_1(\beta\bar{\lambda})/
        // I_0(\beta\bar{\lambda})\beta\bar{\lambda}(ln(\beta a/b)-ln(\beta\bar{\lambda})).
        for (int i = 0; i < componenti; i++) {
            double digammando = a[i];
            double digammaacorrente = 0.0;
            while (digammando < 6.0) {
                digammaacorrente -= 1.0/digammando;
                digammando++;
            }
            const double AA = 1.0/digammando;
            const double AA2 = AA*AA;
            const double AA4 = AA2*AA2;
            digammaacorrente += std::log(digammando)-0.5*AA-AA2/12.0+AA4/120.0-
                AA4*AA2/252.0;
            const double a0corrente = a0[i];
            const double b0corrente = b0[i];
            const double beta0corrente = beta0[i];
            const double acorrente = a[i];
            const double bcorrente = b[i];
            const double betacorrente = beta[i];
            double lambdacorrente = lambdabarrati[i];
            double kq = betacorrente*acorrente/bcorrente;
            if (lambdacorrente < 0.000001 || beta0corrente*lambdacorrente < 0.000001 || betacorrente*lambdacorrente < 0.000001){lambdacorrente += 0.000001;}
            if (kq < 0.000001){kq += 0.000001;}
            const double ilambda = std::cyl_bessel_i(0.0, lambdacorrente);
            const double ibeta0 = std::cyl_bessel_i(0.0, beta0corrente*lambdacorrente);
            const double ibeta = std::cyl_bessel_i(0.0, betacorrente*lambdacorrente);
            elbo2 -= a0corrente*std::log(bcorrente/b0corrente)-std::lgamma(acorrente)-std::lgamma(a0corrente)+
                (acorrente-a0corrente)*digammaacorrente+acorrente*(b0corrente/bcorrente-1.0)+(kq-beta0corrente*
                a0corrente/b0corrente*(cosenim0[i]*cosenim[i]+senim0[i]*senim[i]))*std::cyl_bessel_i(1.0, kq)/
                    std::cyl_bessel_i(0.0, kq)-(a0corrente-1.0)*std::log(acorrente/bcorrente)-b0corrente*acorrente/
                        bcorrente-n*std::log(ilambda)-std::log(ibeta0/ibeta)-n*std::cyl_bessel_i(1.0, lambdacorrente)/
                        ilambda*(acorrente/bcorrente-lambdacorrente)-
                            std::cyl_bessel_i(1.0, beta0corrente*lambdacorrente)/ibeta0*(beta0corrente*a0corrente/
                                bcorrente-beta0corrente*lambdacorrente)+
                                    std::cyl_bessel_i(1.0, betacorrente*lambdacorrente)/ibeta*betacorrente*
                                    lambdacorrente*std::log(kq/(betacorrente*lambdacorrente));
        }
        // Se questo ELBO è molto vicino a quello della scorsa iterazione, dichiariamo raggiunta
        // la convergenza e terminiamo l'algoritmo.
        if (elbo2-elbo1 < 0.000001){break;}
        elbo1 = elbo2;
        // Altrimenti, ricalcoliamo le quantità \tilde{ln\rho_{nk}} per poter ricominciare da capo.
        for (int i = 0; i < componenti; i++) {
            double digammaalfacorrente = digammaalfa[i];
            double ab = a[i]/b[i];
            double lambdacorrente = lambdabarrati[i];
            if (lambdacorrente < 0.000001){lambdacorrente += 0.000001;}
            for (int j = 0; j < n; j++) {
                lnrhonk[j][i] = digammaalfacorrente-digammasommaalfa-log2pi+ab*
                                (cosenim[i]*coseni[j]+senim[i]*seni[j])-
                                std::log(std::cyl_bessel_i(0.0, lambdacorrente))-
                                std::cyl_bessel_i(1.0, lambdacorrente)/std::cyl_bessel_i(0.0, lambdacorrente)*
                                    (ab-lambdacorrente);
            }
        }
    }
    // Terminato l'algoritmo, abbiamo le nostre stime dei parametri; ora calcoliamo il BIC
    // in modo da poter scegliere in seguito il numero di componenti. Lo poniamo dentro un vettore
    // monoelemento in modo da avere una function signature pulita.
    double logverosimiglianza = 0.0;
    double sommaalfafinale = 0.0;
    for (int i = 0; i < componenti; i++) {
        sommaalfafinale += alfa[i];
    }
    if (sommaalfafinale < 0.00001){sommaalfafinale += 0.00001;}
    for (int i = 0; i < n; i++) {
        long double sommatoriainterna = 0.0;
        for (int j = 0; j < componenti; j++) {
            long double agbg = beta[j]*a[j]/b[j];
            sommatoriainterna += static_cast<long double>(alfa[j])/sommaalfafinale*std::exp(agbg*std::cos(angoli[i]-m[j]))/
                (2.0L*pigreco*std::cyl_bessel_i(0.0L, agbg));
        }
        logverosimiglianza += static_cast<double>(std::log(sommatoriainterna));
    }
    std::vector<double> elnrhon(n);
    for (int i = 0; i < n; i++) {
        const std::vector<double>& lnrhoncorrente = lnrhonk[i];
        double lnrhonsingolo = 0.0;
        for (int j = 0; j < componenti; j++) {
            lnrhonsingolo += std::exp(lnrhoncorrente[j]);
        }
        elnrhon[i] = lnrhonsingolo;
    }
    for (int i = 0; i < n; i++) {
        const std::vector<double>& lnrhonkcorrente = lnrhonk[i];
        double lnrhoncorrente = elnrhon[i];
        for (int j = 0; j < componenti; j++) {
            if (lnrhoncorrente < 0.000001){lnrhoncorrente += 0.000001;}
            xinuovi[j][i] = std::exp(lnrhonkcorrente[j])/lnrhoncorrente;
        }
    }
    std::vector<double> assegnazioni(n);
    for (int i = 0; i < n; i++){
        int componentemassima = 0;
        double ximassimo = xinuovi[0][i];
        for (int j = 1; j < componenti; j++) {
            if (xinuovi[j][i] > ximassimo){componentemassima = j; ximassimo = xinuovi[j][i];}
        }
        assegnazioni[i] = componentemassima;
    }
    std::vector<double> contenitoreBIC(1);
    // Notare che per ogni componente ci sono sei parametri: due coordinate di m, beta, a, b e \alpha.
    contenitoreBIC[0] = -2.0*logverosimiglianza+6.0*componenti*std::log(n);
    return {m, beta, a, b, alfa, contenitoreBIC, assegnazioni};
}


// Funzione che esegue i calcoli per alfaverorayleigh.
int calcolaalfaverorayleigh(const int iterazioni, const double soglia, const int taglia) {
    // Prepariamo il generatore di numeri casuali.
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::uniform_real_distribution<> uniforme(-pigreco, pigreco);
    int accettazioni = 0;
    // Per ogni iterazione Monte Carlo:
    for (int i = 0; i < iterazioni; i++) {
        // Estraiamo un campione da un'uniforme circolare e contemporaneamente calcoliamo \bar{C} e \bar{S}.
        double sommacoseni = 0.0;
        double sommaseni = 0.0;
        for (int j = 0; j < taglia; j++) {
            const double angolo = uniforme(generatore);
            sommacoseni += std::cos(angolo);
            sommaseni += std::sin(angolo);
        }
        sommacoseni /= taglia;
        sommaseni /= taglia;
        // Se la statistica di Rayleigh è sotto la soglia critica, accettiamo l'ipotesi nulla.
        if (std::sqrt(sommacoseni*sommacoseni+sommaseni*sommaseni)*taglia*2.0 <= soglia){accettazioni++;}
    }
    return accettazioni;
}


// Funzione che calcola l'\alpha effettivo del test di Rayleigh per l'uniformità circolare.
// I parametri sono il numero di iterazioni Monte Carlo, il quantile 1-alfanominale di una chi
// quadro con 2 gradi di libertà e la numerosità del campione.
double alfaverorayleigh(const int iterazioni, const double soglia, const int taglia) {
    if (iterazioni < 1 || taglia < 2){throw std::runtime_error("Le iterazioni devono essere positive o il dataset è troppo piccolo");}
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    std::vector<int> accettazionithread(numerothread);
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcolaalfaverorayleigh(volteperthread, soglia, taglia);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazionithread]() {
        accettazionithread[numerothread-1] = calcolaalfaverorayleigh(volteperthread+iterazionirestanti,
            soglia, taglia);
    });
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return 1.0-static_cast<double>(accettazioni)/iterazioni;
}


// Funzione che esegue i calcoli per betaaffidabilitavonmises.
int calcolabetaaffidabilitavonmises(const int iterazioni, const bool uniforme, const int taglia,
    const double soglia, const double distanzacomponenti, const double kappa) {
    // Prepariamo il generatore di numeri casuali.
    int accettazioni = 0;
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    // Se l'ipotesi alternativa (vera) è che la distribuzione sia uniforme:
    if (uniforme) {
        std::uniform_real_distribution<> distribuzione(-pigreco, pigreco);
        // Per ogni iterazione:
        for (int i = 0; i < iterazioni; i++) {
            // Estraiamo un campione da un'uniforme circolare e contemporaneamente calcoliamo
            // \bar{C} e \bar{S} per trovare l'angolo medio.
            std::vector<double> campione(taglia);
            double sommacoseni = 0.0;
            double sommaseni = 0.0;
            for (int j = 0; j < taglia; j++) {
                const double elemento = distribuzione(generatore);
                campione[j] = elemento;
                sommacoseni += std::cos(elemento);
                sommaseni += std::sin(elemento);
            }
            // Troviamo \bar{\theta} e \bar{\kappa}.
            const double angolomedio = std::atan2(sommaseni/taglia, sommacoseni/taglia);
            const double risultante = std::sqrt(sommacoseni*sommacoseni+sommaseni*sommaseni)/taglia;
            double concentrazione;
            if (risultante < 0.53){concentrazione = 2.0*risultante+risultante*risultante*risultante+5.0*risultante*risultante*
                risultante*risultante*risultante/6.0;}
            if (0.53 <= risultante && risultante < 0.85){concentrazione = -0.4+1.39*risultante+0.43/(1.0-risultante);}
            if (risultante >= 0.85) {concentrazione = 1.0/(risultante*risultante*risultante-4.0*risultante*risultante+3.0*risultante);}
            if (taglia < 15 && concentrazione < 2.0){concentrazione = concentrazione-2.0/(taglia*concentrazione);}
            if (concentrazione < 0.0){concentrazione = 0.0;}
            // Calcoliamo subito le funzioni di Bessel che ci serviranno più avanti.
            const double i0 = std::cyl_bessel_i(0.0, concentrazione);
            const double i1 = std::cyl_bessel_i(1.0, concentrazione);
            const double i2 = std::cyl_bessel_i(2.0, concentrazione);
            const double i3 = std::cyl_bessel_i(3.0, concentrazione);
            const double i4 = std::cyl_bessel_i(4.0, concentrazione);
            // Ora calcoliamo le statistiche s_c e s_s.
            double sc = 0.0;
            double ss = 0.0;
            for (int j = 0; j < taglia; j++) {
                sc += std::cos(2.0*(campione[j]-angolomedio));
                ss += std::sin(2.0*(campione[j]-angolomedio));
            }
            sc -= taglia*i1/i0;
            // E adesso passiamo a v_c e v_s.
            const double numeratorevc = i0*i3+i0*i1-2.0*i1*i2;
            const double pezzonumeratorevs = i0-i3;
            const double vc = (i0*i0+i0*i4-2.0*i2*i2)/(2.0*i0*i0)-numeratorevc*numeratorevc/(2.0*i0*i0*(i0*i0+i0*i1-2.0*i1*i1));
            const double vs = ((i0-i4)*(i0-i2)-pezzonumeratorevs*pezzonumeratorevs)/(2.0*i0*(i0-i2));
            if (sc*sc/vc+ss*ss/vs < soglia){accettazioni++;}
        }
        // Se invece abbiamo scelto una mistura di due von Mises:
    } else {
        std::uniform_int_distribution<> divisore(0, 1);
        std::uniform_real_distribution<> distribuzione(0.0, 1.0);
        // Prepariamo alcune costanti utili alla generazione.
        const double tau = 1.0+std::sqrt(1.0+kappa*kappa);
        const double rho = (tau-std::sqrt(2.0*tau))/(2.0*kappa);
        const double r = (1.0+rho*rho)/(2.0*rho);
        // Per ogni iterazione Monte Carlo:
        for (int i = 0; i < iterazioni; i++) {
            std::vector<double> campione(taglia);
            // Per generare da una mistura di von Mises, scegliamo prima una delle due componenti
            // e poi estraiamo un'osservazione da essa: le due componenti hanno in particolare
            // \kappa uguale, mentre la prima ha media 0 e l'altra media distanzacomponenti.
            // Poiché si estrae da una von Mises con un metodo del tipo accettazione-rifiuto, serve
            // un ciclo while che insista a estrarre dalla componente finché non ci riesce.
            // Contemporaneamente, calcoleremo \bar{C} e \bar{S}.
            double sommacoseni = 0.0;
            double sommaseni = 0.0;
            for (int j = 0; j < taglia; j++) {
                bool estraendo = true;
                const int componente = divisore(generatore);
                const double media = componente*distanzacomponenti;
                const double cosenomedia = std::cos(media);
                const double senomedia = std::sin(media);
                while (estraendo) {
                    const double u1 = distribuzione(generatore);
                    const double z = std::cos(pigreco*u1);
                    if (r+z == 0.0){continue;}
                    const double f = (1.0+r*z)/(r+z);
                    const double c = kappa*(r-f);
                    if (c <= 0.0){continue;}
                    const double u2 = distribuzione(generatore);
                    if (c*(2.0-c)-u2 > 0.0){
                        const double u3 = distribuzione(generatore);
                        if (u3-0.5 > 0.0) {
                            const double angolo = std::acos(f)+media;
                            campione[j] = angolo;
                            sommacoseni += std::cos(angolo);
                            sommaseni   += std::sin(angolo);
                            estraendo = false;
                        } else {
                            const double angolo = -std::acos(f)+media;
                            campione[j] = angolo;
                            sommacoseni += std::cos(angolo);
                            sommaseni   += std::sin(angolo);
                            estraendo = false;
                        }
                    } else if (std::log(c/u2)+1.0-c >= 0.0) {
                        const double u3 = distribuzione(generatore);
                        if (u3-0.5 > 0.0) {
                            const double angolo = std::acos(f)+media;
                            campione[j] = angolo;
                            sommacoseni += std::cos(angolo);
                            sommaseni   += std::sin(angolo);
                            estraendo = false;
                        } else {
                            const double angolo = -std::acos(f)+media;
                            campione[j] = angolo;
                            sommacoseni += std::cos(angolo);
                            sommaseni   += std::sin(angolo);
                            estraendo = false;
                        }
                    }
                }
            }
            // Generato il campione, eseguiamo il test al solito modo, come nella funzione
            // testvonmises.
            const double angolomedio = std::atan2(sommaseni/taglia, sommacoseni/taglia);
            const double risultante = std::sqrt(sommacoseni*sommacoseni+sommaseni*sommaseni)/taglia;
            double concentrazione;
            if (risultante < 0.53){concentrazione = 2.0*risultante+risultante*risultante*risultante+5.0*risultante*risultante*
                risultante*risultante*risultante/6.0;}
            if (0.53 <= risultante && risultante < 0.85){concentrazione = -0.4+1.39*risultante+0.43/(1.0-risultante);}
            if (risultante >= 0.85) {concentrazione = 1.0/(risultante*risultante*risultante-4.0*risultante*risultante+3.0*risultante);}
            if (taglia < 15 && concentrazione < 2.0){concentrazione = concentrazione-2.0/(taglia*concentrazione);}
            if (concentrazione < 0.0){concentrazione = 0.0;}
            // Calcoliamo subito le funzioni di Bessel che ci serviranno più avanti.
            const double i0 = std::cyl_bessel_i(0.0, concentrazione);
            const double i1 = std::cyl_bessel_i(1.0, concentrazione);
            const double i2 = std::cyl_bessel_i(2.0, concentrazione);
            const double i3 = std::cyl_bessel_i(3.0, concentrazione);
            const double i4 = std::cyl_bessel_i(4.0, concentrazione);
            // Ora calcoliamo le statistiche s_c e s_s.
            double sc = 0.0;
            double ss = 0.0;
            for (int j = 0; j < taglia; j++) {
                sc += std::cos(2.0*(campione[j]-angolomedio));
                ss += std::sin(2.0*(campione[j]-angolomedio));
            }
            sc -= taglia*i1/i0;
            // E adesso passiamo a v_c e v_s.
            const double numeratorevc = i0*i3+i0*i1-2.0*i1*i2;
            const double pezzonumeratorevs = i0-i3;
            const double vc = (i0*i0+i0*i4-2.0*i2*i2)/(2.0*i0*i0)-numeratorevc*numeratorevc/(2.0*i0*i0*(i0*i0+i0*i1-2.0*i1*i1));
            const double vs = ((i0-i4)*(i0-i2)-pezzonumeratorevs*pezzonumeratorevs)/(2.0*i0*(i0-i2));
            if (sc*sc/vc+ss*ss/vs <= soglia){accettazioni++;}
        }
    }
    return accettazioni;
}


// Funzione che calcola la probabilità di errore di II tipo del test di goodness of fit della von Mises.
// I parametri sono:
// Iterazioni: numero di iterazioni Monte Carlo.
// Uniforme: se "true", l'ipotesi nulla è che i dati provengano da un'uniforme circolare; se "false",
// è che provengano da una mistura di due von Mises con concentrazione kappa e una media pari a 0
// e l'altra media pari a distanzacomponenti.
// Taglia: numerosità del campione.
// Soglia: quantile 1-alfa di una chi quadro con 2 gradi di libertà.
double betaaffidabilitavonmises(const int iterazioni, const bool uniforme, const int taglia,
    const double soglia, const double distanzacomponenti, const double kappa) {
    if (iterazioni < 1 || taglia < 2 || kappa < 0.000001){throw std::runtime_error("Numero di iterazioni/taglia/kappa improprio");}
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    std::vector<int> accettazionithread(numerothread);
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcolabetaaffidabilitavonmises(volteperthread, uniforme, taglia, soglia,
                distanzacomponenti, kappa);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazionithread]() {
        accettazionithread[numerothread-1] = calcolabetaaffidabilitavonmises(volteperthread+iterazionirestanti, uniforme,
            taglia, soglia, distanzacomponenti, kappa);
    });
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return static_cast<double>(accettazioni)/iterazioni;
}


// Funzione che esegue i calcoli per alfaveroaffidabilitavonmises.
int calcolaalfaveroaffidabilitavonmises(const int iterazioni, const double kappa,
    const int taglia, const double soglia) {
    // Prepariamo il generatore di numeri casuali e alcune costanti utili alla generazione.
    int accettazioni = 0;
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::uniform_real_distribution<> distribuzione(0.0, 1.0);
    const double tau = 1.0+std::sqrt(1.0+kappa*kappa);
    const double rho = (tau-std::sqrt(2.0*tau))/(2.0*kappa);
    const double r = (1.0+rho*rho)/(2.0*rho);
    for (int i = 0; i < iterazioni; i++) {
        std::vector<double> campione(taglia);
        double sommacoseni = 0.0;
        double sommaseni = 0.0;
        int indice = 0;
        while (indice < taglia) {
            const double u1 = distribuzione(generatore);
            const double z = std::cos(pigreco*u1);
            if (r+z == 0.0){continue;}
            const double f = (1.0+r*z)/(r+z);
            const double c = kappa*(r-f);
            if (c <= 0.0){continue;}
            const double u2 = distribuzione(generatore);
            if (c*(2.0-c)-u2 > 0.0){
                const double u3 = distribuzione(generatore);
                sommacoseni += f;
                sommaseni   += (u3 > 0.5 ? +1 : -1) * std::sqrt(1.0 - f*f);
                indice++;
            } else if (std::log(c/u2)+1.0-c >= 0.0) {
                const double u3 = distribuzione(generatore);
                sommacoseni += f;
                sommaseni   += (u3 > 0.5 ? +1 : -1) * std::sqrt(1.0 - f*f);
                indice++;
            }
        }
        // Ora eseguiamo il test di goodness of fit della von Mises al modo solito.
        const double angolomedio = std::atan2(sommaseni/taglia, sommacoseni/taglia);
        const double risultante = std::sqrt(sommacoseni*sommacoseni+sommaseni*sommaseni)/taglia;
        double concentrazione;
        if (risultante < 0.53){concentrazione = 2.0*risultante+risultante*risultante*risultante+5.0*risultante*risultante*
            risultante*risultante*risultante/6.0;}
        if (0.53 <= risultante && risultante < 0.85){concentrazione = -0.4+1.39*risultante+0.43/(1.0-risultante);}
        if (risultante >= 0.85) {concentrazione = 1.0/(risultante*risultante*risultante-4.0*risultante*risultante+3.0*risultante);}
        if (taglia < 15 && concentrazione < 2.0){concentrazione = concentrazione-2.0/(taglia*concentrazione);}
        if (concentrazione < 0.0){concentrazione = 0.0;}
        // Calcoliamo subito le funzioni di Bessel che ci serviranno più avanti.
        const double i0 = std::cyl_bessel_i(0.0, concentrazione);
        const double i1 = std::cyl_bessel_i(1.0, concentrazione);
        const double i2 = std::cyl_bessel_i(2.0, concentrazione);
        const double i3 = std::cyl_bessel_i(3.0, concentrazione);
        const double i4 = std::cyl_bessel_i(4.0, concentrazione);
        // Ora calcoliamo le statistiche s_c e s_s.
        double sc = 0.0;
        double ss = 0.0;
        for (int j = 0; j < taglia; j++) {
            sc += std::cos(2.0*(campione[j]-angolomedio));
            ss += std::sin(2.0*(campione[j]-angolomedio));
        }
        sc -= taglia*i1/i0;
        // E adesso passiamo a v_c e v_s.
        const double numeratorevc = i0*i3+i0*i1-2.0*i1*i2;
        const double pezzonumeratorevs = i0-i3;
        const double vc = (i0*i0+i0*i4-2.0*i2*i2)/(2.0*i0*i0)-numeratorevc*numeratorevc/(2.0*i0*i0*(i0*i0+i0*i1-2.0*i1*i1));
        const double vs = ((i0-i4)*(i0-i2)-pezzonumeratorevs*pezzonumeratorevs)/(2.0*i0*(i0-i2));
        if (sc*sc/vc+ss*ss/vs < soglia){accettazioni++;}
    }
    return accettazioni;
}


// Funzione per calcolare la probabilità effettiva di errore di I tipo del test di goodness of fit
// della von Mises. I parametri sono:
// Iterazioni: numero di iterazioni Monte Carlo.
// Kappa: parametro di concentrazione della distribuzione da cui si estrae il campione.
// Taglia: numerosità del campione.
// Soglia: quantile 1-alfa di una chi quadro con 2 gradi di libertà.
double alfaveroaffidabilitavonmises(const int iterazioni, const double kappa, const int taglia,
    const double soglia) {
    if (iterazioni < 1 || kappa < 0.000001 || taglia < 2){throw std::runtime_error("Numero di iterazioni/taglia/kappa improprio");}
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni%numerothread;
    std::vector<int> accettazionithread(numerothread);
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazionithread](){
            accettazionithread[i] = calcolaalfaveroaffidabilitavonmises(volteperthread, kappa, taglia, soglia);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazionithread]() {
        accettazionithread[numerothread-1] = calcolaalfaveroaffidabilitavonmises(volteperthread+iterazionirestanti, kappa, taglia, soglia);
    });
    for (auto& t : thread){
        t.join();
    }
    int accettazioni = 0;
    for (int i = 0; i < numerothread; i++){accettazioni += accettazionithread[i];}
    return 1.0-static_cast<double>(accettazioni)/iterazioni;
}

void calcolabootstrapstazionariohotelling(std::vector<double> &statistiche, const int thread, const int iterazioni, const int aggiuntive,
    const int taglia, const std::vector<double> &ascisse, const std::vector<double> &ordinate, const double p,
    const double mediaascisse, const double mediaordinate){
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::geometric_distribution<> geometrica(p);
    std::uniform_int_distribution<> uniforme(0, taglia-1);
    for (int i = 0; i < iterazioni+aggiuntive; i++){
        std::vector<double> campioneascisse(taglia);
        std::vector<double> campioneordinate(taglia);
        int indice = 0;
        while (true){
            const int inizioblocco = uniforme(generatore);
            const int fineblocco = inizioblocco+geometrica(generatore)+1;
            for (int j = inizioblocco; j < fineblocco; j++) {
                const int posizione = j % taglia;
                campioneascisse[indice] = ascisse[posizione];
                campioneordinate[indice] = ordinate[posizione];
                indice++;
                if (indice >= taglia){break;}
            }
            if (indice >= taglia){break;}
        }
        double mba = 0.0;
        double mbo = 0.0;
        for (int j = 0; j < taglia; j++) {
            mba += campioneascisse[j];
            mbo += campioneordinate[j];
        }
        mba /= taglia;
        mbo /= taglia;
        double vba = 0.0;
        double vbo = 0.0;
        double cb = 0.0;
        for (int j = 0; j < taglia; j++) {
            const double d1 = campioneascisse[j]-mba;
            const double d2 = campioneordinate[j]-mbo;
            vba += d1*d1;
            vbo += d2*d2;
            cb += d1*d2;
        }
        vba /= taglia-1;
        vbo /= taglia-1;
        cb /= taglia-1;
        double determinante = vba*vbo-cb*cb;
        if (determinante < 0.000000001){
            vba += 0.000001;
            vbo += 0.000001;
            determinante = vba*vbo-cb*cb;
        }
        const double inversa11 = vbo/determinante;
        const double inversa22 = vba/determinante;
        const double inversa12 = -cb/determinante;
        const double distanzaascisse = mediaascisse-mba;
        const double distanzaordinate = mediaordinate-mbo;
        statistiche[thread*iterazioni+i] = inversa11*distanzaascisse*distanzaascisse+2.0*inversa12*
            distanzaascisse*distanzaordinate+inversa22*distanzaordinate*distanzaordinate;
    }
}

double bootstrapstazionariohotelling(const int taglia, const std::vector<double> ascisse, const std::vector<double> ordinate,
    const double p, const int B, const double mediaascisse, const double mediaordinate, const double alfa){
    if (taglia < 3){throw std::runtime_error("Dataset troppo piccolo");}
    if (ascisse.size() != taglia || ordinate.size() != taglia){throw std::runtime_error("Dataset sbagliato");}
    if (p <= 0 || p >= 1){throw std::runtime_error("Probabilità sbagliata");}
    if (B < 1){throw std::runtime_error("Iterazioni insufficienti");}
    if (alfa <= 0.0 || alfa >= 1.0){throw std::runtime_error("Alfa sbagliato");}
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = B/numerothread;
    const int iterazionirestanti = B%numerothread;
    std::vector<double> statistiche(B, 0.0);
    for (int i = 0; i < numerothread - 1; i++){
        int indicethread = i;
        thread[i] = std::thread([=, &statistiche](){
            calcolabootstrapstazionariohotelling(statistiche, indicethread, volteperthread, 0, taglia,
                ascisse, ordinate, p, mediaascisse, mediaordinate);
        });
    }
    thread[numerothread - 1] = std::thread([=, &statistiche](){
        calcolabootstrapstazionariohotelling(statistiche, numerothread-1, volteperthread, iterazionirestanti,
            taglia, ascisse, ordinate, p, mediaascisse, mediaordinate);
    });
    for (auto& t : thread){
        t.join();
    }
    std::sort(statistiche.begin(), statistiche.end());
    const double indice = B*(1.0-alfa);
    const double peso = indice-std::floor(indice);
    if (indice >= B){return statistiche[B-1];}
    if (indice < 1.0){return statistiche[0];}
    const int indiceintero = static_cast<int>(indice);
    return (1.0-peso)*statistiche[indiceintero-1]+peso*statistiche[indiceintero];
}


double alfaverobootstrapstazionariohotelling(const int taglia, const double varianzaascisse,
    const double varianzaordinate, const double p, const int B, const double alfa, const int volte) {
    if (volte <= 0 || varianzaascisse <= 0.0 || varianzaordinate <= 0.0){throw std::runtime_error("Errore");}
    int accettazioni = 0;
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normaleascisse(0.0, varianzaascisse);
    std::normal_distribution<> normaleordinate(0.0, varianzaordinate);
    for (int i = 0; i < volte; i++) {
        std::vector<double> ascisse(taglia);
        std::vector<double> ordinate(taglia);
        double mediaascisse = 0.0;
        double mediaordinate = 0.0;
        for (int j = 0; j < taglia; j++) {
            const double ascissa = normaleascisse(generatore);
            const double ordinata = normaleordinate(generatore);
            mediaascisse += ascissa;
            mediaordinate += ordinata;
            ascisse[j] = ascissa;
            ordinate[j] = ordinata;
        }
        mediaascisse /= taglia;
        mediaordinate /= taglia;
        double varianzaascissecampionaria = 0.0;
        double varianzaordinatecampionaria = 0.0;
        double covarianza = 0.0;
        for (int j = 0; j < taglia; j++) {
            const double distanzaascisse = ascisse[j]-mediaascisse;
            const double distanzaordinate = ordinate[j]-mediaordinate;
            varianzaascissecampionaria += distanzaascisse*distanzaascisse;
            varianzaordinatecampionaria += distanzaordinate*distanzaordinate;
            covarianza += distanzaascisse*distanzaordinate;
        }
        varianzaascissecampionaria /= taglia-1;
        varianzaordinatecampionaria /= taglia-1;
        covarianza /= taglia-1;
        double determinante = varianzaascissecampionaria*varianzaordinatecampionaria-covarianza*covarianza;
        if (determinante < 0.00000001) {
            varianzaascissecampionaria += 0.0001;
            varianzaordinatecampionaria += 0.0001;
            determinante = varianzaascissecampionaria*varianzaordinatecampionaria-covarianza*covarianza;
        }
        const double soglia = bootstrapstazionariohotelling(taglia, ascisse, ordinate, p, B, mediaascisse, mediaordinate, alfa);
        const double formaquadratica = (varianzaordinatecampionaria*mediaascisse*mediaascisse-2.0*covarianza*mediaascisse*
            mediaordinate+varianzaascissecampionaria*mediaordinate*mediaordinate)/determinante;
        if (formaquadratica <= soglia){accettazioni++;}
    }
    return 1.0-static_cast<double>(accettazioni)/volte;
}


void calcolabootstrapmobilehotelling(std::vector<double> &statistiche, const int thread, const int iterazioni, const int aggiuntive,
    const int taglia, const std::vector<double> &ascisse, const std::vector<double> &ordinate, const int b,
    const double mediaascisse, const double mediaordinate){
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    const int k = taglia/b+1;
    const int eccessiva = b*k;
    std::uniform_int_distribution<> uniforme(0, taglia-b);
    for (int i = 0; i < iterazioni+aggiuntive; i++){
        std::vector<double> campioneascisse(eccessiva);
        std::vector<double> campioneordinate(eccessiva);
        for (int j = 0; j < k; j++) {
            const int inizioblocco = uniforme(generatore);
            for (int l = 0; l < b; l++) {
                campioneascisse[b*j+l] = ascisse[inizioblocco+l];
                campioneordinate[b*j+l] = ordinate[inizioblocco+l];
            }
        }
        double mba = 0.0;
        double mbo = 0.0;
        for (int j = 0; j < taglia; j++) {
            mba += campioneascisse[j];
            mbo += campioneordinate[j];
        }
        mba /= taglia;
        mbo /= taglia;
        double vba = 0.0;
        double vbo = 0.0;
        double cb = 0.0;
        for (int j = 0; j < taglia; j++) {
            const double d1 = campioneascisse[j]-mba;
            const double d2 = campioneordinate[j]-mbo;
            vba += d1*d1;
            vbo += d2*d2;
            cb += d1*d2;
        }
        vba /= taglia-1;
        vbo /= taglia-1;
        cb /= taglia-1;
        double determinante = vba*vbo-cb*cb;
        if (determinante < 0.000000001){
            vba += 0.000001;
            vbo += 0.000001;
            determinante = vba*vbo-cb*cb;
        }
        const double inversa11 = vbo/determinante;
        const double inversa22 = vba/determinante;
        const double inversa12 = -cb/determinante;
        const double distanzaascisse = mediaascisse-mba;
        const double distanzaordinate = mediaordinate-mbo;
        statistiche[thread*iterazioni+i] = inversa11*distanzaascisse*distanzaascisse+2.0*inversa12*
            distanzaascisse*distanzaordinate+inversa22*distanzaordinate*distanzaordinate;
    }
}


double bootstrapmobilehotelling(const int taglia, const std::vector<double> ascisse, const std::vector<double> ordinate,
    const int b, const int B, const double mediaascisse, const double mediaordinate, const double alfa){
    if (taglia < 3){throw std::runtime_error("Dataset troppo piccolo");}
    if (ascisse.size() != taglia || ordinate.size() != taglia){throw std::runtime_error("Dataset sbagliato");}
    if (b <= 0 || b >= taglia){throw std::runtime_error("Lunghezza dei blocchi sbagliata");}
    if (B < 1){throw std::runtime_error("Iterazioni insufficienti");}
    if (alfa <= 0.0 || alfa >= 1.0){throw std::runtime_error("Alfa sbagliato");}
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = B/numerothread;
    const int iterazionirestanti = B%numerothread;
    std::vector<double> statistiche(B, 0.0);
    for (int i = 0; i < numerothread - 1; i++){
        int indicethread = i;
        thread[i] = std::thread([=, &statistiche](){
            calcolabootstrapmobilehotelling(statistiche, indicethread, volteperthread, 0, taglia,
                ascisse, ordinate, b, mediaascisse, mediaordinate);
        });
    }
    thread[numerothread - 1] = std::thread([=, &statistiche](){
        calcolabootstrapmobilehotelling(statistiche, numerothread-1, volteperthread, iterazionirestanti,
            taglia, ascisse, ordinate, b, mediaascisse, mediaordinate);
    });
    for (auto& t : thread){
        t.join();
    }
    std::sort(statistiche.begin(), statistiche.end());
    const double indice = B*(1.0-alfa);
    const double peso = indice-std::floor(indice);
    if (indice >= B){return statistiche[B-1];}
    if (indice < 1.0){return statistiche[0];}
    const int indiceintero = static_cast<int>(indice);
    return (1.0-peso)*statistiche[indiceintero-1]+peso*statistiche[indiceintero];
}


double alfaverobootstrapmobilehotelling(const int taglia, const double varianzaascisse,
    const double varianzaordinate, const int b, const int B, const double alfa, const int volte) {
    if (volte <= 0 || varianzaascisse <= 0.0 || varianzaordinate <= 0.0){throw std::runtime_error("Errore");}
    int accettazioni = 0;
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normaleascisse(0.0, varianzaascisse);
    std::normal_distribution<> normaleordinate(0.0, varianzaordinate);
    for (int i = 0; i < volte; i++){
        std::vector<double> ascisse(taglia);
        std::vector<double> ordinate(taglia);
        double mediaascisse = 0.0;
        double mediaordinate = 0.0;
        for (int j = 0; j < taglia; j++) {
            const double ascissa = normaleascisse(generatore);
            const double ordinata = normaleordinate(generatore);
            mediaascisse += ascissa;
            mediaordinate += ordinata;
            ascisse[j] = ascissa;
            ordinate[j] = ordinata;
        }
        mediaascisse /= taglia;
        mediaordinate /= taglia;
        double varianzaascissecampionaria = 0.0;
        double varianzaordinatecampionaria = 0.0;
        double covarianza = 0.0;
        for (int j = 0; j < taglia; j++) {
            const double distanzaascisse = ascisse[j]-mediaascisse;
            const double distanzaordinate = ordinate[j]-mediaordinate;
            varianzaascissecampionaria += distanzaascisse*distanzaascisse;
            varianzaordinatecampionaria += distanzaordinate*distanzaordinate;
            covarianza += distanzaascisse*distanzaordinate;
        }
        varianzaascissecampionaria /= taglia-1;
        varianzaordinatecampionaria /= taglia-1;
        covarianza /= taglia-1;
        double determinante = varianzaascissecampionaria*varianzaordinatecampionaria-covarianza*covarianza;
        if (determinante < 0.00000001) {
            varianzaascissecampionaria += 0.0001;
            varianzaordinatecampionaria += 0.0001;
            determinante = varianzaascissecampionaria*varianzaordinatecampionaria-covarianza*covarianza;
        }
        const double soglia = bootstrapmobilehotelling(taglia, ascisse, ordinate, b, B, mediaascisse, mediaordinate, alfa);
        const double formaquadratica = (varianzaordinatecampionaria*mediaascisse*mediaascisse-2.0*covarianza*mediaascisse*
            mediaordinate+varianzaascissecampionaria*mediaordinate*mediaordinate)/determinante;
        if (formaquadratica <= soglia){accettazioni++;}
    }
    return 1.0-static_cast<double>(accettazioni)/volte;
}


void calcolabootstrapnorme(std::vector<double> &statistiche, const int thread, const int iterazioni, const int aggiuntive,
    const int taglia, const std::vector<double> &norme){
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::uniform_int_distribution<> uniforme(0, taglia-1);
    for (int i = 0; i < iterazioni+aggiuntive; i++){
        double media = 0.0;
        for (int j = 0; j < taglia; j++) {
            media += norme[uniforme(generatore)];
        }
        statistiche[thread*iterazioni+i] = media/taglia;
    }
}


std::array<double, 2> bootstrapnorme(const int taglia, const std::vector<double> norme,
    const int B, const double alfa){
    if (taglia < 3){throw std::runtime_error("Dataset troppo piccolo");}
    if (norme.size() != taglia){throw std::runtime_error("Dataset sbagliato");}
    if (B < 1){throw std::runtime_error("Iterazioni insufficienti");}
    if (alfa <= 0.0 || alfa >= 1.0){throw std::runtime_error("Alfa sbagliato");}
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = B/numerothread;
    const int iterazionirestanti = B%numerothread;
    std::vector<double> statistiche(B, 0.0);
    for (int i = 0; i < numerothread - 1; i++){
        int indicethread = i;
        thread[i] = std::thread([=, &statistiche](){
            calcolabootstrapnorme(statistiche, indicethread, volteperthread, 0, taglia, norme);
        });
    }
    thread[numerothread - 1] = std::thread([=, &statistiche](){
        calcolabootstrapnorme(statistiche, numerothread-1, volteperthread, iterazionirestanti,
            taglia, norme);
    });
    for (auto& t : thread){
        t.join();
    }
    std::sort(statistiche.begin(), statistiche.end());
    std::array<double, 2> intervallo = {0.0, 0.0};
    const double indice1 = B*alfa/2.0;
    const double indice2 = B*(1.0-alfa/2.0);
    const double peso1 = indice1-std::floor(indice1);
    const double peso2 = indice2-std::floor(indice2);
    const int indiceintero1 = static_cast<int>(indice1);
    const int indiceintero2 = static_cast<int>(indice2);
    if (indice2 >= B){intervallo[1] = statistiche[B-1];} else {intervallo[1] = (1.0-peso2)*statistiche[indiceintero2-1]+peso2*statistiche[indiceintero2];}
    if (indice1 < 1.0){intervallo[0] = statistiche[0];} else {intervallo[0] = (1.0-peso1)*statistiche[indiceintero1-1]+peso1*statistiche[indiceintero1];}
    return intervallo;
}


double alfaverobootstrapnorme(const int taglia, const double varianza, const int B, const double alfa,
                              const int volte) {
    if (volte <= 0 || varianza <= 0.0) {throw std::runtime_error("Errore");}
    int accettazioni = 0;
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normale(0.0, std::sqrt(varianza));
    for (int i = 0; i < volte; i++){
        std::vector<double> norme(taglia);
        double media = 0.0;
        for (int j = 0; j < taglia; j++) {
            const double norma = normale(generatore);
            media += norma;
            norme[j] = norma;
        }
        media /= taglia;
        const std::array<double, 2> stimaintervallare = bootstrapnorme(taglia, norme, B, alfa);
        if (0.0 > stimaintervallare[0] && 0.0 < stimaintervallare[1]){accettazioni++;}
    }
    return 1.0-static_cast<double>(accettazioni)/volte;
}


std::array<double, 5> detrenda(std::vector<double> ascisse, std::vector<double> ordinate,
    const double betaa1, const double betaa0, const double betao1, const double betao0, const int n){
    if (ascisse.size() != n || ordinate.size() != n || n < 3){throw std::runtime_error("Errore");}
    for (int i = 0; i < n; i++){
        ascisse[i] -= betaa1*ascisse[i]+betaa0;
        ordinate[i] -= betao1+ordinate[i]+betao0;
    }
    double mediaa = 0.0;
    double mediao = 0.0;
    for (int i = 0; i < n; i++){mediaa += ascisse[i]; mediao += ordinate[i];}
    mediaa /= n;
    mediao /= n;
    double varianzaa = 0.0;
    double varianzao = 0.0;
    double covarianza = 0.0;
    for (int i = 0; i < n; i++) {
        const double d1 = ascisse[i]-mediaa;
        const double d2 = ordinate[i] -mediao;
        varianzaa += d1*d1;
        varianzao += d2*d2;
        covarianza += d1*d2;
    }
    return {varianzaa/(n-1), varianzao/(n-1), covarianza/(n-1), mediaa, mediao};
}


void calcolabootstrapttest(const std::vector<double> norme, const double mu0, const int iterazioni, const int aggiuntive,
    const int n, std::vector<double> &statistiche, const int thread) {
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::uniform_int_distribution<> indice(0, n-1);
    for (int i = 0; i < iterazioni+aggiuntive; i++) {
        std::vector<double> campione(n);
        double media = 0.0;
        for (int j = 0; j < n; j++) {
            const double norma = norme[indice(generatore)];
            campione[j] = norma;
            media += norma;
        }
        media /= n;
        double varianza = 0.0;
        for (int j = 0; j < n; j++) {
            const double differenza = campione[j]-media;
            varianza += differenza*differenza;
        }
        varianza /= n-1;
        statistiche[iterazioni*thread+i] = (media-mu0)*std::sqrt(n/varianza);
    }
}


bool bootstrapttest(const std::vector<double> norme, const char nulla, const int B, const double mu0, const int n, const double alfa) {
    if (norme.size() != n || B < 1 || n < 2 || alfa <= 0.0 || alfa >= 1.0){throw std::runtime_error("Errore");}
    if (nulla != '<' && nulla != '=' && nulla != '>'){throw std::runtime_error("Errore");}
    double media = 0.0;
    for (int i = 0; i < n; i++){media += norme[i];}
    media /= n;
    double varianza = 0.0;
    for (int i = 0; i < n; i++){const double d = norme[i]-media; varianza += d*d;}
    varianza /= n-1;
    const double statistica = (media-mu0)*std::sqrt(n/varianza);
    std::vector<double> norme_(n);
    for (int i = 0; i < n; i++){norme_[i] = norme[i]-media+mu0;}
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = B/numerothread;
    const int iterazionirestanti = B%numerothread;
    std::vector<double> statistiche(B, 0.0);
    for (int i = 0; i < numerothread - 1; i++){
        int indicethread = i;
        thread[i] = std::thread([=, &statistiche](){
            calcolabootstrapttest(norme_, media, volteperthread, 0, n, statistiche, indicethread);
        });
    }
    thread[numerothread - 1] = std::thread([=, &statistiche](){
        calcolabootstrapttest(norme_, media, volteperthread, iterazionirestanti, n, statistiche, numerothread-1);
    });
    for (auto& t : thread){
        t.join();
    }
    std::sort(statistiche.begin(), statistiche.end());
    if (nulla == '<') {
        double indice = B*(1.0-alfa);
        const double peso = indice-std::floor(indice);
        if (indice < 1.0){indice = 1.0;}
        if (indice >= B){indice = B-1.0;}
        const int indiceintero = static_cast<int>(indice);
        return statistica <= (1.0-peso)*statistiche[indiceintero-1]+peso*statistiche[indiceintero];
    }
    if (nulla == '>') {
        double indice = B*alfa;
        const double peso = indice-std::floor(indice);
        if (indice < 1.0){indice = 1.0;}
        if (indice >= B){indice = B-1.0;}
        const int indiceintero = static_cast<int>(indice);
        return statistica >= (1.0-peso)*statistiche[indiceintero-1]+peso*statistiche[indiceintero];
    }
    if (nulla == '=') {
        double indice1 = B*alfa/2.0;
        double indice2 = B*(1.0-alfa/2.0);
        const double peso1 = indice1-std::floor(indice1);
        const double peso2 = indice2-std::floor(indice2);
        if (indice1 < 1.0){indice1 = 1.0;}
        if (indice2 >= B){indice2 = B-1.0;}
        const int indiceintero1 = static_cast<int>(indice1);
        const int indiceintero2 = static_cast<int>(indice2);
        return statistica >= (1.0-peso1)*statistiche[indiceintero1-1]+peso1*statistiche[indiceintero1] && statistica <= (1.0-peso2)*statistiche[indiceintero2-1]+peso2*statistiche[indiceintero2];
    }
    return true;
}


double potenzabootstrapttest(const char nulla, const int B, const double mu0, const int n, const double alfa, const int volte,
    const double media, const double varianza) {
    int accettazioni = 0;
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normale(media, std::sqrt(varianza));
    for (int i = 0; i < volte; i++) {
        std::vector<double> campione(n);
        for (int j = 0; j < n; j++){
            campione[j] = normale(generatore);
        }
        if (bootstrapttest(campione, nulla, B, mu0, n, alfa)){accettazioni++;}
    }
    return 1.0-static_cast<double>(accettazioni)/volte;
}


double varianzaneweywest(const std::vector<double> dati, const int n) {
    double media = 0.0;
    for (int i = 0; i < n; i++) {
        media += dati[i];
    }
    media /= n;
    double varianza = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = dati[i]-media;
        varianza += d*d;
    }
    varianza /= n;
    for (int i = 1; i < n; i++) {
        double autocovarianza = 0.0;
        for (int j = 0; j < n-i; j++) {
            autocovarianza += (dati[j]-media)*(dati[j+i]-media);
        }
        autocovarianza /= n;
        autocovarianza *= static_cast<double>(n-i)/n;
        varianza += 2.0*autocovarianza;
    }
    return varianza;
}


int calcolaalfaveroneweywest(const int n, const double mu0, const double sigma2, const int iterazioni,
    const double t, std::vector<double> &medie, std::vector<double> &neweywest) {
    int accettazioni = 0;
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> epsilon(0.0, std::sqrt(sigma2));
    for (int i = 0; i < iterazioni; i++){
        std::vector<double> campione(n);
        double precedente = 0.0;
        double media = 0.0;
        for (int j = 0; j < n; j++) {
            precedente = precedente*0.5+epsilon(generatore);
            campione[j] = precedente+mu0;
            media += precedente;
        }
        media += n*mu0;
        media /= n;
        double varianza = 0.0;
        for (int j = 0; j < n; j++) {
            const double d = campione[j]-media;
            varianza += d*d;
        }
        varianza /= n;
        for (int j = 1; j < n; j++) {
            double autocovarianza = 0.0;
            for (int k = 0; k < n-j; k++) {
                autocovarianza += (campione[k]-media)*(campione[k+j]-media);
            }
            autocovarianza /= n;
            autocovarianza *= static_cast<double>(n-j)/n;
            varianza += 2.0*autocovarianza;
        }
        medie[i] = media;
        neweywest[i] = varianza;
        if (mu0 > media-t*std::sqrt(varianza/n) && mu0 < media+t*std::sqrt(varianza/n)){accettazioni++;}
    }
    return accettazioni;
}


double alfaveroneweywest(const int n, const double mu0, const double sigma2, const int volte, const double t) {
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    int accettazioni = 0;
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = volte/numerothread;
    const int iterazionirestanti = volte%numerothread;
    std::vector<std::vector<double>> medie(numerothread-1, std::vector<double>(volteperthread));
    std::vector<std::vector<double>> neweywest(numerothread-1, std::vector<double>(volteperthread));
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazioni, &medie, &neweywest](){
            accettazioni += calcolaalfaveroneweywest(n, mu0, sigma2, volteperthread, t, medie[i], neweywest[i]);
        });
    }
    std::vector<double> fantoccio(volteperthread+iterazionirestanti);
    std::vector<double> spaventapasseri(volteperthread+iterazionirestanti);
    thread[numerothread - 1] = std::thread([=, &accettazioni, &fantoccio, &spaventapasseri](){
        accettazioni += calcolaalfaveroneweywest(n, mu0, sigma2, volteperthread+iterazionirestanti, t, fantoccio, spaventapasseri);
    });
    for (auto& th : thread){
        th.join();
    }
    double mediamedie = 0.0;
    for (int i = 0; i < numerothread-1; i++) {
        for (int j = 0; j < volteperthread; j++){mediamedie += medie[i][j];}
    }
    mediamedie /= volteperthread*(numerothread-1);
    double varianzamedie = 0.0;
    double mediavarianze = 0.0;
    for (int i = 0; i < numerothread-1; i++) {
        for (int j = 0; j < volteperthread; j++) {
            const double d = medie[i][j]-mediamedie;
            varianzamedie += d*d;
            mediavarianze += neweywest[i][j];
        }
    }
    varianzamedie /= volteperthread*(numerothread-1)+1;
    mediavarianze /= volteperthread*(numerothread-1);
    return 1.0-static_cast<double>(accettazioni)/volte;
}


int calcolapotenzaneweywest(const int n, const double mu0, const double sigma2, const int iterazioni,
        const double mediavera, const double t, const char nulla) {
    int accettazioni = 0;
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> epsilon(0.0, std::sqrt(sigma2));
    for (int i = 0; i < iterazioni; i++){
        std::vector<double> campione(n);
        double precedente = 0.0;
        double media = 0.0;
        for (int j = 0; j < n; j++) {
            precedente = precedente*0.5+epsilon(generatore);
            campione[j] = precedente+mediavera;
            media += precedente;
        }
        media += n*mediavera;
        media /= n;
        double varianza = 0.0;
        for (int j = 0; j < n; j++) {
            const double d = campione[j]-media;
            varianza += d*d;
        }
        for (int j = 1; j < n; j++) {
            double autocovarianza = 0.0;
            for (int k = 0; k < n-j; k++) {
                autocovarianza += (campione[k]-media)*(campione[k+j]-media);
            }
            autocovarianza *= static_cast<double>(n-j)/n;
            varianza += 2.0*autocovarianza;
        }
        varianza /= n-1;
        const double statistica = (media-mu0)*std::sqrt(n/varianza);
        if (nulla == '<' && statistica <= t){accettazioni++;}
        if (nulla == '=' && statistica >= -t && statistica <= t){accettazioni++;}
        if (nulla == '>' && statistica >= t){accettazioni++;}
    }
    return accettazioni;
}


double potenzaneweywest(const int n, const double mu0, const double sigma2, const int volte, const double media, const double t,
    const char nulla) {
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    int accettazioni = 0;
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = volte/numerothread;
    const int iterazionirestanti = volte%numerothread;
    for (int i = 0; i < numerothread - 1; i++){
        thread[i] = std::thread([=, &accettazioni](){
            accettazioni += calcolapotenzaneweywest(n, mu0, sigma2, volteperthread, media, t, nulla);
        });
    }
    thread[numerothread - 1] = std::thread([=, &accettazioni](){
        accettazioni += calcolapotenzaneweywest(n, mu0, sigma2, volteperthread+iterazionirestanti, media, t, nulla);
    });
    for (auto& th : thread){
        th.join();
    }
    return 1.0-static_cast<double>(accettazioni)/volte;
}


void calcolabootstrapvarianze(const std::vector<double> ascisse, const std::vector<double> ordinate,
    const double sigma01, const double sigma02, const int iterazioni, const int aggiuntive,
    const int n, std::vector<double> &statisticheascisse, std::vector<double> &statisticheordinate,
    std::vector<double> &varianzeascisse, std::vector<double> &varianzeordinate,
    std::vector<double> &correlazioni, const int thread) {
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::uniform_int_distribution<> indice(0, n-1);
    for (int i = 0; i < iterazioni+aggiuntive; i++) {
        std::vector<double> campioneascisse(n);
        std::vector<double> campioneordinate(n);
        double mediaascisse = 0.0;
        double mediaordinate = 0.0;
        for (int j = 0; j < n; j++) {
            const int dato = indice(generatore);
            const double ascissa = ascisse[dato];
            const double ordinata = ordinate[dato];
            campioneascisse[j] = ascissa;
            campioneordinate[j] = ordinata;
            mediaascisse += ascissa;
            mediaordinate += ordinata;
        }
        mediaascisse /= n;
        mediaordinate /= n;
        double varianzaascisse = 0.0;
        double varianzaordinate = 0.0;
        double covarianza = 0.0;
        for (int j = 0; j < n; j++) {
            const double differenzaascisse = campioneascisse[j]-mediaascisse;
            const double differenzaordinate = campioneordinate[j]-mediaordinate;
            varianzaascisse += differenzaascisse*differenzaascisse;
            varianzaordinate += differenzaordinate*differenzaordinate;
            covarianza += differenzaascisse*differenzaordinate;
        }
        varianzaascisse /= n-1;
        varianzaordinate /= n-1;
        covarianza /= n-1;
        const int posizione = iterazioni*thread+i;
        varianzeascisse[posizione] = varianzaascisse;
        varianzeordinate[posizione] = varianzaordinate;
        correlazioni[posizione] = covarianza/std::sqrt(varianzaascisse*varianzaordinate);
        statisticheascisse[posizione] = (n-1.0)*varianzaascisse/sigma01;
        statisticheordinate[posizione] = (n-1.0)*varianzaordinate/sigma02;
    }
}


std::array<double, 8> bootstrapvarianze(const std::vector<double> ascisse, const std::vector<double> ordinate,
    const char nulla1, const char nulla2, const int B, const double sigma01, const double sigma02,
    const int n, const double alfa1, const double alfa2, const double alfa3) {
    if (ascisse.size() != n || ordinate.size() != n || B < 1 || n < 2 || alfa1 <= 0.0 || alfa1 >= 1.0 || alfa2 <= 0.0 || alfa2 >= 1.0 || alfa3 <= 0.0 || alfa3 >= 1.0){throw std::runtime_error("Errore");}
    if (nulla1 != '<' && nulla1 != '=' && nulla1 != '>'){throw std::runtime_error("Errore");}
    if (nulla2 != '<' && nulla2 != '=' && nulla2 != '>'){throw std::runtime_error("Errore");}
    double mediaascisse = 0.0;
    double mediaordinate = 0.0;
    for (int i = 0; i < n; i++){mediaascisse += ascisse[i]; mediaordinate += ordinate[i];}
    mediaascisse /= n;
    mediaordinate /= n;
    double varianzaascisse = 0.0;
    double varianzaordinate = 0.0;
    for (int i = 0; i < n; i++){
        const double d1 = ascisse[i]-mediaascisse;
        const double d2 = ordinate[i]-mediaordinate;
        varianzaascisse += d1*d1;
        varianzaordinate += d2*d2;
    }
    varianzaascisse /= n-1;
    varianzaordinate /= n-1;
    const double statisticaascisse = (n-1.0)*varianzaascisse/sigma01;
    const double statisticaordinate = (n-1.0)*varianzaordinate/sigma02;
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = B/numerothread;
    const int iterazionirestanti = B%numerothread;
    std::vector<double> varianzeascisse(B, 0.0);
    std::vector<double> varianzeordinate(B, 0.0);
    std::vector<double> correlazioni(B, 0.0);
    std::vector<double> statisticheascisse(B, 0.0);
    std::vector<double> statisticheordinate(B, 0.0);
    for (int i = 0; i < numerothread - 1; i++){
        int indicethread = i;
        thread[i] = std::thread([=, &varianzeascisse, &varianzeordinate, &correlazioni, &statisticheascisse, &statisticheordinate](){
            calcolabootstrapvarianze(ascisse, ordinate, varianzaascisse, varianzaordinate, volteperthread, 0, n,
                statisticheascisse, statisticheordinate, varianzeascisse, varianzeordinate, correlazioni, indicethread);
        });
    }
    thread[numerothread - 1] = std::thread([=, &varianzeascisse, &varianzeordinate, &correlazioni, &statisticheascisse, &statisticheordinate](){
        calcolabootstrapvarianze(ascisse, ordinate, varianzaascisse, varianzaordinate, volteperthread, iterazionirestanti, n,
            statisticheascisse, statisticheordinate, varianzeascisse, varianzeordinate, correlazioni, numerothread-1);
    });
    for (auto& t : thread){
        t.join();
    }
    std::sort(varianzeascisse.begin(), varianzeascisse.end());
    std::sort(varianzeordinate.begin(), varianzeordinate.end());
    std::sort(correlazioni.begin(), correlazioni.end());
    std::sort(statisticheascisse.begin(), statisticheascisse.end());
    std::sort(statisticheordinate.begin(), statisticheordinate.end());
    double indice1 = B*alfa1/2.0;
    const double peso1 = indice1-std::floor(indice1);
    if (indice1 < 1.0){indice1 = 1.0;}
    if (indice1 >= B){indice1 = B-1.0;}
    const int indiceintero1 = static_cast<int>(indice1);
    double indice4 = B*(1.0-alfa1/2.0);
    const double peso4 = indice4-std::floor(indice4);
    if (indice4 < 1.0){indice4 = 1.0;}
    if (indice4 >= B){indice4 = B-1.0;}
    const int indiceintero4 = static_cast<int>(indice4);
    double indice5 = B*alfa2/2.0;
    const double peso5 = indice5-std::floor(indice5);
    if (indice5 < 1.0){indice5 = 1.0;}
    if (indice5 >= B){indice5 = B-1.0;}
    const int indiceintero5 = static_cast<int>(indice5);
    double indice8 = B*(1.0-alfa2/2.0);
    const double peso8 = indice8-std::floor(indice8);
    if (indice8 < 1.0){indice8 = 1.0;}
    if (indice8 >= B){indice8 = B-1.0;}
    const int indiceintero8 = static_cast<int>(indice8);
    double indice9 = B*alfa3/2.0;
    const double peso9 = indice9-std::floor(indice9);
    if (indice9 < 1.0){indice9 = 1.0;}
    if (indice9 >= B){indice9 = B-1.0;}
    const int indiceintero9 = static_cast<int>(indice9);
    double indice10 = B*(1.0-alfa3/2.0);
    const double peso10 = indice10-std::floor(indice10);
    if (indice10 < 1.0){indice10 = 1.0;}
    if (indice10 >= B){indice10 = B-1.0;}
    const int indiceintero10 = static_cast<int>(indice10);
    const double l1 = (1.0-peso1)*varianzeascisse[indiceintero1-1]+peso1*varianzeascisse[indiceintero1];
    const double u1 = (1.0-peso4)*varianzeascisse[indiceintero4-1]+peso4*varianzeascisse[indiceintero4];
    const double l2 = (1.0-peso5)*varianzeordinate[indiceintero5-1]+peso5*varianzeordinate[indiceintero5];
    const double u2 = (1.0-peso8)*varianzeordinate[indiceintero8-1]+peso8*varianzeordinate[indiceintero8];
    const double l3 = (1.0-peso9)*correlazioni[indiceintero9-1]+peso9*correlazioni[indiceintero9];
    const double u3 = (1.0-peso10)*correlazioni[indiceintero10-1]+peso10*correlazioni[indiceintero10];
    int spartiacque1 = B;
    for (int i = 0; i < B; i++){if (statisticaascisse < statisticheascisse[i]){spartiacque1 = i; break;}}
    int spartiacque2 = B;
    for (int i = 0; i < B; i++){if (statisticaordinate < statisticheordinate[i]){spartiacque2 = i; break;}}
    double p1;
    double p2;
    if (nulla1 == '<') {p1 = static_cast<double>(B-spartiacque1)/B;}
    if (nulla1 == '>') {p1 = static_cast<double>(spartiacque1)/B;}
    if (nulla1 == '=') {if (spartiacque1 <= B/2){p1 = 2.0*static_cast<double>(spartiacque1)/B;} else {p1 = 2.0*static_cast<double>(B-spartiacque1)/B;}}
    if (nulla2 == '<') {p2 = static_cast<double>(B-spartiacque2)/B;}
    if (nulla2 == '>') {p2 = static_cast<double>(spartiacque2)/B;}
    if (nulla2 == '=') {if (spartiacque2 <= B/2){p2 = 2.0*static_cast<double>(spartiacque2)/B;} else {p2 = 2.0*static_cast<double>(B-spartiacque2)/B;}}
    return {l1, u1, l2, u2, l3, u3, p1, p2};
}


std::array<double, 5> potenzabootstrapvarianze(const char nulla1, const char nulla2, const int B,
    const double sigma01, const double sigma02, const int n, const double alfa1, const double alfa2,
    const double alfa3, const int volte, const double varianza1, const double varianza2,
    const double correlazione, const double alfai1, const double alfai2) {
    int accettazioni1 = 0;
    int accettazioni2 = 0;
    int accettazioni3 = 0;
    int accettazioni4 = 0;
    int accettazioni5 = 0;
    const double covarianza = correlazione*std::sqrt(varianza1*varianza2);
    const double cholesky1 = std::sqrt(varianza1);
    const double cholesky2 = covarianza/cholesky1;
    const double cholesky3 = std::sqrt(varianza2-cholesky2*cholesky2);
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normale1(0.0, 1.0);
    std::normal_distribution<> normale2(0.0, 1.0);
    for (int i = 0; i < volte; i++) {
        std::vector<double> campioneascisse(n);
        std::vector<double> campioneordinate(n);
        for (int j = 0; j < n; j++){
            const double base = normale1(generatore);
            campioneascisse[j] = base*cholesky1;
            campioneordinate[j] = base*cholesky2+normale2(generatore)*cholesky3;
        }
        std::array<double, 8> risultato = bootstrapvarianze(campioneascisse, campioneordinate, nulla1, nulla2, B,
            sigma01, sigma02, n, alfa1, alfa2, alfa3);
        if (risultato[0] <= varianza1 && risultato[1] >= varianza1){accettazioni1++;}
        if (risultato[2] <= varianza2 && risultato[3] >= varianza2){accettazioni2++;}
        if (risultato[4] <= correlazione && risultato[5] >= correlazione){accettazioni3++;}
        if (risultato[6] >= alfai1){accettazioni4++;}
        if (risultato[7] >= alfai2){accettazioni5++;}
    }
    const double v = static_cast<double>(volte);
    return {accettazioni1/v, accettazioni2/v, accettazioni3/v, accettazioni4/v, accettazioni5/v};
}


void calcolabootstrapmobilevarianze(const std::vector<double> ascisse, const std::vector<double> ordinate,
    const double sigma01, const double sigma02, const int iterazioni, const int aggiuntive,
    const int n, std::vector<double> &statisticheascisse, std::vector<double> &statisticheordinate,
    std::vector<double> &varianzeascisse, std::vector<double> &varianzeordinate,
    std::vector<double> &correlazioni, const int thread, const int b) {
    const int k = n/b+1;
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::uniform_int_distribution<> indice(0, n-b+1);
    for (int i = 0; i < iterazioni+aggiuntive; i++) {
        std::vector<double> campioneascisse(k*b);
        std::vector<double> campioneordinate(k*b);
        double mediaascisse = 0.0;
        double mediaordinate = 0.0;
        for (int j = 0; j < k; j++) {
            const int dato = indice(generatore);
            for (int l = 0; l < b; l++){
                const double ascissa = ascisse[dato+l];
                const double ordinata = ordinate[dato+l];
                campioneascisse[j*b+l] = ascissa;
                campioneordinate[j*b+l] = ordinata;
                mediaascisse += ascissa;
                mediaordinate += ordinata;
            }
        }
        mediaascisse /= n;
        mediaordinate /= n;
        double varianzaascisse = 0.0;
        double varianzaordinate = 0.0;
        double covarianza = 0.0;
        for (int j = 0; j < n; j++) {
            const double differenzaascisse = campioneascisse[j]-mediaascisse;
            const double differenzaordinate = campioneordinate[j]-mediaordinate;
            varianzaascisse += differenzaascisse*differenzaascisse;
            varianzaordinate += differenzaordinate*differenzaordinate;
            covarianza += differenzaascisse*differenzaordinate;
        }
        varianzaascisse /= n-1;
        varianzaordinate /= n-1;
        covarianza /= n-1;
        const int posizione = iterazioni*thread+i;
        varianzeascisse[posizione] = varianzaascisse;
        varianzeordinate[posizione] = varianzaordinate;
        correlazioni[posizione] = covarianza/std::sqrt(varianzaascisse*varianzaordinate);
        statisticheascisse[posizione] = (n-1.0)*varianzaascisse/sigma01;
        statisticheordinate[posizione] = (n-1.0)*varianzaordinate/sigma02;
    }
}


std::array<double, 8> bootstrapmobilevarianze(const std::vector<double> ascisse, const std::vector<double> ordinate,
    const char nulla1, const char nulla2, const int B, const double sigma01, const double sigma02,
    const int n, const double alfa1, const double alfa2, const double alfa3, const int b) {
    if (ascisse.size() != n || ordinate.size() != n || B < 1 || b < 1 || n < 2 || alfa1 <= 0.0 || alfa1 >= 1.0 || alfa2 <= 0.0 || alfa2 >= 1.0 || alfa3 <= 0.0 || alfa3 >= 1.0){throw std::runtime_error("Errore");}
    if (nulla1 != '<' && nulla1 != '=' && nulla1 != '>'){throw std::runtime_error("Errore");}
    if (nulla2 != '<' && nulla2 != '=' && nulla2 != '>'){throw std::runtime_error("Errore");}
    double mediaascisse = 0.0;
    double mediaordinate = 0.0;
    for (int i = 0; i < n; i++){mediaascisse += ascisse[i]; mediaordinate += ordinate[i];}
    mediaascisse /= n;
    mediaordinate /= n;
    double varianzaascisse = 0.0;
    double varianzaordinate = 0.0;
    for (int i = 0; i < n; i++){
        const double d1 = ascisse[i]-mediaascisse;
        const double d2 = ordinate[i]-mediaordinate;
        varianzaascisse += d1*d1;
        varianzaordinate += d2*d2;
    }
    varianzaascisse /= n-1;
    varianzaordinate /= n-1;
    const double statisticaascisse = (n-1.0)*varianzaascisse/sigma01;
    const double statisticaordinate = (n-1.0)*varianzaordinate/sigma02;
    int numerothread = static_cast<int>(std::thread::hardware_concurrency());
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = B/numerothread;
    const int iterazionirestanti = B%numerothread;
    std::vector<double> varianzeascisse(B, 0.0);
    std::vector<double> varianzeordinate(B, 0.0);
    std::vector<double> correlazioni(B, 0.0);
    std::vector<double> statisticheascisse(B, 0.0);
    std::vector<double> statisticheordinate(B, 0.0);
    for (int i = 0; i < numerothread - 1; i++){
        int indicethread = i;
        thread[i] = std::thread([=, &varianzeascisse, &varianzeordinate, &correlazioni, &statisticheascisse, &statisticheordinate](){
            calcolabootstrapmobilevarianze(ascisse, ordinate, varianzaascisse, varianzaordinate, volteperthread, 0, n,
                statisticheascisse, statisticheordinate, varianzeascisse, varianzeordinate, correlazioni, indicethread, b);
        });
    }
    thread[numerothread - 1] = std::thread([=, &varianzeascisse, &varianzeordinate, &correlazioni, &statisticheascisse, &statisticheordinate](){
        calcolabootstrapmobilevarianze(ascisse, ordinate, varianzaascisse, varianzaordinate, volteperthread, iterazionirestanti, n,
            statisticheascisse, statisticheordinate, varianzeascisse, varianzeordinate, correlazioni, numerothread-1, b);
    });
    for (auto& t : thread){
        t.join();
    }
    std::sort(varianzeascisse.begin(), varianzeascisse.end());
    std::sort(varianzeordinate.begin(), varianzeordinate.end());
    std::sort(correlazioni.begin(), correlazioni.end());
    std::sort(statisticheascisse.begin(), statisticheascisse.end());
    std::sort(statisticheordinate.begin(), statisticheordinate.end());
    double indice1 = B*alfa1/2.0;
    const double peso1 = indice1-std::floor(indice1);
    if (indice1 < 1.0){indice1 = 1.0;}
    if (indice1 >= B){indice1 = B-1.0;}
    const int indiceintero1 = static_cast<int>(indice1);
    double indice4 = B*(1.0-alfa1/2.0);
    const double peso4 = indice4-std::floor(indice4);
    if (indice4 < 1.0){indice4 = 1.0;}
    if (indice4 >= B){indice4 = B-1.0;}
    const int indiceintero4 = static_cast<int>(indice4);
    double indice5 = B*alfa2/2.0;
    const double peso5 = indice5-std::floor(indice5);
    if (indice5 < 1.0){indice5 = 1.0;}
    if (indice5 >= B){indice5 = B-1.0;}
    const int indiceintero5 = static_cast<int>(indice5);
    double indice8 = B*(1.0-alfa2/2.0);
    const double peso8 = indice8-std::floor(indice8);
    if (indice8 < 1.0){indice8 = 1.0;}
    if (indice8 >= B){indice8 = B-1.0;}
    const int indiceintero8 = static_cast<int>(indice8);
    double indice9 = B*alfa3/2.0;
    const double peso9 = indice9-std::floor(indice9);
    if (indice9 < 1.0){indice9 = 1.0;}
    if (indice9 >= B){indice9 = B-1.0;}
    const int indiceintero9 = static_cast<int>(indice9);
    double indice10 = B*(1.0-alfa3/2.0);
    const double peso10 = indice10-std::floor(indice10);
    if (indice10 < 1.0){indice10 = 1.0;}
    if (indice10 >= B){indice10 = B-1.0;}
    const int indiceintero10 = static_cast<int>(indice10);
    const double l1 = (1.0-peso1)*varianzeascisse[indiceintero1-1]+peso1*varianzeascisse[indiceintero1];
    const double u1 = (1.0-peso4)*varianzeascisse[indiceintero4-1]+peso4*varianzeascisse[indiceintero4];
    const double l2 = (1.0-peso5)*varianzeordinate[indiceintero5-1]+peso5*varianzeordinate[indiceintero5];
    const double u2 = (1.0-peso8)*varianzeordinate[indiceintero8-1]+peso8*varianzeordinate[indiceintero8];
    const double l3 = (1.0-peso9)*correlazioni[indiceintero9-1]+peso9*correlazioni[indiceintero9];
    const double u3 = (1.0-peso10)*correlazioni[indiceintero10-1]+peso10*correlazioni[indiceintero10];
    int spartiacque1 = B;
    for (int i = 0; i < B; i++){if (statisticaascisse < statisticheascisse[i]){spartiacque1 = i; break;}}
    int spartiacque2 = B;
    for (int i = 0; i < B; i++){if (statisticaordinate < statisticheordinate[i]){spartiacque2 = i; break;}}
    double p1;
    double p2;
    if (nulla1 == '<') {p1 = static_cast<double>(B-spartiacque1)/B;}
    if (nulla1 == '>') {p1 = static_cast<double>(spartiacque1)/B;}
    if (nulla1 == '=') {if (spartiacque1 <= B/2){p1 = 2.0*static_cast<double>(spartiacque1)/B;} else {p1 = 2.0*static_cast<double>(B-spartiacque1)/B;}}
    if (nulla2 == '<') {p2 = static_cast<double>(B-spartiacque2)/B;}
    if (nulla2 == '>') {p2 = static_cast<double>(spartiacque2)/B;}
    if (nulla2 == '=') {if (spartiacque2 <= B/2){p2 = 2.0*static_cast<double>(spartiacque2)/B;} else {p2 = 2.0*static_cast<double>(B-spartiacque2)/B;}}
    return {l1, u1, l2, u2, l3, u3, p1, p2};
}


std::array<double, 5> potenzabootstrapmobilevarianze(const char nulla1, const char nulla2, const int B,
    const double sigma01, const double sigma02, const int n, const double alfa1, const double alfa2,
    const double alfa3, const int volte, const double varianza1, const double varianza2,
    const double correlazione, const double alfai1, const double alfai2, const int b) {
    int accettazioni1 = 0;
    int accettazioni2 = 0;
    int accettazioni3 = 0;
    int accettazioni4 = 0;
    int accettazioni5 = 0;
    const double covarianza = correlazione*std::sqrt(varianza1*varianza2);
    const double cholesky1 = std::sqrt(varianza1);
    const double cholesky2 = covarianza/cholesky1;
    const double cholesky3 = std::sqrt(varianza2-cholesky2*cholesky2);
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normale1(0.0, 1.0);
    std::normal_distribution<> normale2(0.0, 1.0);
    for (int i = 0; i < volte; i++) {
        std::vector<double> campioneascisse(n);
        std::vector<double> campioneordinate(n);
        for (int j = 0; j < n; j++){
            const double base = normale1(generatore);
            campioneascisse[j] = base*cholesky1;
            campioneordinate[j] = base*cholesky2+normale2(generatore)*cholesky3;
        }
        std::array<double, 8> risultato = bootstrapmobilevarianze(campioneascisse, campioneordinate, nulla1, nulla2, B,
            sigma01, sigma02, n, alfa1, alfa2, alfa3, b);
        if (risultato[0] <= varianza1 && risultato[1] >= varianza1){accettazioni1++;}
        if (risultato[2] <= varianza2 && risultato[3] >= varianza2){accettazioni2++;}
        if (risultato[4] <= correlazione && risultato[5] >= correlazione){accettazioni3++;}
        if (risultato[6] >= alfai1){accettazioni4++;}
        if (risultato[7] >= alfai2){accettazioni5++;}
    }
    const double v = static_cast<double>(volte);
    return {accettazioni1/v, accettazioni2/v, accettazioni3/v, accettazioni4/v, accettazioni5/v};
}


// Registrala sotto
std::array<double, 2> regressionecircolare(const std::vector<double> &angoli, const std::vector<double> &covariata,
    const int n) {
    if (angoli.size() != n || covariata.size() != n){throw std::runtime_error("Dati sbagliati per la regressione circolare");}
    double sommacoseni = 0.0;
    double sommaseni = 0.0;
    for (int i = 0; i < n; i++) {
        sommacoseni += std::cos(angoli[i]);
        sommaseni += std::sin(angoli[i]);
    }
    double intercetta = std::atan2(sommaseni/n, sommacoseni/n);
    double inclinazione = 0.0;
    for (int it = 0; it <= 10000; it++){
        if (it == 10000){throw std::runtime_error("Non sono riuscito a convergere");}
        double scarticoseni = 0.0;
        double scartiseni = 0.0;
        for (int i = 0; i < n; i++) {
            const double argomento = angoli[i]-2.0*std::atan(intercetta+inclinazione*covariata[i]);
            scarticoseni += std::cos(argomento);
            scartiseni += std::sin(argomento);
        }
        scarticoseni /= n;
        scartiseni /= n;
        const double raggio = std::sqrt(scarticoseni*scarticoseni+scartiseni*scartiseni);
        const double mu = std::atan2(scartiseni, scarticoseni);
        double sommag = 0.0;
        double sommatg = 0.0;
        double sommat2g = 0.0;
        double sommawg = 0.0;
        double sommawtg = 0.0;
        for (int i = 0; i < n; i++){
            const double tempo = covariata[i];
            const double previsione = intercetta+inclinazione*tempo;
            const double gi = 2.0/(1.0+previsione*previsione);
            const double ui = std::sin(angoli[i]-mu-2.0*std::atan(previsione));
            const double wi = ui*(1.0+previsione*previsione)/(2.0*raggio);
            const double tg = tempo*gi;
            sommag += gi;
            sommatg += tg;
            sommat2g += tempo*tg;
            sommawg += wi*gi;
            sommawtg += wi*tg;
        }
        const double determinante = sommag*sommat2g-sommatg*sommatg;
        intercetta = (sommat2g*sommawg-sommatg*sommawtg)/determinante;
        inclinazione = (sommag*sommawtg-sommawg*sommatg)/determinante;
    }
    return {inclinazione, intercetta};
}

PYBIND11_MODULE(miomodulo, m) {
    py::class_<CoseTest>(m, "CoseTest")
        .def(py::init<>())
        .def_readwrite("statistica", &CoseTest::statistica)
        .def_readwrite("accettazione", &CoseTest::accettazione)
        .def_readwrite("pvalue", &CoseTest::pvalue);
    m.def("autocorrelazioniangolari", &autocorrelazioniangolari, "Funzione per calcolare le autocorrelazioni degli angoli.");
    m.def("testhotelling", &testhotelling, "Funzione per eseguire il test di Hotelling.");
    m.def("betatesthotelling",
        [](const int t, const double s, const int v, const double d, const double va, const double vo){py::gil_scoped_release release; return betatesthotelling(t, s, v, d, va, vo);},
        "Funzione per calcolare la probabilita di errore di II tipo del test di Hotelling.");
    m.def("ljungbox", &ljungbox, "Funzione per eseguire il test di Ljung-Box.");
    m.def("betaljungbox",
        [](const int t, const std::vector<int> &c, const std::vector<double> &s, const int n, const int i){py::gil_scoped_release release; return betaljungbox(t, c, s, n, i);},
        "Funzione per calcolare la probabilita di errore di II tipo del test di Ljung-Box.");
    m.def("alfaveroljungbox",
        [](const int t, const std::vector<int> &c, const std::vector<double> &s, const int n, const int i){py::gil_scoped_release release; return alfaveroljungbox(t, c, s, n, i);},
        "Funzione per calcolare la probabilita di errore di I tipo effettiva del test di Ljung-Box.");
    m.def("testhenzezirkler", &testhenzezirkler, "Funzione per eseguire il test di Henze-Zirkler restituendone le statistiche rilevanti.");
    m.def("testmardia", &testmardia, "Funzione per eseguire il test di Mardia.");
    m.def("betatesthenzezirkler",
        [](const char d, const int t, const int i, const int g, const double aa, const double ao, const double dc){py::gil_scoped_release release; return betatesthenzezirkler(d, t, i, g, aa, ao, dc);},
        "Funzione per calcolare la probabilita di errore di II tipo del test di Henze-Zirkler.");
    m.def("betatestmardia",
        [](const int i, const int t, const char d, const double aa, const double ao, const double dc, const int g, const double sa, const double sc){py::gil_scoped_release release; return betatestmardia(i, t, d, aa, ao, dc, g, sa, sc);},
        "Funzione per calcolare la probabilita di errore di II tipo del test di Mardia.");
    m.def("alfaverohenzezirkler",
        [](const int i, const int t){py::gil_scoped_release release; return alfaverohenzezirkler(i, t);},
        "Funzione per calcolare la probabilita di errore di I tipo effettiva del test di Henze-Zirkler.");
    m.def("alfaveromardia",
        [](const int i, const int t, const double sa, const double sc){py::gil_scoped_release release; return alfaveromardia(i, t, sa, sc);},
        "Funzione per calcolare la probabilita di errore di I tipo effettiva del test di Mardia.");
    m.def("clustering",
        [](const char i, const int t, const std::vector<double> &a, const std::vector<double> &o, const int ik, const int cm, const char s, const int cb, const double ab, const int ii, const int ie, const bool n, const char al, const double sc, const char cs, const int bi, int f){py::gil_scoped_release release; return clustering(i, t, a, o, ik, cm, s, cb, ab, ii, ie, n, al, sc, cs, bi, f);},
        "Funzione per effettuare il clustering. Restituisce una lista di etichette.");
    m.def("betatestnorme",
        [](const int t, const bool ma, const bool du, const int i, const double me, const double dt, const double v, const double s){py::gil_scoped_release release; return betatestnorme(t, ma, du, i, me, dt, v, s);},
        "Funzione per calcolare la probabilita di errore di II tipo del test sulle norme.");
    m.def("betatestvarianze",
        [](const int i, const int t, const bool ma, const bool du, const double dt, const double s, const double ss, const double in){py::gil_scoped_release release; return betatestvarianze(i, t, ma, du, dt, s, ss, in);},
        "Funzione per calcolare la probabilita di errore di II tipo del test sulle varianze.");
    m.def("freccedifettose", &freccedifettose, "Funzione per calcolare se ogni freccia è difettosa.");
    m.def("betahotellingduecampioni",
        [](const int i, const double va, const double vo, const double c, const int t, const double d, const int n, const double s){py::gil_scoped_release release; return betahotellingduecampioni(i, va, vo, c, t, d, n, s);},
        "Funzione per calcolare la probabilita di errore di II tipo del test di Hotelling a due campioni.");
    m.def("betavarianzeduecampioni",
        [](const int i, const double v, const int t, const int n, const double s){py::gil_scoped_release release; return betavarianzeduecampioni(i, v, t, n, s);},
        "Funzione per calcolare la probabilita di errore di II tipo di un test a due campioni sulle varianze.");
    m.def("testvonmises", &testvonmises, "Funzione per testare se gli angoli vengono da una distribuzione di von Mises.");
    m.def("alfaverointervalloangolomedio",
        [](const int i, const double k, const int t, const bool f, const double q){py::gil_scoped_release release; return alfaverointervalloangolomedio(i, k, t, f, q);},
        "Funzione per calcolare il livello di confidenza effettivo dell'intervallo sull'angolo medio.");
    m.def("angolomediobayesiano", &angolomediobayesiano, "Funzione per calcolare con un paradigma bayesiano l'angolo medio.");
    m.def("misturevonmisesvariazionali", &misturevonmisesvariazionali, "Funzione per calcolare i parametri di modello mistura von Mises usando l'inferenza variazionale.");
    m.def("betarayleigh",
        [](const int i, const double c, const int t, const double s){py::gil_scoped_release release; return betarayleigh(i, c, t, s);},
        "Funzione per calcolare la probabilità di errore di II tipo del test di Rayleigh.");
    m.def("alfaverorayleigh",
        [](const int i, const double s, const int t){py::gil_scoped_release release; return alfaverorayleigh(i, s, t);},
        "Funzione per calcolare l'errore di I tipo effettivo del test di Rayleigh.");
    m.def("betaaffidabilitavonmises",
        [](const int i, const bool u, const int t, const double s, const double d, const double k){py::gil_scoped_release release; return betaaffidabilitavonmises(i, u, t, s, d, k);},
        "Funzione per calcolare l'errore di II tipo del test di goodness of fit della von Mises.");
    m.def("alfaveroaffidabilitavonmises",
        [](const int i, const double k, const int t, const double s){py::gil_scoped_release release; return alfaveroaffidabilitavonmises(i, k, t, s);},
        "Funzione per calcolare l'errore di I tipo effettivo del test di goodness of fit della von Mises.");
    m.def("intervallokappa", &intervallokappa, "Funzione per calcolare l'intervallo di confidenza del parametro di concentrazione degli angoli.");
    m.def("alfaverointervallokappa",
        [](const int i, const double a, const double k, const int ib, const int t){py::gil_scoped_release release; return alfaverointervallokappa(i, a, k, ib, t);},
        "Funzione per calcolare l'errore di I tipo effettivo dell'intervallo bootstrap per il parametro di concentrazione.");
    m.def("bootstrapstazionariohotelling",
        [](const int t, const std::vector<double> a, const std::vector<double> o, const double p, const int B, const double ma, const double mo, const double al){py::gil_scoped_release release; return bootstrapstazionariohotelling(t, a, o, p, B, ma, mo, al);},
        "Funzione per calcolare la quantità Q che divisa per n viene usata per costruire un ellisse di confidenza robusto alle dipendenze usando il bootstrap stazionario.");
    m.def("alfaverobootstrapstazionariohotelling",
        [](const int t, const double va, const double vo, const double p, const int B, const double a, const int v){py::gil_scoped_release release; return alfaverobootstrapstazionariohotelling(t, va, vo, p, B, a, v);},
        "Funzione per calcolare la probabilità effettiva di copertura di un intervallo di Hotelling usando il bootstrap stazionario.");
    m.def("bootstrapmobilehotelling",
        [](const int t, const std::vector<double> a, const std::vector<double> o, const int b, const int B, const double ma, const double mo, const double al){py::gil_scoped_release release; return bootstrapmobilehotelling(t, a, o, b, B, ma, mo, al);},
        "Funzione per calcolare la quantità Q che divisa per n viene usata per costruire un ellisse di confidenza robusto alle dipendenze usando il bootstrap a blocchi mobili.");
    m.def("alfaverobootstrapmobilehotelling",
        [](const int t, const double va, const double vo, const int b, const int B, const double a, const int v){py::gil_scoped_release release; return alfaverobootstrapmobilehotelling(t, va, vo, b, B, a, v);},
        "Funzione per calcolare la probabilità effettiva di copertura di un intervallo di Hotelling usando il bootstrap a blocchi mobili.");
    m.def("bootstrapnorme",
        [](const int t, const std::vector<double> n, const int B, const double a){py::gil_scoped_release release; return bootstrapnorme(t, n, B, a);},
        "Funzione per calcolare una stima intervallare sulle norme per bootstrap.");
    m.def("alfaverobootstrapnorme",
        [](const int t, const double va, const int B, const double a, const int vo){py::gil_scoped_release release; return alfaverobootstrapnorme(t, va, B, a, vo);},
        "Funzione per stimare l'alfa effettivo della stima intervallare per bootstrap.");
    m.def("detrenda",
        [](std::vector<double> a, std::vector<double> o, const double ba1, const double ba0, const double bo1, const double bo0, const int n){py::gil_scoped_release release; return detrenda(a, o, ba1, ba0, bo1, bo0, n);},
        "Funzione per calcolare la media e la matrice di covarianze dei residui di un dataset.");
    m.def("bootstrapttest",
        [](const std::vector<double> no, const char nu, const int B, const double mu, const int n, const double a){py::gil_scoped_release release; return bootstrapttest(no, nu, B, mu, n, a);},
        "Funzione per eseguire un t-test bootstrap.");
    m.def("potenzabootstrapttest",
        [](const char nu, const int B, const double mu, const int n, const double a, const int v, const double me, const double va){py::gil_scoped_release release; return potenzabootstrapttest(nu, B, mu, n, a, v, me, va);},
        "Funzione per stimare la potenza del t-test bootstrap.");
    m.def("alfaveroneweywest",
        [](const int n, const double mu, const double s, const int v, const double t){py::gil_scoped_release release; return alfaveroneweywest(n, mu, s, v, t);},
        "Funzione per stimare la vera probabilità di copertura di un intervallo-t con varianza di Newey-West.");
    m.def("potenzaneweywest",
        [](const int n, const double mu, const double s, const int v, const double me, const double t, const char nu){py::gil_scoped_release release; return potenzaneweywest(n, mu, s, v, me, t, nu);},
        "Funzione per stimare la potenza di un t-test con varianza di Newey-West.");
    m.def("varianzaneweywest",
        [](const std::vector<double> d, const int n){py::gil_scoped_release release; return varianzaneweywest(d, n);},
        "Funzione per calcolare lo stimatore di Newey-West per la varianza univariata.");
    m.def("bootstrapvarianze",
        [](const std::vector<double> a, const std::vector<double> o, const char n1, const char n2, const int B, const double s1, const double s2, const int n, const double a1, const double a2, const double a3){py::gil_scoped_release release; return bootstrapvarianze(a, o, n1, n2, B, s1, s2, n, a1, a2, a3);},
        "Funzione per eseguire stime sulle varianze con il bootstrap.");
    m.def("potenzabootstrapvarianze",
        [](const char n1, const char n2, const int B, const double s1, const double s2, const int n, const double a1, const double a2, const double a3, const int vo, const double v1, const double v2, const double c, const double ai1, const double ai2) {py::gil_scoped_release release; return potenzabootstrapvarianze(n1, n2, B, s1, s2, n, a1, a2, a3, vo, v1, v2, c, ai1, ai2);},
        "Funzione per stimare l'accuratezza delle procedure bootstrap nella stima delle varianze.");
    m.def("bootstrapmobilevarianze",
        [](const std::vector<double> a, const std::vector<double> o, const char n1, const char n2, const int B, const double s1, const double s2, const int n, const double a1, const double a2, const double a3, const int b){py::gil_scoped_release release; return bootstrapmobilevarianze(a, o, n1, n2, B, s1, s2, n, a1, a2, a3, b);},
        "Funzione per eseguire stime sulle varianze con il bootstrap.");
    m.def("potenzabootstrapmobilevarianze",
        [](const char n1, const char n2, const int B, const double s1, const double s2, const int n, const double a1, const double a2, const double a3, const int vo, const double v1, const double v2, const double c, const double ai1, const double ai2, const int b) {py::gil_scoped_release release; return potenzabootstrapmobilevarianze(n1, n2, B, s1, s2, n, a1, a2, a3, vo, v1, v2, c, ai1, ai2, b);},
        "Funzione per stimare l'accuratezza delle procedure bootstrap nella stima delle varianze.");
    m.def("regressionecircolare", &regressionecircolare, "Funzione per eseguire una regressione circolare.");
}
