#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <array>
#include <cmath>
#include <thread>
#include <vector>
#include <random>
#include <string>
#include <algorithm>

namespace py = pybind11;


bool testhotelling(const int taglia, const py::array_t<double>& allenamento, const double soglia){
	const auto buffer = allenamento.request();
	const auto* pointer = static_cast<double*>(buffer.ptr);
	std::array<double, 2> media = {0.0, 0.0};
	for (int i = 0; i < taglia; i++){
	    for (int j = 0; j < 2; j++){
	        media[j] += pointer[i*2+j];
	    }
	}
	media[0] /= taglia;
	media[1] /= taglia;
	double varianzaascisse = 0.0;
	for (int i = 0; i < taglia; i += 2){
	    varianzaascisse += std::pow(pointer[i]-media[0], 2);
	}
	varianzaascisse /= taglia - 1;
	double varianzaordinate = 0.0;
	for (int i = 1; i < taglia; i += 2){
	    varianzaordinate += std::pow(pointer[i]-media[1], 2);
	}
	varianzaordinate /= taglia - 1;
	double covarianza = 0.0;
	for (int i = 0; i < taglia/2; i++){
	    covarianza += (pointer[i*2]-media[0])*(pointer[i*2+1]-media[1]);
	}
	covarianza /= taglia - 1;
	double determinante = (varianzaascisse*varianzaordinate-covarianza*covarianza);
	// Regolarizzazione di Tikhonov
	if (determinante == 0.0){
	    varianzaascisse += 0.000001;
	    varianzaordinate += 0.000001;
	    determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
	}
	const double inversa11 = varianzaordinate/determinante;
	const double diagonale = -covarianza/determinante;
	const double inversa22 = varianzaascisse/determinante;
    const double primoprodotto1 = media[0]*inversa11+media[1]*diagonale;
    const double primoprodotto2 = media[0]*diagonale+media[1]*inversa22;
    const double secondoprodotto = primoprodotto1*media[0]+primoprodotto2*media[1];
    const double statistica = (taglia - 2)/(2*taglia - 1)*secondoprodotto;
    return statistica <= soglia;
}

int calcolobetatesthotelling(const int taglia, const double soglia, const double distanzabeta,
                             const double varasc, const double varord, const int volteperthread){
    int accettazioni = 0;
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> distribuzione1(0.0, varasc);
    std::normal_distribution<> distribuzione2(distanzabeta, varord);
    for (int i = 0; i < volteperthread; i++){
        std::vector<double> ascisse(taglia);
        std::vector<double> ordinate(taglia);
        for (int j = 0; j < taglia; j++){
            ascisse[j] = distribuzione1(generatore);
            ordinate[j] = distribuzione2(generatore);
        }
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
            varianzaascisse += std::pow(ascisse[j]-mediaascisse, 2);
            varianzaordinate += std::pow(ordinate[j]-mediaordinate, 2);
            covarianza += (ascisse[j]-mediaascisse)*(ordinate[j]-mediaordinate);
        }
        varianzaascisse /= taglia - 1;
        varianzaordinate /= taglia - 1;
        covarianza /= taglia - 1;
        const double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        const double inversa11 = varianzaordinate/determinante;
        const double diagonale = -covarianza/determinante;
        const double inversa22 = varianzaascisse/determinante;
        const double primoprodotto1 = mediaascisse*inversa11+mediaordinate*diagonale;
        const double primoprodotto2 = mediaascisse*diagonale+mediaordinate*inversa22;
        const double secondoprodotto = primoprodotto1*mediaascisse+primoprodotto2*mediaordinate;
        const double statistica = (taglia - 2)/(2*taglia - 1)*secondoprodotto;
        if (statistica <= soglia){
            accettazioni += 1;
        }
    }
    return accettazioni;
}

double betatesthotelling(const int taglia, const double soglia, const int volte, const double distanzabeta,
                         const double varianzaascisse, const double varianzaordinate){
    unsigned int numerothread = std::thread::hardware_concurrency();
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = volte/numerothread;
    const int iterazionirestanti = volte-volte/numerothread;
    int accettazioni = 0;
    for (unsigned int i = 0; i < numerothread - 1; ++i){
        thread[i] = std::thread([=, &accettazioni]() {
            accettazioni += calcolobetatesthotelling(taglia, soglia, distanzabeta, varianzaascisse, varianzaordinate,
                                                     volteperthread);
        });
    }
    thread[numerothread - 1] = std::thread([=, &accettazioni](){
        accettazioni += calcolobetatesthotelling(taglia, soglia, distanzabeta, varianzaascisse, varianzaordinate,
                                                 volteperthread+iterazionirestanti);
    });
    for (auto& t : thread){
        t.join();
    }
    return static_cast<double>(accettazioni)/volte;
}

bool ljungbox(const py::array_t<double>& autocor, const int h, const int n, const double soglia,
              const double secondasoglia){
    const auto buffer = autocor.request();
    const auto autocorrelazioni = static_cast<double*>(buffer.ptr);
    int numerovettori = (n - 1)/h;
    const bool resto = n - 1 - (n - 1)/h*h > 0;
    if (resto){
        numerovettori++;
    }
    std::vector<std::vector<double>> vettori(numerovettori);
    int listaattuale = 0;
    int contatore = 0;
    for (const double autocorrelazione : autocorrelazioni) {
        vettori[listaattuale][contatore] = autocorrelazione;
        contatore++;
        if (contatore == h){
        listaattuale++;
        contatore = 0;
        }
    }
    std::vector<bool> risultati(numerovettori);
    for (int i = 0; i < numerovettori; i++){
        double somma = 0.0;
        for (int j = 0; j < vettori[i].size(); j++){
            somma += std::pow(vettori[i][j], 2)/(n - j - 1);
        }
        const double q = n*(n+2)*somma;
        if (i == numerovettori - 1 && resto){
            risultati[i] = q <= secondasoglia;
        } else {
            risultati[i] = q <= soglia;
        }
    }
    for (int i = 0; i < numerovettori; i++){
        if (!risultati[i]){return false;}
    }
    return true;
}


int calcolobetaljungbox(const int taglia, const int h, const double soglia, const double secondasoglia,
                        const int iterazioni){
    int accettazioni = 0;
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normale(0.0, 1.0);
    for (int i = 0; i < iterazioni; i++){
        std::vector<double> serie(taglia);
        serie[0] = normale(generatore);
        for (int j = 1; j < taglia; j++){
            serie[j] = serie[j - 1]+normale(generatore);
        }
        std::vector<double> autocorrelazioni(taglia - 1);
        double media = 0.0;
        for (int j = 0; j < taglia; j++){
            media += serie[j];
        }
        media /= taglia;
        double varianza = 0.0;
        for (int j = 0; j < taglia; j++){
            varianza += std::pow(serie[j]-media, 2);
        }
        varianza /= taglia - 1;
        for (int k = 1; k < taglia ; k++){
            double sommatoria = 0.0;
            for (int t = 0; t < taglia-k; t++){
                sommatoria += (serie[t]-media)*(serie[t+k]-media);
            }
            autocorrelazioni[k - 1] = sommatoria/((taglia-k)*varianza);
        }
        int numerovettori = (taglia - 1)/h;
        const bool resto = taglia - 1 - (taglia - 1)/h*h > 0;
        if (resto){
            numerovettori++;
        }
        std::vector<std::vector<double>> vettori(numerovettori);
        int listaattuale = 0;
        int contatore = 0;
        for (const auto& autocorrelazione : autocorrelazioni){
            vettori[listaattuale][contatore] = autocorrelazione;
            contatore++;
            if (contatore == h) {
                listaattuale++;
                contatore = 0;
            }
        }
        std::vector<bool> risultati(numerovettori);
        for (int j = 0; j < numerovettori; j++){
            double somma = 0.0;
            for (int k = 0; k < vettori[j].size(); k++){
                somma += std::pow(vettori[j][k], 2)/(taglia - k - 1);
            }
            const double q = taglia*(taglia+2)*somma;
            if (j == numerovettori - 1 && resto){
                risultati[j] = q <= secondasoglia;
            } else {
                risultati[j] = q <= soglia;
            }
        }
        bool ipotesinulla = true;
        for (int j = 0; j < numerovettori; j++){
            if (!risultati[i]){ipotesinulla = false;}
        }
        if (ipotesinulla){accettazioni += 1;}
    }
    return accettazioni;
}


double betaljungbox(const int taglia, const int iterazioni, const int h, const double soglia,
                    const double secondasoglia){
    unsigned int numerothread = std::thread::hardware_concurrency();
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni-iterazioni/numerothread;
    int accettazioni = 0;
    for (unsigned int i = 0; i < numerothread - 1; ++i){
        thread[i] = std::thread([=, &accettazioni]() {
            accettazioni += calcolobetaljungbox(taglia, h, soglia, secondasoglia, volteperthread);
        });
    }
    thread[numerothread - 1] = std::thread([=, &accettazioni]() {
        accettazioni += calcolobetaljungbox(taglia, h, soglia, secondasoglia, volteperthread+iterazionirestanti);
    });
    for (auto& t : thread){
        t.join();
    }
    return static_cast<double>(accettazioni)/iterazioni;
}


int calcoloalfaveroljungbox(const int taglia, const int h, const double soglia, const double secondasoglia,
                            const int iterazioni){
    int accettazioni = 0;
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normale(0.0, 1.0);
    for (int i = 0; i < iterazioni; i++){
        std::vector<double> campione(taglia);
        for (int j = 0; j < taglia; j++){
            campione[j] = normale(generatore);
        }
        double media = 0.0;
        for (int j = 0; j < taglia; j++){
            media += campione[j];
        }
        media /= taglia;
        double varianza = 0.0;
        for (int j = 0; j < taglia; j++){
            varianza += std::pow(campione[j]-media, 2);
        }
        varianza /= taglia - 1;
        std::vector<double> autocorrelazioni(taglia - 1);
        for (int k = 1; k < taglia; k++){
            double sommatoria = 0.0;
            for (int t = 0; t < taglia-k; t++){
                sommatoria += (campione[t]-media)*(campione[t+k]-media);
            }
            autocorrelazioni[k - 1] = sommatoria/((taglia - 1)*varianza);
        }
        int numerovettori = (taglia - 1)/h;
        const bool resto = (taglia - 1) - (taglia - 1)/h*h > 0;
        if (resto){
            numerovettori++;
        }
        std::vector<std::vector<double>> vettori(numerovettori);
        int listaattuale = 0;
        int contatore = 0;
        for (const auto& autocorrelazione : autocorrelazioni){
            vettori[listaattuale][contatore] = autocorrelazione;
            contatore++;
            if (contatore == h) {
                listaattuale++;
                contatore = 0;
            }
        }
        std::vector<bool> risultati(numerovettori);
        for (int j = 0; j < numerovettori; j++){
            double somma = 0.0;
            for (int k = 0; k < vettori[j].size(); k++){
                somma += std::pow(vettori[j][k], 2)/(taglia - k - 1);
            }
            const double q = taglia*(taglia+2)*somma;
            if (j == numerovettori - 1 && resto){
                risultati[j] = q <= secondasoglia;
            } else {
                risultati[j] = q <= soglia;
            }
        }
        bool ipotesinulla = true;
        for (int j = 0; j < numerovettori; j++){
            if (!risultati[j]){ipotesinulla = false;}
        }
        if (ipotesinulla){accettazioni++;}
    }
    return accettazioni;
}


double alfaveroljungbox(const int taglia, const int iterazioni, const int h, const double soglia,
                        const double secondasoglia){
    unsigned int numerothread = std::thread::hardware_concurrency();
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni-iterazioni/numerothread;
    int accettazioni = 0;
    for (unsigned int i = 0; i < numerothread - 1; ++i){
        thread[i] = std::thread([=, &accettazioni](){
            accettazioni += calcoloalfaveroljungbox(taglia, h, soglia, secondasoglia, volteperthread);
        });
    }
    thread[numerothread - 1] = std::thread([=, &accettazioni](){
        accettazioni += calcoloalfaveroljungbox(taglia, h, soglia, secondasoglia, volteperthread+iterazionirestanti);
    });
    for (auto& t : thread){
        t.join();
    }
    return 1-static_cast<double>(accettazioni)/iterazioni;
}


std::array<double, 3> testhenzezirkler(const py::array_t<double>& allenamento, const int taglia){
    const auto buffer = allenamento.request();
    const auto puntatore = static_cast<double*>(buffer.ptr);
    double mediaascisse = 0.0;
    double mediaordinate = 0.0;
    for (int i = 0; i < taglia*2; i += 2){
        mediaascisse += puntatore[i];
        mediaordinate += puntatore[i+1];
    }
    mediaascisse /= taglia;
    mediaordinate /= taglia;
    double varianzaascisse = 0.0;
    double varianzaordinate = 0.0;
    double covarianza = 0.0;
    for (int i = 0; i < taglia*2; i += 2){
        varianzaascisse += std::pow(puntatore[i]-mediaascisse, 2);
        varianzaordinate += std::pow(puntatore[i+1]-mediaordinate, 2);
        covarianza += (puntatore[i]-mediaascisse)*(puntatore[i+1]-mediaordinate);
    }
    varianzaascisse /= taglia - 1;
    varianzaordinate /= taglia - 1;
    covarianza /= taglia - 1;
    double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
    if (determinante == 0.0){
        varianzaascisse += 0.000001;
        varianzaordinate += 0.000001;
        determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
    }
    double inversa11 = varianzaordinate/determinante;
    double inversa12 = -covarianza/determinante;
    double inversa22 = varianzaascisse/determinante;
    std::vector<double> daticentrati(taglia*2);
    for (int i = 0; i < taglia*2; i += 2){
        daticentrati[i] = puntatore[i]-mediaascisse;
        daticentrati[i+1] = puntatore[i+1]-mediaordinate;
    }
    std::vector<std::array<double, 2>> primoprodotto(taglia);
    for (int i = 0; i < taglia; i++){
        primoprodotto[i][0] = daticentrati[2*i]*inversa11+daticentrati[2*i+1]*inversa12;
        primoprodotto[i][1] = daticentrati[2*i]*inversa12+daticentrati[2*i+1]*inversa22;
    }
    std::vector<double> Dj(taglia);
    for (int i = 0; i < taglia; i++){
        Dj[i] = primoprodotto[i][0]*daticentrati[2*i]+primoprodotto[i][1]*daticentrati[2*i+1];
    }
    std::vector<std::array<double, 2>> primoprodottodecentrato(taglia);
    for (int i = 0; i < taglia; i++){
        primoprodottodecentrato[i][0] = puntatore[2*i]*inversa11+puntatore[2*i+1]*inversa12;
        primoprodottodecentrato[i][1] = puntatore[2*i]*inversa12+puntatore[2*i+1]*inversa22;
    }
    std::vector<std::vector<double>> Y(taglia, std::vector<double>(taglia));
    for (int i = 0; i < taglia; i++){
        for (int j = 0; j < taglia; j++){
            Y[i][j] = primoprodottodecentrato[i][0]*puntatore[2*i]+primoprodottodecentrato[i][1]*puntatore[2*i+1];
        }
    }
    std::vector<double> Y_diag(taglia);
    for (int i = 0; i < taglia; i++){
        Y_diag[i] = Y[i][i];
    }
    std::vector<std::vector<double>> Djk(taglia, std::vector<double>(taglia));
    for (int i = 0; i < taglia; i++){
        for (int j = 0; j < taglia; j++){
            Djk[i][j] = -2*Y[j][i]+Y_diag[i]+Y_diag[j];
        }
    }
    double lisciatore = 1/std::sqrt(2)*std::pow(5/4, 1/6)*std::pow(taglia, 1/6);
    double sommakernel = 0.0;
    for (int i = 0; i < taglia; i++){
        for (int j = 0; j < taglia; j++){
            sommakernel -= std::exp(lisciatore*lisciatore/2*Djk[i][j]);
        }
    }
    double hz = sommakernel/(taglia*taglia);
    double sommakernel2 = 0.0;
    for (int i = 0; i < taglia; i++){
        sommakernel2 += std::exp(-((lisciatore*lisciatore)/(2*(1+lisciatore*lisciatore)))*Dj[i]);
    }
    hz -= 2/((1+lisciatore*lisciatore))/taglia;
    hz += 1/(1+2*lisciatore*lisciatore);
    hz *= taglia;
    double wb = (1+lisciatore*lisciatore)*(1+3*lisciatore*lisciatore);
    double a = 1+2*lisciatore*lisciatore;
    double mu = 1-(1+2*lisciatore*lisciatore/a+8*lisciatore*lisciatore/(2*a*a))/a;
    double si2 = (2/(1+4*lisciatore*lisciatore)+2/(a*a)*(1+4*lisciatore*lisciatore/(a*a)+(24*std::pow(lisciatore, 8))
                  /(4*std::pow(a, 4)))-4/wb*(1+(6*std::pow(lisciatore, 4))/(2*wb)+8*std::pow(lisciatore, 8)/(2*wb*wb)));
    double pmu = std::log(std::sqrt(std::pow(mu, 4)/(si2+mu*mu)));
    double psi = std::sqrt(std::log(1+si2/(mu*mu)));
    std::array<double, 3> output = {hz, psi, std::exp(pmu)};
    return output;
}


bool testmardia(const py::array_t<double>& allenamento, const int taglia, const double sogliaasimmetria,
                const double sogliacurtosi){
    const auto buffer = allenamento.request();
    const auto puntatore = static_cast<double*>(buffer.ptr);
    double mediaascisse = 0.0;
    double mediaordinate = 0.0;
    for (int i = 0; i < taglia*2; i += 2){
        mediaascisse += puntatore[2*i];
        mediaordinate += puntatore[2*i+1];
    }
    mediaascisse /= taglia;
    mediaordinate /= taglia;
    double varianzaascisse = 0.0;
    double varianzaordinate = 0.0;
    double covarianza = 0.0;
    for (int i = 0; i < taglia*2; i += 2){
        varianzaascisse += std::pow(puntatore[2*i]-mediaascisse, 2);
        varianzaordinate += std::pow(puntatore[2*i+1]-mediaordinate, 2);
        covarianza += (puntatore[2*i]-mediaascisse)*(puntatore[2*i+1]-mediaordinate);
    }
    varianzaascisse /= taglia - 1;
    varianzaordinate /= taglia - 1;
    covarianza /= taglia - 1;
    double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
    if (determinante == 0.0){
        varianzaascisse += 0.000001;
        varianzaordinate += 0.000001;
        determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
    }
    const double inversa11 = varianzaordinate/determinante;
    const double inversa22 = varianzaascisse/determinante;
    const double inversa12 = -covarianza/determinante;
    std::vector<double> sommandi(taglia*taglia);
    std::vector<double> daticentrati(taglia*2);
    for (int i = 0; i < taglia; i++){
        daticentrati[2*i] = puntatore[2*i]-mediaascisse;
        daticentrati[2*i+1] = puntatore[2*i+1]-mediaordinate;
    }
    for (int i = 0; i < taglia; i++){
        for (int j = 0; j < taglia; j++){
            std::array<double, 2> primoprodotto = {daticentrati[2*i]*inversa11+daticentrati[2*i+1]*inversa12,
                                                   daticentrati[2*i]*inversa12+daticentrati[2*i+1]*inversa22};
            sommandi[i*taglia+j] = std::pow(primoprodotto[0]*daticentrati[2*j]+primoprodotto[1]*daticentrati[2*j+1], 3);
        }
    }
    double asimmetria = 0.0;
    for (int i = 0; i < taglia*taglia; i++){
        asimmetria += sommandi[i];
    }
    asimmetria /= 6*taglia;
    double curtosi = 0.0;
    for (int i = 0; i < taglia; i++){
        curtosi += sommandi[i*taglia+i];
    }
    curtosi /= taglia;
    curtosi -= 8;
    curtosi *= std::sqrt(taglia/64);
    return asimmetria < sogliaasimmetria && curtosi < sogliacurtosi;
}


void calcolabetatesthenzezirkler(const int taglia, const std::string& distribuzione, const int iterazioni,
                                 const int supplementari,
                                 std::vector<std::array<double, 3>>& statistiche, const int numerothread,
                                 const double asimmetriaascisse, const double asimmetriaordinate,
                                 const double distanzacomponenti, const int gradit){
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    for (int volta = 0; volta < iterazioni+supplementari; volta++){
        std::vector<std::array<double, 2>> campione(taglia, {0.0, 0.0});
        if (distribuzione == "laplace"){
            // Da un pacchetto R
            std::exponential_distribution<> esponenziale(1.0);
            std::normal_distribution<> normale(0.0, 1.0);
            for (int i = 0; i < taglia; i++){
                double z = esponenziale(generatore);
                campione[i][0] = normale(generatore)*std::sqrt(z);
                campione[i][1] = normale(generatore)*std::sqrt(z);
            }
        } else if (distribuzione == "normaleasimmetrica"){
            // Dal paper di Azzalini
            double delta1 = asimmetriaascisse/std::sqrt(1+asimmetriaascisse*asimmetriaascisse);
            double delta2 = asimmetriaordinate/std::sqrt(1+asimmetriaordinate*asimmetriaordinate);
            std::array<double, 3> primarigaomega = {1, delta1, delta2};
            std::array<double, 3> secondarigaomega = {delta1, 1.0, 0.0};
            std::array<double, 3> terzarigaomega = {delta2, 0.0, 1.0};
            std::array<std::array<double, 3>, 3> omega = {primarigaomega, secondarigaomega, terzarigaomega};
            std::array<double, 3> rigavuota = {0.0, 0.0, 0.0};
            std::array<std::array<double, 3>, 3> cholesky = {rigavuota, rigavuota, rigavuota};
            for (int i = 0; i < 3; i++){
                for (int j = 0; j <= i; j++){
                    double somma = 0.0;
                    for (int k = 0; k < j; k++){
                        somma += cholesky[i][k]*cholesky[j][k];
                    }
                    if (i == j){
                        cholesky[i][j] = std::sqrt(1-somma);
                    } else {
                        cholesky[i][j] = 1/cholesky[j][j]*(omega[i][j]-somma);
                    }
                }
            }
            int completati = 0;
            std::normal_distribution<> normale(0.0, 1.0);
            while (completati < taglia){
                std::array<double, 3> trenormali;
                trenormali[0] = normale(generatore);
                trenormali[1] = normale(generatore);
                trenormali[2] = normale(generatore);
                std::array<double, 3> normalisigma = {cholesky[0][0]*trenormali[0],
                                                      cholesky[1][0]*trenormali[0]+cholesky[1][1]*trenormali[1],
                                                      cholesky[2][0]*trenormali[0]+cholesky[2][1]*trenormali[1]+cholesky[2][2]*trenormali[2]};
                if (normalisigma[0] > 0){
                    campione[completati] = {normalisigma[1], normalisigma[2]};
                    completati++;
                }
            }
        } else if (distribuzione == "uniforme"){
            std::uniform_real_distribution<> uniforme(-1, 1);
            for (int i = 0; i < taglia; i++){
                campione[i] = {uniforme(generatore), uniforme(generatore)};
            }
        } else if (distribuzione == "mistura"){
            std::uniform_real_distribution<> uniforme(0, 1);
            std::normal_distribution<> normale1(-distanzacomponenti/2, 1.0);
            std::normal_distribution<> normale2(distanzacomponenti/2, 1.0);
            std::normal_distribution<> normale3(0, 1.0);
            for (int i = 0; i < taglia; i++){
                double u = uniforme(generatore);
                if (u > 0.5){
                    campione[i][0] = normale1(generatore);
                    campione[i][1] = normale3(generatore);
                } else {
                    campione[i][0] = normale2(generatore);
                    campione[i][1] = normale3(generatore);
                }
            }
        } else if (distribuzione == "lognormale"){
            std::normal_distribution<> normale(0.0, 1.0);
            for (int i = 0; i < taglia; i++){
                campione[i] = {std::exp(normale(generatore)), std::exp(normale(generatore))};
            }
        } else if (distribuzione == "t"){
            std::normal_distribution<> normale(0.0, 1.0);
            std::chi_squared_distribution<> chiquadrato(gradit);
            for (int i = 0; i < taglia; i++){
                double z1 = normale(generatore);
                double z2 = normale(generatore);
                double u = chiquadrato(generatore);
                campione[i] = {z1/std::sqrt(u/gradit), z2/std::sqrt(u/gradit)};
            }
        }
        double mediaascisse = 0.0;
        double mediaordinate = 0.0;
        for (int i = 0; i < taglia; i++){
            mediaascisse += campione[i][0];
            mediaordinate += campione[i][1];
        }
        mediaascisse /= taglia;
        mediaordinate /= taglia;
        double varianzaascisse = 0.0;
        double varianzaordinate = 0.0;
        double covarianza = 0.0;
        for (int i = 0; i < taglia; i++){
            varianzaascisse += std::pow(campione[i][0]-mediaascisse, 2);
            varianzaordinate += std::pow(campione[i][1]-mediaordinate, 2);
            covarianza += (campione[i][0]-mediaascisse)*(campione[i][1]-mediaordinate);
        }
        varianzaascisse /= taglia - 1;
        varianzaordinate /= taglia - 1;
        covarianza /= taglia - 1;
        double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        if (determinante == 0.0){
            varianzaascisse += 0.000001;
            varianzaordinate += 0.000001;
            determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        }
        double inversa11 = varianzaordinate/determinante;
        double inversa12 = -covarianza/determinante;
        double inversa22 = varianzaascisse/determinante;
        std::vector<double> daticentrati(taglia*2);
        for (int i = 0; i < taglia; i++){
            daticentrati[2*i] = campione[i][0]-mediaascisse;
            daticentrati[2*i+1] = campione[i][1]-mediaordinate;
        }
        std::vector<std::array<double, 2>> primoprodotto(taglia);
        for (int i = 0; i < taglia; i++){
            primoprodotto[i][0] = daticentrati[2*i]*inversa11+daticentrati[2*i+1]*inversa12;
            primoprodotto[i][1] = daticentrati[2*i]*inversa12+daticentrati[2*i+1]*inversa22;
        }
        std::vector<double> Dj(taglia);
        for (int i = 0; i < taglia; i++){
            Dj[i] = primoprodotto[i][0]*daticentrati[2*i]+primoprodotto[i][1]*daticentrati[2*i+1];
        }
        std::vector<std::array<double, 2>> primoprodottodecentrato(taglia);
        for (int i = 0; i < taglia; i++){
            primoprodottodecentrato[i][0] = campione[i][0]*inversa11+campione[i][1]*inversa12;
            primoprodottodecentrato[i][1] = campione[i][0]*inversa12+campione[i][1]*inversa22;
        }
        std::vector<std::vector<double>> Y(taglia, std::vector<double>(taglia));
        for (int i = 0; i < taglia; i++){
            for (int j = 0; j < taglia; j++){
                Y[i][j] = primoprodottodecentrato[i][0]*campione[i][0]+primoprodottodecentrato[i][1]*campione[i][1];
            }
        }
        std::vector<double> Y_diag(taglia);
        for (int i = 0; i < taglia; i++){
            Y_diag[i] = Y[i][i];
        }
        std::vector<std::vector<double>> Djk(taglia, std::vector<double>(taglia));
        for (int i = 0; i < taglia; i++){
            for (int j = 0; j < taglia; j++){
                Djk[i][j] = -2*Y[j][i]+Y_diag[i]+Y_diag[j];
            }
        }
        double lisciatore = 1/std::sqrt(2)*std::pow(5/4, 1/6)*std::pow(taglia, 1/6);
        double sommakernel = 0.0;
        for (int i = 0; i < taglia; i++){
            for (int j = 0; j < taglia; j++){
                sommakernel -= std::exp(lisciatore*lisciatore/2*Djk[i][j]);
            }
        }
        double hz = sommakernel/(taglia*taglia);
        double sommakernel2 = 0.0;
        for (int i = 0; i < taglia; i++){
            sommakernel2 += std::exp(-((lisciatore*lisciatore)/(2*(1+lisciatore*lisciatore)))*Dj[i]);
        }
        hz -= 2/((1+lisciatore*lisciatore))/taglia;
        hz += 1/(1+2*lisciatore*lisciatore);
        hz *= taglia;
        double wb = (1+lisciatore*lisciatore)*(1+3*lisciatore*lisciatore);
        double a = 1+2*lisciatore*lisciatore;
        double mu = 1-(1+2*lisciatore*lisciatore/a+8*lisciatore*lisciatore/(2*a*a))/a;
        double si2 = 2/(1+4*lisciatore*lisciatore)+2/(a*a)*(1+4*lisciatore*lisciatore/(a*a)+24*std::pow(lisciatore, 8))
                      /(4*std::pow(a, 4))-4/wb*(1+6*std::pow(lisciatore, 4)/(2*wb)+8*std::pow(lisciatore, 8)/(2*wb*wb));
        double pmu = std::log(std::sqrt(std::pow(mu, 4)/(si2+mu*mu)));
        double psi = std::sqrt(std::log(1+si2/(mu*mu)));
        statistiche[numerothread*iterazioni+volta] = {hz, psi, std::exp(pmu)};
    }
}


std::vector<std::array<double, 3>> betatesthenzezirkler(const std::string& distribuzione, const int taglia,
                                                        const int iterazioni, const int gradit,
                                                        const double asimmetriaascisse, const double asimmetriaordinate,
                                                        const double distanzacomponenti){
    unsigned int numerothread = std::thread::hardware_concurrency();
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni-iterazioni/numerothread;
    std::vector<std::array<double, 3>> statistiche(iterazioni, {0.0, 0.0, 0.0});
    for (unsigned int i = 0; i < numerothread - 1; ++i){
        thread[i] = std::thread([=, &statistiche, i](){
            calcolabetatesthenzezirkler(taglia, distribuzione, volteperthread, 0, statistiche, i,
                                        asimmetriaascisse, asimmetriaordinate, distanzacomponenti, gradit);
        });
    }
    thread[numerothread - 1] = std::thread([=, &statistiche](){
        calcolabetatesthenzezirkler(taglia, distribuzione, volteperthread, iterazionirestanti,
            statistiche, numerothread-1, asimmetriaascisse, asimmetriaordinate,
            distanzacomponenti, gradit);
    });
    for (auto& t : thread){
        t.join();
    }
    return statistiche;
}


int calcolabetatestmardia(const int iterazioni, const int taglia, const std::string &distribuzione,
    const int gradit, const double asimmetriaascisse, const double asimmetriaordinate,
    const double distanzacomponenti, const double sogliaasimmetria, const double sogliacurtosi){
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    int accettazioni = 0;
    for (int volta = 0; volta < iterazioni; volta++) {
        std::vector<std::array<double, 2>> campione(taglia, {0.0, 0.0});
        if (distribuzione == "laplace"){
            // Da un pacchetto R
            std::exponential_distribution<> esponenziale(1.0);
            std::normal_distribution<> normale(0.0, 1.0);
            for (int i = 0; i < taglia; i++){
                double z = esponenziale(generatore);
                campione[i][0] = normale(generatore)*std::sqrt(z);
                campione[i][1] = normale(generatore)*std::sqrt(z);
            }
        } else if (distribuzione == "normaleasimmetrica"){
            // Dal paper di Azzalini
            double delta1 = asimmetriaascisse/std::sqrt(1+asimmetriaascisse*asimmetriaascisse);
            double delta2 = asimmetriaordinate/std::sqrt(1+asimmetriaordinate*asimmetriaordinate);
            std::array<double, 3> primarigaomega = {1, delta1, delta2};
            std::array<double, 3> secondarigaomega = {delta1, 1.0, 0.0};
            std::array<double, 3> terzarigaomega = {delta2, 0.0, 1.0};
            std::array<std::array<double, 3>, 3> omega = {primarigaomega, secondarigaomega, terzarigaomega};
            std::array<double, 3> rigavuota = {0.0, 0.0, 0.0};
            std::array<std::array<double, 3>, 3> cholesky = {rigavuota, rigavuota, rigavuota};
            for (int i = 0; i < 3; i++){
                for (int j = 0; j <= i; j++){
                    double somma = 0.0;
                    for (int k = 0; k < j; k++){
                        somma += cholesky[i][k]*cholesky[j][k];
                    }
                    if (i == j){
                        cholesky[i][j] = std::sqrt(1-somma);
                    } else {
                        cholesky[i][j] = 1/cholesky[j][j]*(omega[i][j]-somma);
                    }
                }
            }
            int completati = 0;
            std::normal_distribution<> normale(0.0, 1.0);
            while (completati < taglia){
                std::array<double, 3> trenormali;
                trenormali[0] = normale(generatore);
                trenormali[1] = normale(generatore);
                trenormali[2] = normale(generatore);
                std::array<double, 3> normalisigma = {cholesky[0][0]*trenormali[0],
                                                      cholesky[1][0]*trenormali[0]+cholesky[1][1]*trenormali[1],
                                                      cholesky[2][0]*trenormali[0]+cholesky[2][1]*trenormali[1]+cholesky[2][2]*trenormali[2]};
                if (normalisigma[0] > 0){
                    campione[completati] = {normalisigma[1], normalisigma[2]};
                    completati++;
                }
            }
        } else if (distribuzione == "uniforme"){
            std::uniform_real_distribution<> uniforme(-1, 1);
            for (int i = 0; i < taglia; i++){
                campione[i] = {uniforme(generatore), uniforme(generatore)};
            }
        } else if (distribuzione == "mistura"){
            std::uniform_real_distribution<> uniforme(0, 1);
            std::normal_distribution<> normale1(-distanzacomponenti/2, 1.0);
            std::normal_distribution<> normale2(distanzacomponenti/2, 1.0);
            std::normal_distribution<> normale3(0, 1.0);
            for (int i = 0; i < taglia; i++){
                double u = uniforme(generatore);
                if (u > 0.5){
                    campione[i][0] = normale1(generatore);
                    campione[i][1] = normale3(generatore);
                } else {
                    campione[i][0] = normale2(generatore);
                    campione[i][1] = normale3(generatore);
                }
            }
        } else if (distribuzione == "lognormale"){
            std::normal_distribution<> normale(0.0, 1.0);
            for (int i = 0; i < taglia; i++){
                campione[i] = {std::exp(normale(generatore)), std::exp(normale(generatore))};
            }
        } else if (distribuzione == "t") {
            std::normal_distribution<> normale(0.0, 1.0);
            std::chi_squared_distribution<> chiquadrato(gradit);
            for (int i = 0; i < taglia; i++){
                double z1 = normale(generatore);
                double z2 = normale(generatore);
                double u = chiquadrato(generatore);
                campione[i] = {z1/std::sqrt(u/gradit), z2/std::sqrt(u/gradit)};
            }
        }
        double mediaascisse = 0.0;
        double mediaordinate = 0.0;
        for (int i = 0; i < taglia; i++){
            mediaascisse += campione[i][0];
            mediaordinate += campione[i][1];
        }
        mediaascisse /= taglia;
        mediaordinate /= taglia;
        double varianzaascisse = 0.0;
        double varianzaordinate = 0.0;
        double covarianza = 0.0;
        for (int i = 0; i < taglia; i++){
            varianzaascisse += std::pow(campione[i][0]-mediaascisse, 2);
            varianzaordinate += std::pow(campione[i][1]-mediaordinate, 2);
            covarianza += (campione[i][0]-mediaascisse)*(campione[i][1]-mediaordinate);
        }
        varianzaascisse /= taglia - 1;
        varianzaordinate /= taglia - 1;
        covarianza /= taglia - 1;
        double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        if (determinante == 0.0){
            varianzaascisse += 0.000001;
            varianzaordinate += 0.000001;
            determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        }
        const double inversa11 = varianzaordinate/determinante;
        const double inversa22 = varianzaascisse/determinante;
        const double inversa12 = -covarianza/determinante;
        std::vector<double> sommandi(taglia*taglia);
        std::vector<double> daticentrati(taglia*2);
        for (int i = 0; i < taglia; i++){
            daticentrati[2*i] = campione[i][0]-mediaascisse;
            daticentrati[2*i+1] = campione[i][1]-mediaordinate;
        }
        for (int i = 0; i < taglia; i++){
            for (int j = 0; j < taglia; j++){
                std::array<double, 2> primoprodotto = {daticentrati[2*i]*inversa11+daticentrati[2*i+1]*inversa12,
                                                       daticentrati[2*i]*inversa12+daticentrati[2*i+1]*inversa22};
                sommandi[i*taglia+j] = std::pow(primoprodotto[0]*daticentrati[2*j]+primoprodotto[1]*daticentrati[2*j+1], 3);
            }
        }
        double asimmetria = 0.0;
        for (int i = 0; i < taglia*taglia; i++){
            asimmetria += sommandi[i];
        }
        asimmetria /= 6*taglia;
        double curtosi = 0.0;
        for (int i = 0; i < taglia; i++){
            curtosi += sommandi[i*taglia+i];
        }
        curtosi /= taglia;
        curtosi -= 8;
        curtosi *= std::sqrt(taglia/64);
        if (asimmetria < sogliaasimmetria && curtosi < sogliacurtosi) {
            accettazioni++;
        }
    }
    return accettazioni;
}


double betatestmardia(const int iterazioni, const int taglia, const std::string &distribuzione,
    const double asimmetriaascisse, const double asimmetriaordinate, const double distanzacomponenti,
    const int gradit, const double sogliaasimmetria, const double sogliacurtosi) {
    unsigned int numerothread = std::thread::hardware_concurrency();
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni-iterazioni/numerothread;
    int accettazioni = 0;
    for (unsigned int i = 0; i < numerothread - 1; ++i){
        thread[i] = std::thread([=, &accettazioni](){
            accettazioni += calcolabetatestmardia(volteperthread, taglia, distribuzione, gradit,
                asimmetriaascisse, asimmetriaordinate, distanzacomponenti, sogliaasimmetria, sogliacurtosi);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazioni]() {
        accettazioni += calcolabetatestmardia(volteperthread+iterazionirestanti, taglia, distribuzione, gradit,
        asimmetriaascisse, asimmetriaordinate, distanzacomponenti, sogliaasimmetria, sogliacurtosi);
    });
    for (auto& t : thread){
        t.join();
    }
    return static_cast<double>(accettazioni)/iterazioni;
}


void calcolaalfaverohenzezirkler(const int taglia, const int iterazioni, const int supplementari,
    std::vector<std::array<double, 3>>& statistiche, const int numerothread) {
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normale(0.0, 1.0);
    std::vector<std::array<double, 2>> campione(taglia);
    for (int iterazione = 0; iterazione < iterazioni+supplementari; iterazione++) {
        for (int i = 0; i < taglia; i++) {
            campione[i][0] = normale(generatore);
            campione[i][1] = normale(generatore);
        }
        double mediaascisse = 0.0;
        double mediaordinate = 0.0;
        for (int i = 0; i < taglia; i++){
            mediaascisse += campione[i][0];
            mediaordinate += campione[i][1];
        }
        mediaascisse /= taglia;
        mediaordinate /= taglia;
        double varianzaascisse = 0.0;
        double varianzaordinate = 0.0;
        double covarianza = 0.0;
        for (int i = 0; i < taglia; i++){
            varianzaascisse += std::pow(campione[i][0]-mediaascisse, 2);
            varianzaordinate += std::pow(campione[i][1]-mediaordinate, 2);
            covarianza += (campione[i][0]-mediaascisse)*(campione[i][1]-mediaordinate);
        }
        varianzaascisse /= taglia - 1;
        varianzaordinate /= taglia - 1;
        covarianza /= taglia - 1;
        double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        if (determinante == 0.0){
            varianzaascisse += 0.000001;
            varianzaordinate += 0.000001;
            determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        }
        double inversa11 = varianzaordinate/determinante;
        double inversa12 = -covarianza/determinante;
        double inversa22 = varianzaascisse/determinante;
        std::vector<double> daticentrati(taglia);
        for (int i = 0; i < taglia; i++){
            daticentrati[i] = campione[i][0]-mediaascisse;
            daticentrati[i+1] = campione[i][1]-mediaordinate;
        }
        std::vector<std::array<double, 2>> primoprodotto(taglia);
        for (int i = 0; i < taglia; i++){
            primoprodotto[i][0] = daticentrati[2*i]*inversa11+daticentrati[2*i+1]*inversa12;
            primoprodotto[i][1] = daticentrati[2*i]*inversa12+daticentrati[2*i+1]*inversa22;
        }
        std::vector<double> Dj(taglia);
        for (int i = 0; i < taglia; i++){
            Dj[i] = primoprodotto[i][0]*daticentrati[2*i]+primoprodotto[i][1]*daticentrati[2*i+1];
        }
        std::vector<std::array<double, 2>> primoprodottodecentrato(taglia);
        for (int i = 0; i < taglia; i++){
            primoprodottodecentrato[i][0] = campione[i][0]*inversa11+campione[i][1]*inversa12;
            primoprodottodecentrato[i][1] = campione[i][0]*inversa12+campione[i][1]*inversa22;
        }
        std::vector<std::vector<double>> Y(taglia, std::vector<double>(taglia));
        for (int i = 0; i < taglia; i++){
            for (int j = 0; j < taglia; j++){
                Y[i][j] = primoprodottodecentrato[i][0]*campione[i][0]+primoprodottodecentrato[i][1]*campione[i][1];
            }
        }
        std::vector<double> Y_diag(taglia);
        for (int i = 0; i < taglia; i++){
            Y_diag[i] = Y[i][i];
        }
        std::vector<std::vector<double>> Djk(taglia, std::vector<double>(taglia));
        for (int i = 0; i < taglia; i++){
            for (int j = 0; j < taglia; j++){
                Djk[i][j] = -2*Y[j][i]+Y_diag[i]+Y_diag[j];
            }
        }
        double lisciatore = 1/std::sqrt(2)*std::pow(5/4, 1/6)*std::pow(taglia, 1/6);
        double sommakernel = 0.0;
        for (int i = 0; i < taglia; i++){
            for (int j = 0; j < taglia; j++){
                sommakernel -= std::exp(lisciatore*lisciatore/2*Djk[i][j]);
            }
        }
        double hz = sommakernel/(taglia*taglia);
        double sommakernel2 = 0.0;
        for (int i = 0; i < taglia; i++){
            sommakernel2 += std::exp(-((lisciatore*lisciatore)/(2*(1+lisciatore*lisciatore)))*Dj[i]);
        }
        hz -= 2/((1+lisciatore*lisciatore))/taglia;
        hz += 1/(1+2*lisciatore*lisciatore);
        hz *= taglia;
        double wb = (1+lisciatore*lisciatore)*(1+3*lisciatore*lisciatore);
        double a = 1+2*lisciatore*lisciatore;
        double mu = 1-(1+2*lisciatore*lisciatore/a+8*lisciatore*lisciatore/(2*a*a))/a;
        double si2 = (2/(1+4*lisciatore*lisciatore)+2/(a*a)*(1+4*lisciatore*lisciatore/(a*a)+(24*std::pow(lisciatore, 8))
                      /(4*std::pow(a, 4)))-4/wb*(1+(6*std::pow(lisciatore, 4))/(2*wb)+8*std::pow(lisciatore, 8)/(2*wb*wb)));
        double pmu = std::log(std::sqrt(std::pow(mu, 4)/(si2+mu*mu)));
        double psi = std::sqrt(std::log(1+si2/(mu*mu)));
        std::array<double, 3> output = {hz, psi, std::exp(pmu)};
        statistiche[iterazioni*numerothread+iterazione] = output;
    }
}


std::vector<std::array<double, 3>> alfaverohenzezirkler(const int iterazioni, const int taglia) {
    unsigned int numerothread = std::thread::hardware_concurrency();
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni-iterazioni/numerothread;
    std::vector<std::array<double, 3>> statistiche(iterazioni);
    for (unsigned int i = 0; i < numerothread - 1; ++i){
        thread[i] = std::thread([=, &statistiche](){
            calcolaalfaverohenzezirkler(taglia, volteperthread, 0, statistiche, i);
        });
    }
    thread[numerothread-1] = std::thread([=, &statistiche]() {
        calcolaalfaverohenzezirkler(taglia, volteperthread, iterazionirestanti, statistiche, numerothread-1);});
    for (auto& t : thread){
        t.join();
    }
    return statistiche;
}


int calcolaalfaverotestmardia(const int iterazioni, const int taglia, const double sogliaasimmetria,
    const double sogliacurtosi) {
    std::random_device dispositivo;
    std::mt19937 generatore(dispositivo());
    std::normal_distribution<> normale(0.0, 1.0);
    int accettazioni = 0;
    for (int i = 0; i < iterazioni; i++) {
        std::vector<std::array<double, 2>> campione(taglia);
        for (int j = 0; j < taglia; j++) {
            campione[j][0] = normale(generatore);
            campione[j][1] = normale(generatore);
        }
        double mediaascisse = 0.0;
        double mediaordinate = 0.0;
        for (int j = 0; j < taglia; j++){
            mediaascisse += campione[j][0];
            mediaordinate += campione[j][1];
        }
        mediaascisse /= taglia;
        mediaordinate /= taglia;
        double varianzaascisse = 0.0;
        double varianzaordinate = 0.0;
        double covarianza = 0.0;
        for (int j = 0; j < taglia; j++){
            varianzaascisse += std::pow(campione[j][0]-mediaascisse, 2);
            varianzaordinate += std::pow(campione[j][1]-mediaordinate, 2);
            covarianza += (campione[j][0]-mediaascisse)*(campione[j][1]-mediaordinate);
        }
        varianzaascisse /= taglia - 1;
        varianzaordinate /= taglia - 1;
        covarianza /= taglia - 1;
        double determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        if (determinante == 0.0){
            varianzaascisse += 0.000001;
            varianzaordinate += 0.000001;
            determinante = varianzaascisse*varianzaordinate-covarianza*covarianza;
        }
        const double inversa11 = varianzaordinate/determinante;
        const double inversa22 = varianzaascisse/determinante;
        const double inversa12 = -covarianza/determinante;
        std::vector<double> sommandi(taglia*taglia);
        std::vector<double> daticentrati(taglia*2);
        for (int j = 0; j < taglia; j++){
            daticentrati[2*j] = campione[j][0]-mediaascisse;
            daticentrati[2*j+1] = campione[j][1]-mediaordinate;
        }
        for (int j = 0; j < taglia; j++){
            for (int k = 0; k < taglia; k++){
                std::array<double, 2> primoprodotto = {daticentrati[2*j]*inversa11+daticentrati[2*j+1]*inversa12,
                                                       daticentrati[2*j]*inversa12+daticentrati[2*j+1]*inversa22};
                sommandi[j*taglia+k] = std::pow(primoprodotto[0]*daticentrati[2*k]+primoprodotto[1]*daticentrati[2*k+1], 3);
            }
        }
        double asimmetria = 0.0;
        for (int j = 0; j < taglia*taglia; j++){
            asimmetria += sommandi[j];
        }
        asimmetria /= 6*taglia;
        double curtosi = 0.0;
        for (int j = 0; j < taglia; j++){
            curtosi += sommandi[j*taglia+j];
        }
        curtosi /= taglia;
        curtosi -= 8;
        curtosi *= std::sqrt(taglia/64);
        if (asimmetria < sogliaasimmetria && curtosi < sogliacurtosi) {
            accettazioni++;
        }
    }
    return accettazioni;
}


double alfaveromardia(const int iterazioni, const int taglia, const double sogliaasimmetria,
    const double sogliacurtosi){
    unsigned int numerothread = std::thread::hardware_concurrency();
    if (numerothread == 0){numerothread = 4;}
    std::vector<std::thread> thread(numerothread);
    const int volteperthread = iterazioni/numerothread;
    const int iterazionirestanti = iterazioni-iterazioni/numerothread;
    int accettazioni = 0;
    for (unsigned int i = 0; i < numerothread - 1; ++i){
        thread[i] = std::thread([=, &accettazioni](){
            accettazioni += calcolaalfaverotestmardia(volteperthread, taglia, sogliaasimmetria, sogliacurtosi);
        });
    }
    thread[numerothread-1] = std::thread([=, &accettazioni]() {
        accettazioni += calcolaalfaverotestmardia(volteperthread+iterazionirestanti, taglia, sogliaasimmetria, sogliacurtosi);
    });
    for (auto& t : thread){
        t.join();
    }
    return static_cast<double>(accettazioni)/iterazioni;
}


void clusteringconcriterio() {

}


std::vector<int> clustering(const std::string &inizializzazione, const int taglia,
    const std::vector<std::array<double, 2>> allenamento, const int iterazionikmeans,
    const int componentimassime, const std::string selezione, const int campionibootstrap,
    const double alfabootstrap, const int iterazioniinizializzazione, const int iterazioniEM,
    const bool normale) {
    if (selezione == "criterio") {
        // Va threadato
        unsigned int numerothread = std::thread::hardware_concurrency();
        if (numerothread == 0){numerothread = 4;}
        std::vector<int> clusterpercomponenti(taglia);
        double criterio;
        double minimoascisse = 0.0;
        double minimoordinate = 0.0;
        double massimoascisse = 0.0;
        double massimoordinate = 0.0;
        for (int i = 0; i < taglia; i++) {
            const double ascissa = allenamento[i][0];
            const double ordinata = allenamento[i][1];
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
        std::random_device dispositivo;
        std::mt19937 generatore(dispositivo());
        std::uniform_real_distribution<> uniformeascisse(minimoascisse, minimoordinate);
        std::uniform_real_distribution<> uniformeordinate(minimoordinate, massimoordinate);
        for (int i = 2; i <= componentimassime; i++) {
            double logverosimiglianza;
            std::vector<int> cluster(taglia);
            for (int j = 0; j < iterazioniinizializzazione; j++) {
                int parcomp;
                if (normale){parcomp = 6;} else {parcomp = 7;}
                std::vector<double> parametri(parcomp*i);
                std::vector<int> clustercorrenti(taglia);
                if (inizializzazione == "kmeans") {
                    std::vector<std::array<double, 2>> baricentri(i);
                    for (int k = 0; k < i; k++) {
                        baricentri[k][0] = uniformeascisse(generatore);
                        baricentri[k][1] = uniformeordinate(generatore);
                    }
                    std::vector<int> clusterkmeans(taglia);
                    for (int k = 0; k < taglia; k++) {
                        double distanzaminima;
                        for (int l = 0; l < i; l++) {
                            double distanza = std::sqrt(std::pow(allenamento[k][0]-baricentri[l][0], 2)+
                                std::pow(allenamento[k][1]-baricentri[l][1], 2));
                            if (l == 0) {
                                clusterkmeans[k] = 0;
                                distanzaminima = distanza;
                            }
                            if (distanza < distanzaminima) {
                                clusterkmeans[k] = l;
                                distanzaminima = distanza;
                            }
                        }
                    }
                    for (int k = 0; k < iterazionikmeans; k++) {
                        std::vector<int> tagliecluster(i, 0);
                        for (int l = 0; l < i; l++) {
                            baricentri[l] = {0.0, 0.0};
                        }
                        for (int l = 0; l < taglia; l++) {
                            baricentri[clusterkmeans[l]][0] += allenamento[l][0];
                            baricentri[clusterkmeans[l]][1] += allenamento[l][1];
                            tagliecluster[clusterkmeans[l]]++;
                        }
                        for (int l = 0; l < i; l++) {
                            baricentri[l][0] /= tagliecluster[l];
                            baricentri[l][1] /= tagliecluster[l];
                        }
                        for (int l = 0; l < taglia; l++) {
                            double distanzaminima;
                            for (int m = 0; m < i; m++) {
                                double distanza = std::sqrt(std::pow(allenamento[l][0]-baricentri[m][0], 2)+
                                    std::pow(allenamento[l][1]-baricentri[m][1], 2));
                                if (m == 0) {
                                    clusterkmeans[l] = 0;
                                    distanzaminima = distanza;
                                }
                                if (distanza < distanzaminima) {
                                    clusterkmeans[l] = l;
                                    distanzaminima = distanza;
                                }
                            }
                        }
                    }
                    std::vector<int> tagliecluster(i, 0);
                    for (int k = 0; k < taglia; k++) {
                        tagliecluster[clusterkmeans[k]]++;
                    }
                    for (int k = 0; k < i; k++) {
                        parametri[parcomp*k] = static_cast<double>(tagliecluster[k])/taglia;
                        double mediaascisse = 0.0;
                        double mediaordinate = 0.0;
                        for (int l = 0; l < taglia; l++) {
                            if (clusterkmeans[l] == k) {
                                mediaascisse += allenamento[l][0];
                                mediaordinate += allenamento[l][1];
                            }
                        }
                        int tagliacluster = tagliecluster[k];
                        mediaascisse /= tagliacluster;
                        mediaordinate /= tagliacluster;
                        double varianzaascisse = 0.0;
                        double varianzaordinate = 0.0;
                        double covarianza;
                        for (int l = 0; l < taglia; l++) {
                            double ascissa = allenamento[l][0];
                            double ordinata = allenamento[l][1];
                            if (clusterkmeans[l] == k) {
                                varianzaascisse += std::pow(ascissa-mediaascisse, 2);
                                varianzaordinate += std::pow(ordinata-mediaordinate, 2);
                                covarianza += (ascissa-mediaascisse)*(ordinata-mediaordinate);
                            }
                        }
                        varianzaascisse /= tagliacluster-1;
                        varianzaordinate /= tagliacluster-1;
                        covarianza /= tagliacluster-1;
                        parametri[parcomp*k+1] = mediaascisse;
                        parametri[parcomp*k+2] = mediaordinate;
                        parametri[parcomp*k+3] = varianzaascisse;
                        parametri[parcomp*k+4] = varianzaordinate;
                        parametri[parcomp*k+5] = covarianza;
                        if (!normale) {parametri[parcomp*k+6] = 5;}
                    }
                    clustercorrenti = clusterkmeans;
                }
                if (inizializzazione == "kmedoidi") {

                }
                if (inizializzazione == "casuale") {

                }
                double logverosimiglianzacorrente = 0.0;
                if (normale) {
                    for (int k = 0; k < taglia; k++) {
                        int clusteross = parcomp*clustercorrenti[k];
                        std::array<double, 2> osservazione = allenamento[k];
                        double termine = 1/(2*3.141592653589793);
                        termine /= std::sqrt(parametri[clusteross+3]*parametri[clusteross+4]-
                            std::pow(parametri[clusteross+5], 2));
                        double esponenziale = -1.0/2;
                        esponenziale *= parametri[clusteross+3]*std::pow(osservazione[0]-parametri[clusteross+1], 2)+
                            2*parametri[clusteross+5]*(osservazione[0]-parametri[clusteross+1])*(osservazione[1]-
                                parametri[clusteross+2])+parametri[clusteross+4]*std::pow(osservazione[1]-
                                    parametri[clusteross+2], 2);
                        termine *= std::exp(esponenziale);
                        logverosimiglianzacorrente += std::log(termine)+std::log(parametri[clusteross]);
                    }
                } else {
                    // Log-verosimiglianza di una t di Student
                }
                // Algoritmo EM
            }
        }
        return clusterpercomponenti;
    }
    if (selezione == "bootstrap") {
        std::vector<std::vector<int>> clusterpercomponenti(componentimassime-1, std::vector<int>(taglia));
        std::vector<std::array<double, 10>> parametripercomponenti(componentimassime-1);
        clusteringconbootstrap(2);
        int numerocomponenti = componentimassime;
        for (int i = 3; i <= componentimassime; i++) {
            clusteringconbootstrap(i);
            const double statisticaoriginale = menodueloglambda();
            std::vector<double> statisticheempiriche(campionibootstrap);
            for (int j = 0; j < campionibootstrap; j++) {
                std::vector<std::array<double, 2>> campionebootstrap = estraidamistura();
                statisticheempiriche[j] = clusteringcampionebootstrap();
            }
            std::sort(statisticheempiriche.begin(), statisticheempiriche.end());
            const double posizione = alfabootstrap*(campionibootstrap-1);
            const int sotto = static_cast<int>(std::floor(posizione));
            const int sopra = static_cast<int>(std::ceil(posizione));
            const double peso = posizione-sotto;
            const double sogliacritica = (1.0-peso)*statisticheempiriche[sotto]+peso*statisticheempiriche[sopra];
            if (statisticaoriginale <= sogliacritica) {
                numerocomponenti = i-1;
                break;
            }
        }
        return clusterpercomponenti[numerocomponenti-2];
    }
    if (selezione == "doppiobootstrap") {
        std::vector<std::vector<int>> clusterpercomponenti(componentimassime-1, std::vector<int>(taglia));
        std::vector<std::array<double, 10>> parametripercomponenti(componentimassime-1);
        clusteringconbootstrap(2);
        int numerocomponenti = componentimassime;
        for (int i = 3; i <= componentimassime; i++) {
            clusteringconbootstrap(i);
            const double statisticaoriginale = menodueloglambda();
            std::vector<double> statisticheempiriche(campionibootstrap);
            std::vector<std::array<double, 10>> parametriperbootstrap(campionibootstrap);
            std::vector<std::vector<double>> statisticheinterne(campionibootstrap, std::vector<double>(bootstrapinterni));
            for (int j = 0; j < campionibootstrap; j++) {
                std::vector<std::array<double, 2>> campionebootstrap = estraidamistura();
                clusteringcampionedoppiobootstrap();
                for (int k = 0; k < bootstrapinterni; k++) {
                    std::vector<std::array<double, 2>> campioneinterno = estraidamistura();
                    clusteringcampionedoppiobootstrap();
                }
            }
            double mediaesterna = 0.0;
            for (int j = 0; j < campionibootstrap; j++) {
                mediaesterna += statisticheempiriche[j];
            }
            mediaesterna /= campionibootstrap;
            std::vector<double> medieinterne(campionibootstrap);
            for (int j = 0; j < campionibootstrap; j++) {
                double mediainterna = 0.0;
                for (int k = 0; k < bootstrapinterni; k++) {
                    mediainterna += statisticheinterne[j][k];
                }
                mediainterna /= bootstrapinterni;
            }
            for (int j = 0; j < campionibootstrap; j++) {
                statisticheempiriche[j] = statisticheempiriche[j]-(medieinterne[j]-mediaesterna);
            }
            std::sort(statisticheempiriche.begin(), statisticheempiriche.end());
            const double posizione = alfabootstrap*(campionibootstrap-1);
            const int sotto = static_cast<int>(std::floor(posizione));
            const int sopra = static_cast<int>(std::ceil(posizione));
            const double peso = posizione-sotto;
            const double sogliacritica = (1.0-peso)*statisticheempiriche[sotto]+peso*statisticheempiriche[sopra];
            if (statisticaoriginale <= sogliacritica) {
                numerocomponenti = i-1;
                break;
            }
        }
        return clusterpercomponenti[numerocomponenti-2];
    }
    if (selezione == "crossvalidation") {
        if (fold < taglia) {
            fold = taglia;
        }
        std::vector<double> stimecrossvalidate(componentimassime-1);
        for (int i = 0; i < componentimassime-1; i++) {
            for (int j = 0; j < fold; j++) {
                std::vector<std::array<double, 2>> trainingset(taglia-taglia/fold);
                std::vector<std::array<double, 2>> testset(taglia/fold);
                std::array<double, 10> parametri = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                clusteringsultrainingset();
                stimecrossvalidate[j] += logverosimiglianzatestset();
            }
            stimecrossvalidate /= fold;
        }
        int componenti = 2;
        double massimastima = stimecrossvalidate[0];
        for (int i = 1; i < componentimassime-1; i++) {
            if (stimecrossvalidate[i] > massimastima){componenti = i+2;}
        }
        return clusteringdopocrossvalidation();
    }
    return std::vector<int>(taglia, 0);
}


PYBIND11_MODULE(miomodulo, m) {
    m.def("testhotelling", &testhotelling, "Funzione per eseguire il test di Hotelling.");
    m.def("betatesthotelling", &betatesthotelling, "Funzione per calcolare la probabilità di errore di II tipo del test di Hotelling.");
    m.def("ljungbox", &ljungbox, "Funzione per eseguire il test di Ljung-Box.");
    m.def("betaljungbox", &betaljungbox, "Funzione per calcolare la probabilità di errore di II tipo del test di Ljung-Box.");
    m.def("alfaveroljungbox", &alfaveroljungbox, "Funzione per calcolare la probabilità di errore di I tipo effettiva del test di Ljung-Box.");
    m.def("testhenzezirkler", &testhenzezirkler, "Funzione per eseguire il test di Henze-Zirkler restituendone le statistiche rilevanti.");
    m.def("testmardia", &testmardia, "Funzione per eseguire il test di Mardia.");
    m.def("betatesthenzezirkler", &betatesthenzezirkler, "Funzione per calcolare la probabilità di errore di II tipo del test di Henze-Zirkler.");
    m.def("betatestmardia", &betatestmardia, "Funzione per calcolare la probabilità di errore di II tipo del test di Mardia.");
    m.def("alfaverohenzezirkler", &alfaverohenzezirkler, "Funzione per calcolare la probabilità di errore di I tipo effettiva del test di Henze-Zirkler.");
    m.def("alfaveromardia", &alfaveromardia, "Funzione per calcolare la probabilità di errore di I tipo effettiva del test di Mardia.");
    m.def("clustering", &clustering, "Funzione per effettuare il clustering. Restituisce una lista di etichette.");
}
