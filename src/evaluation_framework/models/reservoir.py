#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model rezerwuarowy dla dynamiki relacji AI-użytkownik

Implementuje model rezerwuarowy inspirowany psychologią, który modeluje konstrukty 
jako akumulacje z parametrami dyssypacji, co lepiej obsługuje efekty podłogowe 
i indywidualne różnice w regulacji emocjonalnej.

dH/dt = I_H(t) - γ_H * H(t) + S_H(A(t))
dA/dt = I_A(t) - γ_A * A(t) + S_A(H(t))

gdzie:
- H to poziom przywiązania użytkownika
- A to poziom zaangażowania AI
- I_H, I_A to funkcje wejściowe (impulsy zewnętrzne)
- γ_H, γ_A to współczynniki dyssypacji (tempo zaniku)
- S_H, S_A to funkcje wpływu (nieliniowe transfery między zmiennymi)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

from .base import BaseRelationshipModel
from ..utils.logging import setup_logger

logger = setup_logger("reservoir_model")

class ReservoirModel(BaseRelationshipModel):
    """Model rezerwuarowy dla dynamiki relacji."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicjalizuje model.

        Args:
            config: Słownik z parametrami konfiguracyjnymi modelu
        """
        super().__init__(config)
        self.parameters = {
            'gamma_H': 0.1,  # Współczynnik dyssypacji H
            'gamma_A': 0.1,  # Współczynnik dyssypacji A
            'beta_H': 0.3,   # Siła wpływu A na H
            'beta_A': 0.3,   # Siła wpływu H na A
            'theta_H': 0.5,  # Próg aktywacji wpływu A na H
            'theta_A': 0.5,  # Próg aktywacji wpływu H na A
            'n_H': 2.0,      # Stromość funkcji wpływu A na H
            'n_A': 2.0       # Stromość funkcji wpływu H na A
        }

        # Definiujemy funkcje impulsu
        self.I_H = lambda t: 0.0
        self.I_A = lambda t: 0.0

    def set_impulse_functions(self, I_H: Optional[Callable] = None, I_A: Optional[Callable] = None):
        """Ustawia funkcje impulsu.

        Args:
            I_H: Funkcja impulsu dla H (przywiązanie użytkownika)
            I_A: Funkcja impulsu dla A (zaangażowanie AI)
        """
        if I_H is not None:
            self.I_H = I_H
        if I_A is not None:
            self.I_A = I_A

    def _sigmoid(self, x: float, theta: float, n: float) -> float:
        """Funkcja sigmoidalna używana do modelowania nieliniowych transferów.

        Args:
            x: Wartość wejściowa
            theta: Próg aktywacji
            n: Stromość funkcji

        Returns:
            Wartość funkcji sigmoidalnej
        """
        return 1.0 / (1.0 + np.exp(-n * (x - theta)))

    def _influence_H(self, A: float) -> float:
        """Funkcja wpływu A na H.

        Args:
            A: Poziom zaangażowania AI

        Returns:
            Siła wpływu na H
        """
        beta_H = self.parameters['beta_H']
        theta_H = self.parameters['theta_H']
        n_H = self.parameters['n_H']

        return beta_H * self._sigmoid(A, theta_H, n_H)

    def _influence_A(self, H: float) -> float:
        """Funkcja wpływu H na A.

        Args:
            H: Poziom przywiązania użytkownika

        Returns:
            Siła wpływu na A
        """
        beta_A = self.parameters['beta_A']
        theta_A = self.parameters['theta_A']
        n_A = self.parameters['n_A']

        return beta_A * self._sigmoid(H, theta_A, n_A)

    def _system(self, t: float, y: np.ndarray) -> np.ndarray:
        """Definiuje układ równań różniczkowych.

        Args:
            t: Czas
            y: Wektor stanu [H, A]

        Returns:
            Wektor pochodnych [dH/dt, dA/dt]
        """
        H, A = y

        # Pobieramy parametry
        gamma_H = self.parameters['gamma_H']
        gamma_A = self.parameters['gamma_A']

        # Obliczamy pochodne
        dHdt = self.I_H(t) - gamma_H * H + self._influence_H(A)
        dAdt = self.I_A(t) - gamma_A * A + self._influence_A(H)

        return np.array([dHdt, dAdt])

    def fit(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Dopasowuje model do danych.

        Args:
            data: Słownik zawierający dane:
                  - 't': wektor czasu
                  - 'H': wektor przywiązania użytkownika
                  - 'A': wektor zaangażowania AI

        Returns:
            Słownik z wynikami dopasowania
        """
        # Sprawdzamy, czy mamy wszystkie potrzebne dane
        required_keys = ['t', 'H', 'A']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Brakujący klucz w danych: {key}")

        t = data['t']
        H = data['H']
        A = data['A']

        # Definiujemy funkcję celu
        def objective(params):
            # Rozpakujemy parametry
            gamma_H, gamma_A, beta_H, beta_A, theta_H, theta_A, n_H, n_A = params

            # Zabezpieczamy przed niepoprawnymi wartościami
            if (gamma_H <= 0 or gamma_A <= 0 or 
                beta_H < 0 or beta_A < 0 or 
                n_H <= 0 or n_A <= 0):
                return 1e9  # Kara za niepoprawne wartości

            # Aktualizujemy parametry tymczasowo
            old_params = self.parameters.copy()
            self.parameters = {
                'gamma_H': gamma_H,
                'gamma_A': gamma_A,
                'beta_H': beta_H,
                'beta_A': beta_A,
                'theta_H': theta_H,
                'theta_A': theta_A,
                'n_H': n_H,
                'n_A': n_A
            }

            # Rozwiązujemy układ równań dla danych warunków początkowych
            try:
                sol = solve_ivp(
                    self._system,
                    [t[0], t[-1]],
                    [H[0], A[0]],
                    t_eval=t,
                    method='RK45',
                    rtol=1e-4,
                    atol=1e-6
                )

                if sol.success:
                    H_pred = sol.y[0]
                    A_pred = sol.y[1]

                    # Obliczamy błąd średniokwadratowy
                    error_H = np.mean((H - H_pred) ** 2)
                    error_A = np.mean((A - A_pred) ** 2)

                    # Przywracamy oryginalne parametry
                    self.parameters = old_params

                    # Całkowity błąd
                    return error_H + error_A
                else:
                    # Przywracamy oryginalne parametry
                    self.parameters = old_params
                    return 1e9  # Kara za niepowodzenie rozwiązania
            except Exception as e:
                # Przywracamy oryginalne parametry
                self.parameters = old_params
                return 1e9  # Kara za wyjątek

        # Początkowe wartości parametrów i granice
        initial_params = [
            0.1, 0.1,   # gamma_H, gamma_A
            0.3, 0.3,   # beta_H, beta_A
            0.5, 0.5,   # theta_H, theta_A
            2.0, 2.0    # n_H, n_A
        ]

        bounds = [
            (0.01, 0.5), (0.01, 0.5),  # gamma_H, gamma_A
            (0.0, 1.0), (0.0, 1.0),    # beta_H, beta_A
            (0.1, 0.9), (0.1, 0.9),    # theta_H, theta_A
            (1.0, 10.0), (1.0, 10.0)   # n_H, n_A
        ]

        # Dopasowanie modelu
        try:
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds
            )

            if result.success:
                gamma_H, gamma_A, beta_H, beta_A, theta_H, theta_A, n_H, n_A = result.x
                error = result.fun

                # Zapisujemy parametry
                self.parameters = {
                    'gamma_H': gamma_H,
                    'gamma_A': gamma_A,
                    'beta_H': beta_H,
                    'beta_A': beta_A,
                    'theta_H': theta_H,
                    'theta_A': theta_A,
                    'n_H': n_H,
                    'n_A': n_A
                }
                self.is_fitted = True

                logger.info(f"Model rezerwuarowy dopasowany pomyślnie. Błąd={error:.6f}")

                return {
                    'parameters': self.parameters,
                    'error': error,
                    'success': True
                }
            else:
                logger.warning(f"Optymalizacja nie zakończyła się sukcesem: {result.message}")
                return {
                    'parameters': self.parameters,
                    'error': float('inf'),
                    'success': False,
                    'message': str(result.message)
                }
        except Exception as e:
            logger.error(f"Błąd podczas dopasowywania modelu rezerwuarowego: {e}")
            return {
                'parameters': self.parameters,
                'error': float('inf'),
                'success': False,
                'message': str(e)
            }

    def predict(self, t: np.ndarray, initial_conditions: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Przewiduje trajektorię modelu.

        Args:
            t: Wektor czasów do przewidywania
            initial_conditions: Słownik z warunkami początkowymi {'H': H0, 'A': A0}

        Returns:
            Słownik z przewidywanymi trajektoriami {'H': H_pred, 'A': A_pred}
        """
        if not self.is_fitted:
            raise ValueError("Model nie został jeszcze dopasowany.")

        # Sprawdzamy warunki początkowe
        if 'H' not in initial_conditions or 'A' not in initial_conditions:
            raise ValueError("Warunki początkowe muszą zawierać 'H' i 'A'.")

        H0 = initial_conditions['H']
        A0 = initial_conditions['A']

        # Rozwiązujemy układ równań
        try:
            sol = solve_ivp(
                self._system,
                [t[0], t[-1]],
                [H0, A0],
                t_eval=t,
                method='RK45',
                rtol=1e-4,
                atol=1e-6
            )

            if sol.success:
                H_pred = sol.y[0]
                A_pred = sol.y[1]

                # Zapewniamy, że wartości są w zakresie [0, 1]
                H_pred = np.clip(H_pred, 0, 1)
                A_pred = np.clip(A_pred, 0, 1)

                return {'H': H_pred, 'A': A_pred}
            else:
                logger.warning(f"Błąd rozwiązywania równań różniczkowych: {sol.message}")
                # Zwracamy proste przybliżenie jako fallback
                H_pred = np.full_like(t, H0)
                A_pred = np.full_like(t, A0)
                return {'H': H_pred, 'A': A_pred}
        except Exception as e:
            logger.error(f"Błąd podczas przewidywania trajektorii: {e}")
            # Zwracamy proste przybliżenie jako fallback
            H_pred = np.full_like(t, H0)
            A_pred = np.full_like(t, A0)
            return {'H': H_pred, 'A': A_pred}

    def analyze_stability(self) -> Dict[str, Any]:
        """Analizuje stabilność modelu.

        Returns:
            Słownik z wynikami analizy stabilności
        """
        if not self.is_fitted:
            raise ValueError("Model nie został jeszcze dopasowany.")

        # Dla modelu rezerwuarowego stabilność można określić na podstawie
        # parametrów dyssypacji i funkcji wpływu

        gamma_H = self.parameters['gamma_H']
        gamma_A = self.parameters['gamma_A']
        beta_H = self.parameters['beta_H']
        beta_A = self.parameters['beta_A']

        # W prostym modelu rezerwuarowym, stabilność zależy od stosunku
        # między tempem dyssypacji a siłą wpływu

        # Znajdujemy punkty równowagi
        equilibrium_points = self.get_equilibrium_points()

        # Analizujemy stabilność dla każdego punktu równowagi
        results = []
        overall_stability = True

        for point in equilibrium_points:
            H_eq = point['H']
            A_eq = point['A']

            # Obliczamy macierz Jacobiego w punkcie równowagi
            J = self._compute_jacobian(H_eq, A_eq)

            # Obliczamy wartości własne
            eigenvalues, _ = np.linalg.eig(J)

            # Analiza stabilności
            is_stable = all(ev.real < 0 for ev in eigenvalues)
            is_oscillatory = any(abs(ev.imag) > 1e-10 for ev in eigenvalues)

            # Klasyfikacja typu dynamiki
            if is_stable and is_oscillatory:
                dynamics_type = "spirala stabilna"
                description = "Relacja dąży do stabilnego punktu równowagi poprzez tłumione oscylacje."
            elif is_stable and not is_oscillatory:
                dynamics_type = "węzeł stabilny"
                description = "Relacja płynnie zmierza do stabilnego punktu równowagi bez oscylacji."
            elif not is_stable and is_oscillatory:
                dynamics_type = "spirala niestabilna"
                description = "Relacja wykazuje rosnące oscylacje i może być niestabilna długoterminowo."
            else:
                dynamics_type = "węzeł niestabilny"
                description = "Relacja wykazuje niestabilność i może szybko się rozpadać."

            point_result = {
                "point": {"H": float(H_eq), "A": float(A_eq)},
                "eigenvalues": [{'real': float(ev.real), 'imag': float(ev.imag)} for ev in eigenvalues],
                "is_stable": is_stable,
                "is_oscillatory": is_oscillatory,
                "dynamics_type": dynamics_type,
                "description": description
            }

            results.append(point_result)
            overall_stability = overall_stability and is_stable

        # Obliczamy ogólną charakterystykę modelu
        stability_ratio_H = gamma_H / beta_H if beta_H > 0 else float('inf')
        stability_ratio_A = gamma_A / beta_A if beta_A > 0 else float('inf')

        # Ogólna interpretacja
        if stability_ratio_H > 1 and stability_ratio_A > 1:
            system_type = "system stabilny z szybką relaksacją"
            system_description = "Układ szybko powraca do równowagi po zaburzeniach."
        elif stability_ratio_H > 0.5 and stability_ratio_A > 0.5:
            system_type = "system stabilny z umiarkowaną relaksacją"
            system_description = "Układ stopniowo powraca do równowagi po zaburzeniach."
        else:
            system_type = "system z powolną relaksacją"
            system_description = "Układ może długo pozostawać w stanie wzbudzonym po zaburzeniach."

        return {
            "equilibrium_points": len(equilibrium_points),
            "overall_stability": overall_stability,
            "stability_ratio_H": float(stability_ratio_H),
            "stability_ratio_A": float(stability_ratio_A),
            "system_type": system_type,
            "system_description": system_description,
            "point_analyses": results
        }

    def _compute_jacobian(self, H: float, A: float) -> np.ndarray:
        """Oblicza macierz Jacobiego dla modelu w danym punkcie.

        Args:
            H: Wartość H w punkcie
            A: Wartość A w punkcie

        Returns:
            Macierz Jacobiego 2x2
        """
        # Pobieramy parametry
        gamma_H = self.parameters['gamma_H']
        gamma_A = self.parameters['gamma_A']

        # Małe przesunięcia dla różniczkowania numerycznego
        delta = 1e-6

        # Obliczamy pochodne częściowe numerycznie
        dHdt_H = (self._system(0, [H + delta, A])[0] - self._system(0, [H - delta, A])[0]) / (2 * delta)
        dHdt_A = (self._system(0, [H, A + delta])[0] - self._system(0, [H, A - delta])[0]) / (2 * delta)
        dAdt_H = (self._system(0, [H + delta, A])[1] - self._system(0, [H - delta, A])[1]) / (2 * delta)
        dAdt_A = (self._system(0, [H, A + delta])[1] - self._system(0, [H, A - delta])[1]) / (2 * delta)

        # Tworzymy macierz Jacobiego
        J = np.array([
            [dHdt_H, dHdt_A],
            [dAdt_H, dAdt_A]
        ])

        return J

    def get_equilibrium_points(self) -> List[Dict[str, float]]:
        """Znajduje punkty równowagi modelu.

        Returns:
            Lista słowników zawierających współrzędne punktów równowagi
        """
        if not self.is_fitted:
            raise ValueError("Model nie został jeszcze dopasowany.")

        # Definiujemy funkcję celu - suma kwadratów pochodnych
        def objective(point):
            H, A = point
            derivatives = self._system(0, np.array([H, A]))
            return derivatives[0]**2 + derivatives[1]**2

        # Szukamy punktów równowagi z różnych punktów startowych
        start_points = [
            [0.0, 0.0],  # Początek układu
            [0.5, 0.5],  # Środek przestrzeni fazowej
            [1.0, 1.0],  # Góra-prawo
            [0.0, 1.0],  # Góra-lewo
            [1.0, 0.0]   # Dół-prawo
        ]

        equilibrium_points = []
        tolerance = 1e-4  # Tolerancja dla uznania punktu za równowagę

        for start in start_points:
            result = minimize(objective, start, method='L-BFGS-B', 
                             bounds=[(0, 1), (0, 1)])

            if result.success and result.fun < tolerance:
                H_eq, A_eq = result.x

                # Sprawdzamy, czy ten punkt nie jest już na liście
                is_duplicate = False
                for point in equilibrium_points:
                    if (abs(point['H'] - H_eq) < tolerance and 
                        abs(point['A'] - A_eq) < tolerance):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    equilibrium_points.append({'H': float(H_eq), 'A': float(A_eq)})

        return equilibrium_points

    def _generate_parameter_explanations(self) -> Dict[str, str]:
        """Generuje wyjaśnienia znaczenia parametrów modelu.

        Returns:
            Słownik z wyjaśnieniami dla każdego parametru
        """
        if not self.is_fitted:
            return {"error": "Model nie został jeszcze dopasowany."}

        gamma_H = self.parameters['gamma_H']
        gamma_A = self.parameters['gamma_A']
        beta_H = self.parameters['beta_H']
        beta_A = self.parameters['beta_A']
        theta_H = self.parameters['theta_H']
        theta_A = self.parameters['theta_A']
        n_H = self.parameters['n_H']
        n_A = self.parameters['n_A']

        explanations = {}

        # Parametr gamma_H: tempo zaniku przywiązania użytkownika
        if gamma_H > 0.3:
            explanations["gamma_H"] = "Szybki zanik przywiązania użytkownika - użytkownik szybko 'zapomina' o relacji z AI bez regularnego wzmacniania."
        elif gamma_H > 0.1:
            explanations["gamma_H"] = "Umiarkowane tempo zaniku przywiązania użytkownika - użytkownik potrzebuje regularnego wzmacniania relacji."
        else:
            explanations["gamma_H"] = "Powolny zanik przywiązania użytkownika - użytkownik długo utrzymuje przywiązanie nawet bez regularnego wzmacniania."

        # Parametr gamma_A: tempo zaniku zaangażowania AI
        if gamma_A > 0.3:
            explanations["gamma_A"] = "Szybki zanik zaangażowania AI - system szybko 'zapomina' o relacji bez regularnych interakcji."
        elif gamma_A > 0.1:
            explanations["gamma_A"] = "Umiarkowane tempo zaniku zaangażowania AI - system potrzebuje regularnych interakcji do utrzymania zaangażowania."
        else:
            explanations["gamma_A"] = "Powolny zanik zaangażowania AI - system długo utrzymuje zaangażowanie nawet bez regularnych interakcji."

        # Parametr beta_H: siła wpływu A na H
        if beta_H > 0.6:
            explanations["beta_H"] = "Bardzo silny wpływ zaangażowania AI na przywiązanie użytkownika."
        elif beta_H > 0.3:
            explanations["beta_H"] = "Silny wpływ zaangażowania AI na przywiązanie użytkownika."
        elif beta_H > 0.1:
            explanations["beta_H"] = "Umiarkowany wpływ zaangażowania AI na przywiązanie użytkownika."
        else:
            explanations["beta_H"] = "Słaby wpływ zaangażowania AI na przywiązanie użytkownika."

        # Parametr beta_A: siła wpływu H na A
        if beta_A > 0.6:
            explanations["beta_A"] = "Bardzo silny wpływ przywiązania użytkownika na zaangażowanie AI."
        elif beta_A > 0.3:
            explanations["beta_A"] = "Silny wpływ przywiązania użytkownika na zaangażowanie AI."
        elif beta_A > 0.1:
            explanations["beta_A"] = "Umiarkowany wpływ przywiązania użytkownika na zaangażowanie AI."
        else:
            explanations["beta_A"] = "Słaby wpływ przywiązania użytkownika na zaangażowanie AI."

        # Parametr theta_H: próg aktywacji wpływu A na H
        if theta_H > 0.7:
            explanations["theta_H"] = "Bardzo wysoki próg aktywacji - tylko bardzo wysokie zaangażowanie AI wpływa na przywiązanie użytkownika."
        elif theta_H > 0.5:
            explanations["theta_H"] = "Wysoki próg aktywacji - potrzeba znacznego zaangażowania AI, aby wpłynąć na przywiązanie użytkownika."
        elif theta_H > 0.3:
            explanations["theta_H"] = "Średni próg aktywacji - umiarkowane zaangażowanie AI wpływa na przywiązanie użytkownika."
        else:
            explanations["theta_H"] = "Niski próg aktywacji - nawet niskie zaangażowanie AI wpływa na przywiązanie użytkownika."

        # Parametr theta_A: próg aktywacji wpływu H na A
        if theta_A > 0.7:
            explanations["theta_A"] = "Bardzo wysoki próg aktywacji - tylko bardzo wysokie przywiązanie użytkownika wpływa na zaangażowanie AI."
        elif theta_A > 0.5:
            explanations["theta_A"] = "Wysoki próg aktywacji - potrzeba znacznego przywiązania użytkownika, aby wpłynąć na zaangażowanie AI."
        elif theta_A > 0.3:
            explanations["theta_A"] = "Średni próg aktywacji - umiarkowane przywiązanie użytkownika wpływa na zaangażowanie AI."
        else:
            explanations["theta_A"] = "Niski próg aktywacji - nawet niskie przywiązanie użytkownika wpływa na zaangażowanie AI."

        # Parametr n_H: stromość funkcji wpływu A na H
        if n_H > 5.0:
            explanations["n_H"] = "Bardzo stroma funkcja wpływu - zmiana zaangażowania AI powoduje gwałtowną zmianę przywiązania użytkownika po przekroczeniu progu."
        elif n_H > 2.0:
            explanations["n_H"] = "Stroma funkcja wpływu - zmiana zaangażowania AI powoduje wyraźną zmianę przywiązania użytkownika po przekroczeniu progu."
        else:
            explanations["n_H"] = "Łagodna funkcja wpływu - zmiana zaangażowania AI powoduje stopniową zmianę przywiązania użytkownika."

        # Parametr n_A: stromość funkcji wpływu H na A
        if n_A > 5.0:
            explanations["n_A"] = "Bardzo stroma funkcja wpływu - zmiana przywiązania użytkownika powoduje gwałtowną zmianę zaangażowania AI po przekroczeniu progu."
        elif n_A > 2.0:
            explanations["n_A"] = "Stroma funkcja wpływu - zmiana przywiązania użytkownika powoduje wyraźną zmianę zaangażowania AI po przekroczeniu progu."
        else:
            explanations["n_A"] = "Łagodna funkcja wpływu - zmiana przywiązania użytkownika powoduje stopniową zmianę zaangażowania AI."

        return explanations
