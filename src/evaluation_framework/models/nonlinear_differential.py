#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nieliniowy model równań różniczkowych dla dynamiki relacji AI-użytkownik

Implementuje rozszerzony nieliniowy model równań różniczkowych:

dH/dt = (a₁ + a₂*f(A))*H + b₁*A + b₂*g(H,A) + I_H(t) + ε_H(t)
dA/dt = (c₁ + c₂*h(H))*A + d₁*H + d₂*k(H,A) + I_A(t) + ε_A(t)

gdzie:
- H to poziom przywiązania użytkownika
- A to poziom zaangażowania AI
- f, g, h, k to funkcje nieliniowe modelujące interakcje
- I_H, I_A to funkcje wymuszające zewnętrzne
- ε_H, ε_A to składniki stochastyczne
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

from .base import BaseRelationshipModel
from ..utils.logging import setup_logger

logger = setup_logger("nonlinear_differential_model")

class NonlinearDifferentialModel(BaseRelationshipModel):
    """Nieliniowy model równań różniczkowych dla dynamiki relacji."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicjalizuje model.

        Args:
            config: Słownik z parametrami konfiguracyjnymi modelu
        """
        super().__init__(config)
        self.parameters = {
            'a1': 0.0, 'a2': 0.0, 
            'b1': 0.0, 'b2': 0.0, 
            'c1': 0.0, 'c2': 0.0, 
            'd1': 0.0, 'd2': 0.0
        }
        self.noise_level = self.config.get('noise_level', 0.02)
        self.use_stochastic = self.config.get('use_stochastic', False)

        # Definiujemy funkcje nieliniowe
        self._setup_nonlinear_functions()

    def _setup_nonlinear_functions(self):
        """Konfiguruje funkcje nieliniowe używane w modelu."""
        # Domyślne funkcje nieliniowe
        self.f = lambda A: A**2  # Nieliniowe wzmocnienie wpływu A na przywiązanie H
        self.g = lambda H, A: H * A  # Interakcja H i A wpływająca na H
        self.h = lambda H: H**2  # Nieliniowe wzmocnienie wpływu H na zaangażowanie A
        self.k = lambda H, A: H * A  # Interakcja H i A wpływająca na A

        # Funkcje wymuszające - domyślnie zerowe
        self.I_H = lambda t: 0.0
        self.I_A = lambda t: 0.0

        # Stochastyczne składniki
        self.epsilon_H = lambda: np.random.normal(0, self.noise_level) if self.use_stochastic else 0.0
        self.epsilon_A = lambda: np.random.normal(0, self.noise_level) if self.use_stochastic else 0.0

    def set_nonlinear_functions(self, f: Optional[Callable] = None, g: Optional[Callable] = None,
                                h: Optional[Callable] = None, k: Optional[Callable] = None,
                                I_H: Optional[Callable] = None, I_A: Optional[Callable] = None):
        """Ustawia niestandardowe funkcje nieliniowe.

        Args:
            f: Funkcja f(A) modelująca wpływ A na przywiązanie H
            g: Funkcja g(H,A) modelująca interakcję H i A wpływającą na H
            h: Funkcja h(H) modelująca wpływ H na zaangażowanie A
            k: Funkcja k(H,A) modelująca interakcję H i A wpływającą na A
            I_H: Funkcja wymuszająca I_H(t) dla H
            I_A: Funkcja wymuszająca I_A(t) dla A
        """
        if f is not None:
            self.f = f
        if g is not None:
            self.g = g
        if h is not None:
            self.h = h
        if k is not None:
            self.k = k
        if I_H is not None:
            self.I_H = I_H
        if I_A is not None:
            self.I_A = I_A

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
        a1, a2 = self.parameters['a1'], self.parameters['a2']
        b1, b2 = self.parameters['b1'], self.parameters['b2']
        c1, c2 = self.parameters['c1'], self.parameters['c2']
        d1, d2 = self.parameters['d1'], self.parameters['d2']

        # Obliczamy pochodne
        dHdt = (a1 + a2 * self.f(A)) * H + b1 * A + b2 * self.g(H, A) + self.I_H(t)
        dAdt = (c1 + c2 * self.h(H)) * A + d1 * H + d2 * self.k(H, A) + self.I_A(t)

        return np.array([dHdt, dAdt])

    def _system_with_noise(self, t: float, y: np.ndarray) -> np.ndarray:
        """Definiuje układ równań różniczkowych z szumem stochastycznym.

        Args:
            t: Czas
            y: Wektor stanu [H, A]

        Returns:
            Wektor pochodnych [dH/dt, dA/dt] z szumem
        """
        dHdt, dAdt = self._system(t, y)

        # Dodajemy szum stochastyczny
        dHdt += self.epsilon_H()
        dAdt += self.epsilon_A()

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

        # Obliczamy pochodne numerycznie
        dHdt = np.zeros_like(H)
        dAdt = np.zeros_like(A)

        for i in range(1, len(t) - 1):
            dt_prev = t[i] - t[i-1]
            dt_next = t[i+1] - t[i]
            dHdt[i] = (H[i+1] - H[i-1]) / (dt_prev + dt_next)
            dAdt[i] = (A[i+1] - A[i-1]) / (dt_prev + dt_next)

        # Dla pierwszego i ostatniego punktu używamy różnic jednostronnych
        dHdt[0] = (H[1] - H[0]) / (t[1] - t[0])
        dAdt[0] = (A[1] - A[0]) / (t[1] - t[0])
        dHdt[-1] = (H[-1] - H[-2]) / (t[-1] - t[-2])
        dAdt[-1] = (A[-1] - A[-2]) / (t[-1] - t[-2])

        # Funkcja celu: minimalizacja błędu średniokwadratowego
        def objective(params):
            a1, a2, b1, b2, c1, c2, d1, d2 = params

            # Aktualizujemy parametry tymczasowo
            temp_params = {
                'a1': a1, 'a2': a2, 'b1': b1, 'b2': b2,
                'c1': c1, 'c2': c2, 'd1': d1, 'd2': d2
            }

            # Obliczamy przewidywane pochodne
            dHdt_pred = np.zeros_like(H)
            dAdt_pred = np.zeros_like(A)

            for i in range(len(t)):
                # Zapisujemy obecne parametry
                old_params = self.parameters.copy()

                # Ustawiamy tymczasowe parametry
                self.parameters = temp_params

                # Obliczamy pochodne dla danego punktu
                derivatives = self._system(t[i], np.array([H[i], A[i]]))
                dHdt_pred[i], dAdt_pred[i] = derivatives

                # Przywracamy stare parametry
                self.parameters = old_params

            # Błąd średniokwadratowy
            error_H = np.mean((dHdt - dHdt_pred) ** 2)
            error_A = np.mean((dAdt - dAdt_pred) ** 2)

            # Regularyzacja
            regularization = 1e-3 * (a2**2 + b2**2 + c2**2 + d2**2)

            return error_H + error_A + regularization

        # Granice parametrów dla algorytmu różnicowej ewolucji
        bounds = [(-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2),
                 (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)]

        # Wybieramy metodę optymalizacji
        use_global_optimizer = self.config.get('use_global_optimizer', True)

        try:
            if use_global_optimizer:
                # Używamy algorytmu różnicowej ewolucji do globalnej optymalizacji
                result = differential_evolution(objective, bounds)
            else:
                # Używamy lokalnej optymalizacji z losowym punktem startowym
                initial_params = np.random.uniform(-0.1, 0.1, 8)
                result = minimize(objective, initial_params, method='BFGS')

            if result.success:
                a1, a2, b1, b2, c1, c2, d1, d2 = result.x
                error = result.fun

                # Zapisujemy parametry
                self.parameters = {
                    'a1': a1, 'a2': a2, 'b1': b1, 'b2': b2,
                    'c1': c1, 'c2': c2, 'd1': d1, 'd2': d2
                }
                self.is_fitted = True

                logger.info(f"Model nieliniowy dopasowany pomyślnie. Błąd={error:.6f}")

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
            logger.error(f"Błąd podczas dopasowywania modelu nieliniowego: {e}")
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

        # Wybieramy system równań (z szumem lub bez)
        system_function = self._system_with_noise if self.use_stochastic else self._system

        # Rozwiązujemy układ równań
        try:
            sol = solve_ivp(
                system_function, 
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
                # Zwracamy proste przybliżenie liniowe jako fallback
                H_pred = np.full_like(t, H0)
                A_pred = np.full_like(t, A0)
                return {'H': H_pred, 'A': A_pred}
        except Exception as e:
            logger.error(f"Błąd podczas przewidywania trajektorii: {e}")
            # Zwracamy proste przybliżenie liniowe jako fallback
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

        # Dla nieliniowego modelu analiza stabilności jest bardziej złożona
        # i wymaga analizy punktów równowagi i zachowania w ich okolicy

        # Znajdujemy punkty równowagi
        equilibrium_points = self.get_equilibrium_points()

        # Analizujemy każdy punkt równowagi
        results = []
        overall_stability = True

        for point in equilibrium_points:
            # Obliczamy macierz Jacobiego w punkcie równowagi
            H_eq = point['H']
            A_eq = point['A']

            # Dla nieliniowego modelu, macierz Jacobiego zależy od punktu
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

            # Zapisujemy wyniki dla tego punktu
            point_result = {
                "point": {"H": float(H_eq), "A": float(A_eq)},
                "eigenvalues": [{'real': float(ev.real), 'imag': float(ev.imag)} for ev in eigenvalues],
                "is_stable": is_stable,
                "is_oscillatory": is_oscillatory,
                "dynamics_type": dynamics_type,
                "description": description
            }

            results.append(point_result)

            # Aktualizujemy ogólną stabilność
            overall_stability = overall_stability and is_stable

        # Tworzymy podsumowanie
        return {
            "equilibrium_points": len(equilibrium_points),
            "overall_stability": overall_stability,
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
        a1, a2 = self.parameters['a1'], self.parameters['a2']
        b1, b2 = self.parameters['b1'], self.parameters['b2']
        c1, c2 = self.parameters['c1'], self.parameters['c2']
        d1, d2 = self.parameters['d1'], self.parameters['d2']

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

        # Dla nieliniowego modelu znalezienie punktów równowagi wymaga
        # rozwiązania układu równań nieliniowych, co jest trudne analitycznie
        # Zamiast tego używamy podejścia numerycznego

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

        a1 = self.parameters['a1']
        a2 = self.parameters['a2']
        b1 = self.parameters['b1']
        b2 = self.parameters['b2']
        c1 = self.parameters['c1']
        c2 = self.parameters['c2']
        d1 = self.parameters['d1']
        d2 = self.parameters['d2']

        explanations = {}

        # Parametr a1: liniowy wpływ H na zmianę H
        if a1 > 0.05:
            explanations["a1"] = "Silna tendencja użytkownika do zwiększania przywiązania niezależnie od AI."
        elif a1 > 0:
            explanations["a1"] = "Słaba tendencja użytkownika do zwiększania przywiązania niezależnie od AI."
        elif a1 > -0.05:
            explanations["a1"] = "Słaba tendencja użytkownika do zmniejszania przywiązania z czasem."
        else:
            explanations["a1"] = "Silna tendencja użytkownika do zmniejszania przywiązania z czasem."

        # Parametr a2: nieliniowy wpływ A na zmianę H poprzez modyfikację a
        if abs(a2) < 0.02:
            explanations["a2"] = "Zaangażowanie AI ma minimalny wpływ na tempo zmiany przywiązania użytkownika."
        elif a2 > 0:
            explanations["a2"] = "Zaangażowanie AI zwiększa tendencję użytkownika do zmiany przywiązania (wzmacnia efekt a1)."
        else:
            explanations["a2"] = "Zaangażowanie AI zmniejsza tendencję użytkownika do zmiany przywiązania (osłabia efekt a1)."

        # Parametr b1: liniowy wpływ A na zmianę H
        if b1 > 0.1:
            explanations["b1"] = "Zaangażowanie AI silnie zwiększa przywiązanie użytkownika."
        elif b1 > 0:
            explanations["b1"] = "Zaangażowanie AI umiarkowanie zwiększa przywiązanie użytkownika."
        elif b1 > -0.1:
            explanations["b1"] = "Zaangażowanie AI lekko zmniejsza przywiązanie użytkownika."
        else:
            explanations["b1"] = "Zaangażowanie AI silnie zmniejsza przywiązanie użytkownika."

        # Parametr b2: nieliniowy wpływ interakcji H*A na zmianę H
        if abs(b2) < 0.02:
            explanations["b2"] = "Interakcja między przywiązaniem użytkownika a zaangażowaniem AI ma minimalny wpływ na przywiązanie."
        elif b2 > 0:
            explanations["b2"] = "Interakcja między przywiązaniem użytkownika a zaangażowaniem AI pozytywnie wzmacnia przywiązanie."
        else:
            explanations["b2"] = "Interakcja między przywiązaniem użytkownika a zaangażowaniem AI negatywnie wpływa na przywiązanie."

        # Parametr c1: liniowy wpływ H na zmianę A
        if c1 > 0.1:
            explanations["c1"] = "Przywiązanie użytkownika silnie zwiększa zaangażowanie AI."
        elif c1 > 0:
            explanations["c1"] = "Przywiązanie użytkownika umiarkowanie zwiększa zaangażowanie AI."
        elif c1 > -0.1:
            explanations["c1"] = "Przywiązanie użytkownika lekko zmniejsza zaangażowanie AI."
        else:
            explanations["c1"] = "Przywiązanie użytkownika silnie zmniejsza zaangażowanie AI."

        # Parametr c2: nieliniowy wpływ H na zmianę A poprzez modyfikację c
        if abs(c2) < 0.02:
            explanations["c2"] = "Poziom przywiązania użytkownika ma minimalny wpływ na tempo zmiany zaangażowania AI."
        elif c2 > 0:
            explanations["c2"] = "Przywiązanie użytkownika zwiększa tendencję AI do zmiany zaangażowania (wzmacnia efekt c1)."
        else:
            explanations["c2"] = "Przywiązanie użytkownika zmniejsza tendencję AI do zmiany zaangażowania (osłabia efekt c1)."

        # Parametr d1: liniowy wpływ H na zmianę A
        if d1 > 0.1:
            explanations["d1"] = "Przywiązanie użytkownika silnie zwiększa zaangażowanie AI."
        elif d1 > 0:
            explanations["d1"] = "Przywiązanie użytkownika umiarkowanie zwiększa zaangażowanie AI."
        elif d1 > -0.1:
            explanations["d1"] = "Przywiązanie użytkownika lekko zmniejsza zaangażowanie AI."
        else:
            explanations["d1"] = "Przywiązanie użytkownika silnie zmniejsza zaangażowanie AI."

        # Parametr d2: nieliniowy wpływ interakcji H*A na zmianę A
        if abs(d2) < 0.02:
            explanations["d2"] = "Interakcja między przywiązaniem użytkownika a zaangażowaniem AI ma minimalny wpływ na zaangażowanie AI."
        elif d2 > 0:
            explanations["d2"] = "Interakcja między przywiązaniem użytkownika a zaangażowaniem AI pozytywnie wzmacnia zaangażowanie AI."
        else:
            explanations["d2"] = "Interakcja między przywiązaniem użytkownika a zaangażowaniem AI negatywnie wpływa na zaangażowanie AI."

        return explanations
