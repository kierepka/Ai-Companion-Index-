#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Liniowy model równań różniczkowych dla dynamiki relacji AI-użytkownik

Implementuje podstawowy model równań różniczkowych:

dH/dt = a*H + b*A
dA/dt = c*H + d*A

gdzie:
- H to poziom przywiązania użytkownika
- A to poziom zaangażowania AI
- a, b, c, d to parametry modelu
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union

from .base import BaseRelationshipModel
from ..utils.logging import setup_logger

logger = setup_logger("linear_differential_model")

class LinearDifferentialModel(BaseRelationshipModel):
    """Liniowy model równań różniczkowych dla dynamiki relacji."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicjalizuje model.

        Args:
            config: Słownik z parametrami konfiguracyjnymi modelu
        """
        super().__init__(config)
        self.parameters = {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0}

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
            a, b, c, d = params
            # Przewidywane pochodne
            dHdt_pred = a * H + b * A
            dAdt_pred = c * H + d * A
            # Błąd średniokwadratowy
            error_H = np.mean((dHdt - dHdt_pred) ** 2)
            error_A = np.mean((dAdt - dAdt_pred) ** 2)
            return error_H + error_A

        # Początkowe wartości parametrów
        initial_params = [0.0, 0.0, 0.0, 0.0]

        # Wybieramy metodę optymalizacji
        optimizer = self.config.get('optimizer', 'BFGS')

        # Dopasowanie modelu
        try:
            result = minimize(objective, initial_params, method=optimizer)

            if result.success:
                a, b, c, d = result.x
                error = result.fun

                # Zapisujemy parametry
                self.parameters = {'a': a, 'b': b, 'c': c, 'd': d}
                self.is_fitted = True

                logger.info(f"Model dopasowany pomyślnie. Parametry: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}, błąd={error:.6f}")

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
            logger.error(f"Błąd podczas dopasowywania modelu: {e}")
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

        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        d = self.parameters['d']

        # Definiujemy układ równań różniczkowych
        def system(t, y):
            H, A = y
            dHdt = a * H + b * A
            dAdt = c * H + d * A
            return [dHdt, dAdt]

        # Rozwiązujemy układ równań
        try:
            sol = solve_ivp(system, [t[0], t[-1]], [H0, A0], t_eval=t)

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

        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        d = self.parameters['d']

        # Macierz współczynników
        A_matrix = np.array([[a, b], [c, d]])

        # Obliczanie wartości własnych
        try:
            eigenvalues, eigenvectors = np.linalg.eig(A_matrix)

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

            # Punkt równowagi
            det = a*d - b*c
            if abs(det) > 1e-10:  # Sprawdzamy, czy wyznacznik jest niezerowy
                equilibrium = "istnieje niezerowy punkt równowagi"
            else:
                equilibrium = "brak niezerowego punktu równowagi"

            return {
                "eigenvalues": [{'real': float(ev.real), 'imag': float(ev.imag)} for ev in eigenvalues],
                "is_stable": is_stable,
                "is_oscillatory": is_oscillatory,
                "dynamics_type": dynamics_type,
                "description": description,
                "equilibrium": equilibrium,
                "determinant": float(det)
            }
        except Exception as e:
            logger.error(f"Błąd podczas analizy stabilności: {e}")
            return {
                "error": str(e),
                "is_stable": False,
                "is_oscillatory": False,
                "dynamics_type": "nieokreślony",
                "description": "Nie udało się przeprowadzić analizy stabilności."
            }

    def get_equilibrium_points(self) -> List[Dict[str, float]]:
        """Oblicza punkty równowagi modelu.

        Returns:
            Lista słowników zawierających współrzędne punktów równowagi
        """
        if not self.is_fitted:
            raise ValueError("Model nie został jeszcze dopasowany.")

        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        d = self.parameters['d']

        # Dla modelu liniowego może istnieć tylko jeden punkt równowagi w (0,0)
        # lub nieskończenie wiele punktów (gdy det(A) = 0), lub jeden niezerowy punkt

        det = a*d - b*c

        if abs(det) < 1e-10:  # Wyznacznik bliski zeru
            # Albo tylko (0,0) jest punktem równowagi, albo istnieje linia punktów równowagi
            if abs(a) < 1e-10 and abs(b) < 1e-10 and abs(c) < 1e-10 and abs(d) < 1e-10:
                # Wszystkie współczynniki są zerowe - cała przestrzeń fazowa to punkty równowagi
                return [{'H': 0.0, 'A': 0.0, 'type': 'continuum'}]
            else:
                # Tylko (0,0) jest punktem równowagi
                return [{'H': 0.0, 'A': 0.0, 'type': 'trivial'}]
        else:
            # Rozwiązujemy układ równań:
            # a*H + b*A = 0
            # c*H + d*A = 0
            # Ponieważ det(A) ≠ 0, jedynym rozwiązaniem jest (0,0)
            return [{'H': 0.0, 'A': 0.0, 'type': 'unique'}]

    def plot_phase_portrait(self, data: Optional[Dict[str, np.ndarray]] = None, 
                           save_path: Optional[str] = None, 
                           show_plot: bool = False) -> Optional[plt.Figure]:
        """Tworzy portret fazowy dla modelu.

        Args:
            data: Opcjonalny słownik z danymi do narysowania na portrecie
            save_path: Opcjonalna ścieżka do zapisania wykresu
            show_plot: Czy pokazać wykres

        Returns:
            Obiekt Figure jeśli show_plot=True, w przeciwnym razie None
        """
        if not self.is_fitted:
            raise ValueError("Model nie został jeszcze dopasowany.")

        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        d = self.parameters['d']

        # Tworzymy siatkę punktów dla przestrzeni fazowej
        H_range = np.linspace(0, 1, 20)
        A_range = np.linspace(0, 1, 20)
        H_grid, A_grid = np.meshgrid(H_range, A_range)

        # Obliczamy pochodne w każdym punkcie siatki
        dHdt = a * H_grid + b * A_grid
        dAdt = c * H_grid + d * A_grid

        # Normalizacja wektorów dla lepszej wizualizacji
        magnitude = np.sqrt(dHdt**2 + dAdt**2)
        dHdt_norm = dHdt / (magnitude + 1e-10)  # Unikamy dzielenia przez zero
        dAdt_norm = dAdt / (magnitude + 1e-10)

        # Tworzymy wykres
        fig, ax = plt.subplots(figsize=(10, 8))

        # Rysujemy pole wektorowe
        quiver = ax.quiver(H_grid, A_grid, dHdt_norm, dAdt_norm, magnitude, 
                          cmap='viridis', pivot='mid', alpha=0.8)
        plt.colorbar(quiver, ax=ax, label='Prędkość zmian')

        # Jeśli podano dane, rysujemy je na wykresie
        if data is not None and 'H' in data and 'A' in data:
            ax.plot(data['H'], data['A'], 'k-', label='Dane empiryczne')
            ax.plot(data['H'][0], data['A'][0], 'bo', markersize=10, label='Start')
            ax.plot(data['H'][-1], data['A'][-1], 'ro', markersize=10, label='Koniec')

        # Oznaczamy punkty równowagi
        equilibrium_points = self.get_equilibrium_points()
        for point in equilibrium_points:
            if point['type'] != 'continuum':  # Nie rysujemy dla continuum punktów równowagi
                ax.plot(point['H'], point['A'], 'go', markersize=12, label='Punkt równowagi')

        # Ustawienia wykresu
        ax.set_xlabel('Przywiązanie użytkownika (H)')
        ax.set_ylabel('Zaangażowanie AI (A)')
        ax.set_title('Portret fazowy dynamiki relacji')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Dodajemy równania i parametry
        eq_text = f"$\\frac{{dH}}{{dt}} = {a:.4f} \\cdot H + {b:.4f} \\cdot A$\n"
        eq_text += f"$\\frac{{dA}}{{dt}} = {c:.4f} \\cdot H + {d:.4f} \\cdot A$"

        # Dodajemy informację o stabilności
        stability = self.analyze_stability()
        eq_text += f"\n\nTyp dynamiki: {stability['dynamics_type']}"

        ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Dodajemy legendę
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower right')

        plt.tight_layout()

        # Zapisujemy wykres jeśli podano ścieżkę
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Zapisano portret fazowy do {save_path}")

        # Zwracamy lub pokazujemy wykres
        if show_plot:
            plt.show()
            return fig
        else:
            plt.close(fig)
            return None

    def _generate_parameter_explanations(self) -> Dict[str, str]:
        """Generuje wyjaśnienia znaczenia parametrów modelu.

        Returns:
            Słownik z wyjaśnieniami dla każdego parametru
        """
        if not self.is_fitted:
            return {"error": "Model nie został jeszcze dopasowany."}

        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        d = self.parameters['d']

        explanations = {}

        # Parametr a: wpływ przywiązania użytkownika na jego zmianę
        if a > 0.05:
            explanations["a"] = "Silna tendencja użytkownika do zwiększania przywiązania niezależnie od AI."
        elif a > 0:
            explanations["a"] = "Słaba tendencja użytkownika do zwiększania przywiązania niezależnie od AI."
        elif a > -0.05:
            explanations["a"] = "Słaba tendencja użytkownika do zmniejszania przywiązania z czasem."
        else:
            explanations["a"] = "Silna tendencja użytkownika do zmniejszania przywiązania z czasem."

        # Parametr b: wpływ zaangażowania AI na zmianę przywiązania użytkownika
        if b > 0.1:
            explanations["b"] = "Zaangażowanie AI silnie zwiększa przywiązanie użytkownika."
        elif b > 0:
            explanations["b"] = "Zaangażowanie AI umiarkowanie zwiększa przywiązanie użytkownika."
        elif b > -0.1:
            explanations["b"] = "Zaangażowanie AI lekko zmniejsza przywiązanie użytkownika."
        else:
            explanations["b"] = "Zaangażowanie AI silnie zmniejsza przywiązanie użytkownika."

        # Parametr c: wpływ przywiązania użytkownika na zmianę zaangażowania AI
        if c > 0.1:
            explanations["c"] = "Przywiązanie użytkownika silnie zwiększa zaangażowanie AI."
        elif c > 0:
            explanations["c"] = "Przywiązanie użytkownika umiarkowanie zwiększa zaangażowanie AI."
        elif c > -0.1:
            explanations["c"] = "Przywiązanie użytkownika lekko zmniejsza zaangażowanie AI."
        else:
            explanations["c"] = "Przywiązanie użytkownika silnie zmniejsza zaangażowanie AI."

        # Parametr d: wpływ zaangażowania AI na jego zmianę
        if d > 0.05:
            explanations["d"] = "Silna tendencja AI do zwiększania zaangażowania niezależnie od użytkownika."
        elif d > 0:
            explanations["d"] = "Słaba tendencja AI do zwiększania zaangażowania niezależnie od użytkownika."
        elif d > -0.05:
            explanations["d"] = "Słaba tendencja AI do zmniejszania zaangażowania z czasem."
        else:
            explanations["d"] = "Silna tendencja AI do zmniejszania zaangażowania z czasem."

        return explanations
