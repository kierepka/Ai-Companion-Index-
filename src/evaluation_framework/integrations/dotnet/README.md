# Integracja .NET dla AI Friendliness Evaluator

Ten katalog zawiera komponenty umożliwiające używanie frameworka oceny przyjazności AI w projektach .NET.

## Zawartość

* `client.cs` - Główny klient .NET, który umożliwia aplikacjom .NET komunikację z frameworkiem
* `python_net_bridge.py` - Most pomiędzy Pythonem a .NET, który obsługuje translację danych
* `AIFriendlinessEvaluator.csproj` - Plik projektu .NET

## Szybki start

### Instalacja

```bash
dotnet add package AIFriendlinessEvaluator
```

### Podstawowe użycie

```csharp
using AIFriendliness;

// Inicjalizacja ewaluatora
var evaluator = new FriendlinessEvaluator();

// Ocena pojedynczej odpowiedzi AI
var response = "Dziękuję za pytanie! Z przyjemnością Ci pomogę rozwiązać ten problem.";
var score = evaluator.EvaluateResponse(response);

// Ocena całej sesji
var session = new AISession();
session.AddUserMessage("Jak mogę rozwiązać ten problem?")
       .AddAIResponse("Z przyjemnością Ci pomogę. Najpierw sprawdźmy...")
       .AddUserMessage("Dziękuję, to ma sens.");

var sessionMetrics = evaluator.EvaluateSession(session);
Console.WriteLine($"Wskaźnik przyjazności sesji: {sessionMetrics.FriendlinessScore}");
```

## Zaawansowana konfiguracja

Można dostosować parametry oceny, korzystając z klasy `EvaluationConfig`:

```csharp
var config = new EvaluationConfig
{
    PositiveThreshold = 0.7,     // Próg dla pozytywnych interakcji
    NegativeWeight = 3.0,        // Waga negatywnych interakcji
    EmotionalDecayRate = 0.2     // Współczynnik zaniku emocji
};

var evaluator = new FriendlinessEvaluator(config);
```

## Integracja z modelami różniczkowania

Dla zaawansowanych przypadków, można korzystać z modeli równań różniczkowych dla głębszej analizy:

```csharp
var diffModel = new DifferentialModel();
var prediction = diffModel.PredictEmotionalTrajectory(session);

// Wizualizacja trajektorii emocjonalnej
diffModel.PlotEmotionalTrajectory(prediction);
```

## Rozwiązywanie problemów

Jeśli napotkasz problemy podczas integracji, upewnij się, że:

1. Python oraz framework oceny przyjazności AI są poprawnie zainstalowane
2. Ścieżki w pliku konfiguracyjnym wskazują na poprawne lokalizacje
3. Odpowiednie wersje pakietów są kompatybilne

Dla dodatkowego wsparcia, sprawdź [pełną dokumentację](../../../docs/dotnet_integration.md) lub zgłoś problem w systemie śledzenia błędów projektu.
