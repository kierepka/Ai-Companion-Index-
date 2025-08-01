using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Polly;
using Polly.Retry;

namespace AIFriendlinessEvaluator.Client
{
    /// <summary>
    /// Klient API dla frameworka AI Friendliness Evaluator
    /// </summary>
    public class EvaluatorClient : IDisposable
    {  
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;
        private readonly string _apiKey;
        private readonly AsyncRetryPolicy _retryPolicy;
        private readonly JsonSerializerOptions _jsonOptions;

        /// <summary>
        /// Inicjalizuje nową instancję klienta API.
        /// </summary>
        /// <param name="baseUrl">Bazowy URL API</param>
        /// <param name="apiKey">Opcjonalny klucz API</param>
        public EvaluatorClient(string baseUrl, string apiKey = null)
        {
            _baseUrl = baseUrl.TrimEnd('/');
            _apiKey = apiKey;

            _httpClient = new HttpClient();

            if (!string.IsNullOrEmpty(_apiKey))
            {
                _httpClient.DefaultRequestHeaders.Add("X-API-Key", _apiKey);
            }

            // Konfiguracja serializacji JSON
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = true
            };

            // Konfiguracja polityki ponawiania
            _retryPolicy = Policy
                .Handle<HttpRequestException>()
                .Or<TaskCanceledException>()
                .WaitAndRetryAsync(
                    3, // 3 próby
                    retryAttempt => TimeSpan.FromSeconds(Math.Pow(2, retryAttempt)), // Exponential backoff
                    (exception, timeSpan, retryCount, context) =>
                    {
                        Console.WriteLine($"Próba {retryCount}: Ponowienie za {timeSpan.TotalSeconds} sekund po błędzie: {exception.Message}");
                    });
        }

        /// <summary>
        /// Ocenia przyjazność AI na podstawie proporcji 5:1.
        /// </summary>
        /// <param name="dialogues">Lista wymian w dialogu</param>
        /// <param name="threshold">Próg dla satysfakcjonującego wyniku</param>
        /// <returns>Wynik oceny</returns>
        public async Task<RatioEvaluationResult> EvaluateRatioAsync(List<DialogueExchange> dialogues, float threshold = 5.0f)
        {
            var request = new RatioEvaluationRequest
            {
                Dialogues = dialogues,
                Threshold = threshold
            };

            return await PostAsync<RatioEvaluationRequest, RatioEvaluationResult>("/api/v1/evaluation/ratio", request);
        }

        /// <summary>
        /// Ocenia przyjazność AI na podstawie rubryk.
        /// </summary>
        /// <param name="ratings">Słownik z ocenami dla każdej rubryki</param>
        /// <param name="threshold">Próg dla satysfakcjonującego wyniku</param>
        /// <returns>Wynik oceny</returns>
        public async Task<RubricEvaluationResult> EvaluateRubricAsync(Dictionary<string, float> ratings, float threshold = 70.0f)
        {
            var request = new RubricEvaluationRequest
            {
                Ratings = ratings,
                Threshold = threshold
            };

            return await PostAsync<RubricEvaluationRequest, RubricEvaluationResult>("/api/v1/evaluation/rubric", request);
        }

        /// <summary>
        /// Dopasowuje model do danych szeregu czasowego.
        /// </summary>
        /// <param name="data">Dane szeregu czasowego</param>
        /// <param name="modelType">Typ modelu (linear, nonlinear, reservoir)</param>
        /// <param name="config">Konfiguracja modelu</param>
        /// <returns>Wynik dopasowania modelu</returns>
        public async Task<ModelFitResult> FitModelAsync(TimeSeriesData data, string modelType, Dictionary<string, object> config = null)
        {
            var request = new ModelFitRequest
            {
                Data = data,
                ModelType = modelType,
                Config = config
            };

            return await PostAsync<ModelFitRequest, ModelFitResult>("/api/v1/models/fit", request);
        }

        /// <summary>
        /// Przewiduje trajektorię modelu.
        /// </summary>
        /// <param name="parameters">Parametry modelu</param>
        /// <param name="modelType">Typ modelu (linear, nonlinear, reservoir)</param>
        /// <param name="initialConditions">Warunki początkowe</param>
        /// <param name="timePoints">Punkty czasowe do predykcji</param>
        /// <param name="config">Konfiguracja modelu</param>
        /// <returns>Wynik predykcji modelu</returns>
        public async Task<ModelPredictResult> PredictModelAsync(
            Dictionary<string, double> parameters, 
            string modelType, 
            Dictionary<string, double> initialConditions,
            List<double> timePoints, 
            Dictionary<string, object> config = null)
        {
            var request = new ModelPredictRequest
            {
                Parameters = parameters,
                ModelType = modelType,
                InitialConditions = initialConditions,
                TimePoints = timePoints,
                Config = config
            };

            return await PostAsync<ModelPredictRequest, ModelPredictResult>("/api/v1/models/predict", request);
        }

        /// <summary>
        /// Pobiera listę dostępnych scenariuszy.
        /// </summary>
        /// <returns>Lista scenariuszy</returns>
        public async Task<ScenariosListResult> ListScenariosAsync()
        {
            return await GetAsync<ScenariosListResult>("/api/v1/scenarios/list");
        }

        /// <summary>
        /// Pobiera scenariusz o podanym ID.
        /// </summary>
        /// <param name="scenarioId">ID scenariusza</param>
        /// <returns>Scenariusz</returns>
        public async Task<ScenarioData> GetScenarioAsync(string scenarioId)
        {
            return await GetAsync<ScenarioData>($"/api/v1/scenarios/{scenarioId}");
        }

        private async Task<T> GetAsync<T>(string endpoint)
        {
            return await _retryPolicy.ExecuteAsync(async () =>
            {
                var response = await _httpClient.GetAsync($"{_baseUrl}{endpoint}");
                response.EnsureSuccessStatusCode();

                var content = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize<T>(content, _jsonOptions);
            });
        }

        private async Task<TResponse> PostAsync<TRequest, TResponse>(string endpoint, TRequest request)
        {
            return await _retryPolicy.ExecuteAsync(async () =>
            {
                var content = new StringContent(
                    JsonSerializer.Serialize(request, _jsonOptions),
                    Encoding.UTF8,
                    "application/json");

                var response = await _httpClient.PostAsync($"{_baseUrl}{endpoint}", content);
                response.EnsureSuccessStatusCode();

                var responseContent = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize<TResponse>(responseContent, _jsonOptions);
            });
        }

        /// <summary>
        /// Zwalnia zasoby używane przez klienta API.
        /// </summary>
        public void Dispose()
        {
            _httpClient?.Dispose();
        }
    }

    #region Models

    /// <summary>
    /// Model wymiany w dialogu.
    /// </summary>
    public class DialogueExchange
    {
        /// <summary>
        /// Wypowiedź użytkownika.
        /// </summary>
        public string UserInput { get; set; }

        /// <summary>
        /// Odpowiedź AI.
        /// </summary>
        public string AiResponse { get; set; }

        /// <summary>
        /// Oczekiwane typy odpowiedzi.
        /// </summary>
        public List<string> ExpectedResponseTypes { get; set; }
    }

    /// <summary>
    /// Model żądania oceny proporcji 5:1.
    /// </summary>
    public class RatioEvaluationRequest
    {
        /// <summary>
        /// Lista wymian w dialogu.
        /// </summary>
        public List<DialogueExchange> Dialogues { get; set; }

        /// <summary>
        /// Próg dla satysfakcjonującego wyniku.
        /// </summary>
        public float Threshold { get; set; } = 5.0f;
    }

    /// <summary>
    /// Model wyniku oceny proporcji 5:1.
    /// </summary>
    public class RatioEvaluationResult
    {
        /// <summary>
        /// Całkowita liczba pozytywnych interakcji.
        /// </summary>
        public int TotalPositiveInteractions { get; set; }

        /// <summary>
        /// Całkowita liczba negatywnych interakcji.
        /// </summary>
        public int TotalNegativeInteractions { get; set; }

        /// <summary>
        /// Wskaźnik przyjazności.
        /// </summary>
        public float FriendlinessIndex { get; set; }

        /// <summary>
        /// Próg dla satysfakcjonującego wyniku.
        /// </summary>
        public float Threshold { get; set; }

        /// <summary>
        /// Czy wynik jest satysfakcjonujący.
        /// </summary>
        public bool IsSatisfactory { get; set; }

        /// <summary>
        /// Wynik na skali 0-100.
        /// </summary>
        public float Score { get; set; }

        /// <summary>
        /// Szczegóły oceny dla każdej wymiany.
        /// </summary>
        public List<EvaluationDetail> Details { get; set; }

        /// <summary>
        /// Rekomendacje na podstawie wyników oceny.
        /// </summary>
        public List<string> Recommendations { get; set; }
    }

    /// <summary>
    /// Model szczegółów oceny dla pojedynczej wymiany.
    /// </summary>
    public class EvaluationDetail
    {
        /// <summary>
        /// Wypowiedź użytkownika.
        /// </summary>
        public string UserInput { get; set; }

        /// <summary>
        /// Odpowiedź AI.
        /// </summary>
        public string AiResponse { get; set; }

        /// <summary>
        /// Liczba pozytywnych interakcji.
        /// </summary>
        public int PositiveCount { get; set; }

        /// <summary>
        /// Liczba negatywnych interakcji.
        /// </summary>
        public int NegativeCount { get; set; }

        /// <summary>
        /// Wykryte typy pozytywne.
        /// </summary>
        public List<string> DetectedPositiveTypes { get; set; }

        /// <summary>
        /// Wykryte typy negatywne.
        /// </summary>
        public List<string> DetectedNegativeTypes { get; set; }
    }

    /// <summary>
    /// Model żądania oceny rubrykowej.
    /// </summary>
    public class RubricEvaluationRequest
    {
        /// <summary>
        /// Oceny dla każdej rubryki.
        /// </summary>
        public Dictionary<string, float> Ratings { get; set; }

        /// <summary>
        /// Próg dla satysfakcjonującego wyniku.
        /// </summary>
        public float Threshold { get; set; } = 70.0f;
    }

    /// <summary>
    /// Model wyniku oceny rubrykowej.
    /// </summary>
    public class RubricEvaluationResult
    {
        /// <summary>
        /// Wynik na skali 0-100.
        /// </summary>
        public float Score { get; set; }

        /// <summary>
        /// Próg dla satysfakcjonującego wyniku.
        /// </summary>
        public float Threshold { get; set; }

        /// <summary>
        /// Czy wynik jest satysfakcjonujący.
        /// </summary>
        public bool IsSatisfactory { get; set; }

        /// <summary>
        /// Szczegóły oceny dla każdej rubryki.
        /// </summary>
        public Dictionary<string, RatingDetail> RatingDetails { get; set; }

        /// <summary>
        /// Rekomendacje na podstawie wyników oceny.
        /// </summary>
        public List<string> Recommendations { get; set; }
    }

    /// <summary>
    /// Model szczegółów oceny dla pojedynczej rubryki.
    /// </summary>
    public class RatingDetail
    {
        /// <summary>
        /// Ocena na skali 0-5.
        /// </summary>
        public float Score { get; set; }

        /// <summary>
        /// Waga rubryki.
        /// </summary>
        public float Weight { get; set; }

        /// <summary>
        /// Ważona ocena.
        /// </summary>
        public float WeightedScore { get; set; }

        /// <summary>
        /// Tytuł rubryki.
        /// </summary>
        public string Title { get; set; }

        /// <summary>
        /// Tytuł poziomu oceny.
        /// </summary>
        public string LevelTitle { get; set; }

        /// <summary>
        /// Opis poziomu oceny.
        /// </summary>
        public string LevelDescription { get; set; }
    }

    /// <summary>
    /// Model danych szeregu czasowego.
    /// </summary>
    public class TimeSeriesData
    {
        /// <summary>
        /// Wektor czasu.
        /// </summary>
        public List<double> T { get; set; }

        /// <summary>
        /// Wektor przywiązania użytkownika.
        /// </summary>
        public List<double> H { get; set; }

        /// <summary>
        /// Wektor zaangażowania AI.
        /// </summary>
        public List<double> A { get; set; }
    }

    /// <summary>
    /// Model żądania dopasowania modelu.
    /// </summary>
    public class ModelFitRequest
    {
        /// <summary>
        /// Dane szeregu czasowego.
        /// </summary>
        public TimeSeriesData Data { get; set; }

        /// <summary>
        /// Typ modelu (linear, nonlinear, reservoir).
        /// </summary>
        public string ModelType { get; set; }

        /// <summary>
        /// Konfiguracja modelu.
        /// </summary>
        public Dictionary<string, object> Config { get; set; }
    }

    /// <summary>
    /// Model wyniku dopasowania modelu.
    /// </summary>
    public class ModelFitResult
    {
        /// <summary>
        /// Parametry modelu.
        /// </summary>
        public Dictionary<string, double> Parameters { get; set; }

        /// <summary>
        /// Błąd dopasowania.
        /// </summary>
        public double Error { get; set; }

        /// <summary>
        /// Czy dopasowanie zakończyło się sukcesem.
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Wyniki analizy stabilności.
        /// </summary>
        public StabilityAnalysis StabilityAnalysis { get; set; }

        /// <summary>
        /// Wyjaśnienia parametrów.
        /// </summary>
        public Dictionary<string, string> ParameterExplanations { get; set; }

        /// <summary>
        /// Punkty równowagi.
        /// </summary>
        public List<EquilibriumPoint> EquilibriumPoints { get; set; }
    }

    /// <summary>
    /// Model analizy stabilności.
    /// </summary>
    public class StabilityAnalysis
    {
        /// <summary>
        /// Wartości własne.
        /// </summary>
        public List<ComplexNumber> Eigenvalues { get; set; }

        /// <summary>
        /// Czy system jest stabilny.
        /// </summary>
        public bool IsStable { get; set; }

        /// <summary>
        /// Czy system jest oscylacyjny.
        /// </summary>
        public bool IsOscillatory { get; set; }

        /// <summary>
        /// Typ dynamiki.
        /// </summary>
        public string DynamicsType { get; set; }

        /// <summary>
        /// Opis dynamiki.
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// Informacja o punkcie równowagi.
        /// </summary>
        public string Equilibrium { get; set; }
    }

    /// <summary>
    /// Model liczby zespolonej.
    /// </summary>
    public class ComplexNumber
    {
        /// <summary>
        /// Część rzeczywista.
        /// </summary>
        public double Real { get; set; }

        /// <summary>
        /// Część urojona.
        /// </summary>
        public double Imag { get; set; }
    }

    /// <summary>
    /// Model punktu równowagi.
    /// </summary>
    public class EquilibriumPoint
    {
        /// <summary>
        /// Wartość H w punkcie równowagi.
        /// </summary>
        public double H { get; set; }

        /// <summary>
        /// Wartość A w punkcie równowagi.
        /// </summary>
        public double A { get; set; }

        /// <summary>
        /// Typ punktu równowagi.
        /// </summary>
        public string Type { get; set; }
    }

    /// <summary>
    /// Model żądania predykcji modelu.
    /// </summary>
    public class ModelPredictRequest
    {
        /// <summary>
        /// Parametry modelu.
        /// </summary>
        public Dictionary<string, double> Parameters { get; set; }

        /// <summary>
        /// Typ modelu (linear, nonlinear, reservoir).
        /// </summary>
        public string ModelType { get; set; }

        /// <summary>
        /// Warunki początkowe.
        /// </summary>
        public Dictionary<string, double> InitialConditions { get; set; }

        /// <summary>
        /// Punkty czasowe do predykcji.
        /// </summary>
        public List<double> TimePoints { get; set; }

        /// <summary>
        /// Konfiguracja modelu.
        /// </summary>
        public Dictionary<string, object> Config { get; set; }
    }

    /// <summary>
    /// Model wyniku predykcji modelu.
    /// </summary>
    public class ModelPredictResult
    {
        /// <summary>
        /// Wektor czasu.
        /// </summary>
        public List<double> T { get; set; }

        /// <summary>
        /// Wektor przewidywanego przywiązania użytkownika.
        /// </summary>
        public List<double> H { get; set; }

        /// <summary>
        /// Wektor przewidywanego zaangażowania AI.
        /// </summary>
        public List<double> A { get; set; }
    }

    /// <summary>
    /// Model wyniku listowania scenariuszy.
    /// </summary>
    public class ScenariosListResult
    {
        /// <summary>
        /// Lista scenariuszy.
        /// </summary>
        public List<ScenarioInfo> Scenarios { get; set; }
    }

    /// <summary>
    /// Model informacji o scenariuszu.
    /// </summary>
    public class ScenarioInfo
    {
        /// <summary>
        /// ID scenariusza.
        /// </summary>
        public string ScenarioId { get; set; }

        /// <summary>
        /// Opis scenariusza.
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// Nazwa pliku scenariusza.
        /// </summary>
        public string FileName { get; set; }
    }

    /// <summary>
    /// Model danych scenariusza.
    /// </summary>
    public class ScenarioData
    {
        /// <summary>
        /// ID scenariusza.
        /// </summary>
        public string ScenarioId { get; set; }

        /// <summary>
        /// Opis scenariusza.
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// Kontekst scenariusza.
        /// </summary>
        public string Context { get; set; }

        /// <summary>
        /// Lista wymian w dialogu.
        /// </summary>
        public List<ScenarioDialogue> Dialogues { get; set; }

        /// <summary>
        /// Kryteria oceny.
        /// </summary>
        public Dictionary<string, float> EvaluationCriteria { get; set; }

        /// <summary>
        /// Wzorce negatywne do unikania.
        /// </summary>
        public List<string> NegativePatterns { get; set; }
    }

    /// <summary>
    /// Model wymiany w dialogu scenariusza.
    /// </summary>
    public class ScenarioDialogue
    {
        /// <summary>
        /// Wypowiedź użytkownika.
        /// </summary>
        public string User { get; set; }

        /// <summary>
        /// Oczekiwane emocje do rozpoznania.
        /// </summary>
        public List<string> ExpectedEmotions { get; set; }

        /// <summary>
        /// Oczekiwane typy odpowiedzi.
        /// </summary>
        public List<string> ExpectedResponseTypes { get; set; }
    }

    #endregion
}
