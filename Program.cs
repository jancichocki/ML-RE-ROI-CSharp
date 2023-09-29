using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace RealEstatePrediction
{
    public class RealEstateData
    {
        public float Location_Score { get; set; }
        public float Square_Footage { get; set; }
        public float Age_of_Property { get; set; }
        public float Vacancy_Rate { get; set; }
        public float Market_Growth { get; set; }
        public float Interest_Rate { get; set; }
        public float Property_Tax { get; set; }
        public float Operating_Expenses { get; set; }
        public float Net_Operating_Income { get; set; }
        public float Price { get; set; }

        [ColumnName("Label")]
        public float ROI { get; set; }
    }

    public class RealEstatePrediction
    {
        [ColumnName("Score")]
        public float ROI { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            // Create new MLContext
            MLContext mlContext = new MLContext(seed: 42);

            // Read data into IDataView
            IDataView data = mlContext.Data.LoadFromTextFile<RealEstateData>("./real_estate_data.csv", separatorChar: ',');

            // Split the data into training and test sets
            var tt = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // Data process configuration with pipeline data transformations
            var pipeline = mlContext.Transforms.Concatenate("Features", 
                    "Location_Score", "Square_Footage", "Age_of_Property", "Vacancy_Rate",
                    "Market_Growth", "Interest_Rate", "Property_Tax", "Operating_Expenses",
                    "Net_Operating_Income", "Price")
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Transforms.CopyColumns(inputColumnName: "ROI", outputColumnName: "Label"))
                .Append(mlContext.Transforms.Regression.Trainers.FastForest());

            // Train the model
            var model = pipeline.Fit(tt.TrainSet);

            // Evaluate the model
            var predictions = model.Transform(tt.TestSet);
            var metrics = mlContext.Regression.Evaluate(predictions);
            Console.WriteLine($"Random Forest MSE: {metrics.MeanSquaredError}");
        }
    }
}
