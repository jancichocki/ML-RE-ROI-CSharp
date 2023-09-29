using Microsoft.ML.Data;

public class RealEstateData
{
    [LoadColumn(0)]
    public float Location_Score { get; set; }

    [LoadColumn(1)]
    public float Square_Footage { get; set; }

    [LoadColumn(2)]
    public float Age_of_Property { get; set; }

    [LoadColumn(3)]
    public float Vacancy_Rate { get; set; }

    [LoadColumn(4)]
    public float Market_Growth { get; set; }

    [LoadColumn(5)]
    public float Interest_Rate { get; set; }

    [LoadColumn(6)]
    public float Property_Tax { get; set; }

    [LoadColumn(7)]
    public float Operating_Expenses { get; set; }

    [LoadColumn(8)]
    public float Net_Operating_Income { get; set; }

    [LoadColumn(9)]
    public float Price { get; set; }

    [LoadColumn(10)]
    [ColumnName("Label")]
    public float ROI { get; set; }
}
