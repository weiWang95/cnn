package cnn

import (
	"os"

	"github.com/wcharczuk/go-chart/v2"
)

func GenerateLossChart(data []float64, name string) error {
	if len(data) == 0 {
		return nil
	}

	series := make([]chart.Series, 0)
	xValue := make([]float64, 0, len(data))
	for i, _ := range data {
		xValue = append(xValue, float64(i))
	}

	series = append(series, chart.ContinuousSeries{
		Name:    "loss",
		XValues: xValue,
		YValues: data,
	})
	graph := chart.Chart{Series: series}

	f, err := os.Create(name)
	if err != nil {
		return err
	}
	defer f.Close()

	err = graph.Render(chart.PNG, f)
	if err != nil {
		return err
	}

	return nil
}
