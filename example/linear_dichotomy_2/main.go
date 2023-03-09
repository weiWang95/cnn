package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"strconv"

	"github.com/wcharczuk/go-chart/v2"
	"github.com/weiWang95/cnn"
)

func randPoint(m float64) (float64, float64) {
	return round2(rand.Float64() * m), round2(rand.Float64() * m)
}

func round2(v float64) float64 {
	v, _ = strconv.ParseFloat(fmt.Sprintf("%.02f", v), 0)
	return v
}

type Data struct {
	Inputs  [][]float64 `json:"inputs"`
	Expects [][]float64 `json:"expects"`
}

func getY(x float64) float64 {
	return (0.5 + 25.915607 - 27.071638*x) / 25.124487
}

func main() {
	n := cnn.NewNeuralNetwork([]int64{2, 1}, []cnn.IActive{cnn.Sigmoid}, cnn.LogisticDiff)

	str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"0-0":27.071638499344214,"0-1":25.12448716824279,"weight":-25.915607424291757}]]`
	var wm cnn.WeightMap
	err := json.Unmarshal([]byte(str), &wm)
	if err != nil {
		panic(err)
	}
	n.ApplyWeight(wm)

	// data := generateData(500)
	// saveData(data)
	data := loadData()
	NewPointChart(data)

	n.Calculate(data.Inputs)

	lossData := n.Train(data.Inputs, data.Expects, 100, 0.8, 0.01)

	ws := n.ExportWeight()
	b, _ := json.Marshal(ws)
	fmt.Println("ws -> ", string(b))

	cnn.GenerateLossChart(lossData, "test_out.png")
}

func NewPointChart(data *Data) {
	series := make([]chart.Series, 0)
	x1Value := make([]float64, 0)
	x2Value := make([]float64, 0)
	y1Value := make([]float64, 0)
	y2Value := make([]float64, 0)

	for i, _ := range data.Inputs {
		if data.Expects[i][0] == 1 {
			x1Value = append(x1Value, data.Inputs[i][0])
			y1Value = append(y1Value, data.Inputs[i][1])
		} else {
			x2Value = append(x2Value, data.Inputs[i][0])
			y2Value = append(y2Value, data.Inputs[i][1])
		}
	}

	series = append(series, chart.ContinuousSeries{
		Name: "red",
		Style: chart.Style{
			StrokeWidth: chart.Disabled,
			DotWidth:    3,
			DotColor:    chart.ColorRed,
		},
		XValues: x1Value,
		YValues: y1Value,
	})
	series = append(series, chart.ContinuousSeries{
		Name: "blue",
		Style: chart.Style{
			StrokeWidth: chart.Disabled,
			DotWidth:    3,
			DotColor:    chart.ColorBlue,
		},
		XValues: x2Value,
		YValues: y2Value,
	})

	series = append(series, chart.ContinuousSeries{
		Style: chart.Style{
			StrokeColor: chart.GetDefaultColor(1),
		},
		XValues: []float64{0, 1},
		YValues: []float64{getY(0), getY(1)},
	})
	graph := chart.Chart{Series: series}
	f, err := os.Create("data.png")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	err = graph.Render(chart.PNG, f)
	if err != nil {
		panic(err)
	}
}

func generateData(size int) *Data {
	r := float64(1)
	inputs := make([][]float64, 0)
	expects := make([][]float64, 0)
	for {
		x, y := randPoint(r)
		if 1-x-y < r*0.05 && 1-x-y > -r*0.05 {
			continue
		}
		inputs = append(inputs, []float64{x, y})
		if y > r-x {
			expects = append(expects, []float64{1})
		} else {
			expects = append(expects, []float64{0})
		}
		if len(inputs) == size {
			break
		}
	}

	return &Data{
		Inputs:  inputs,
		Expects: expects,
	}
}

func saveData(data *Data) {
	b, _ := json.Marshal(data)
	f, err := os.Create("data.json")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	f.Write(b)
}

func loadData() *Data {
	f, err := os.Open("data.json")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	b, _ := ioutil.ReadAll(f)
	var data Data
	if err := json.Unmarshal(b, &data); err != nil {
		panic(err)
	}
	return &data
}
