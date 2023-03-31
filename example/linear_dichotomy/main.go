package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/wcharczuk/go-chart/v2"
	"github.com/weiWang95/cnn"
)

func getY(x float64) float64 {
	// a * x + b * y + w1 = 0.5
	// y = (0.5 - w1 - a * x) / b
	return (0.5 - 0.3862034259359683 + 1.4569716331686737*x) / 0.3433739866226968
}

type Data struct {
	Inputs  [][]float64 `json:"inputs"`
	Expects [][]float64 `json:"expects"`
}

func main() {
	rand.Seed(time.Now().UnixNano())
	// n := cnn.NewNeuralNetwork([]int64{2, 4, 1}, []cnn.IActive{cnn.ReLU, cnn.Sigmoid}, cnn.LogisticDiff)
	n := cnn.NewNeuralNetwork([]int64{2, 1}, []cnn.IActive{cnn.Sigmoid}, cnn.LogisticDiff)
	// str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"0-0":-6.01774183129974,"0-1":4.371805133005663,"weight":0.7506932650616476}]]` // 0.22
	// var wm cnn.WeightMap
	// err := json.Unmarshal([]byte(str), &wm)
	// if err != nil {
	// 	panic(err)
	// }
	// n.ApplyWeight(wm)

	// data := generateData(500)
	// saveData(data)
	data := loadData()
	NewPointChart(data)

	// n.Calculate(data.Inputs)

	lr := cnn.NewStepLR(0.1, 0.1, 0.01, 20)
	// lr := cnn.NewConstLR(0.1)
	lossData := n.Train(data.Inputs, data.Expects, 1000, 64, lr, 0.01)
	// lossData, testLossData := n.Train(data.Inputs[0:250], data.Expects[0:250], data.Inputs[250:], data.Expects[250:], 1, 1, lr, 0.01)

	ws := n.ExportWeight()
	b, _ := json.Marshal(ws)
	fmt.Println("ws -> ", string(b))

	// NewLineChart(lossData)
	fmt.Println(len(lossData))
	// fmt.Println(len(lossData), len(testLossData))
	cnn.GenerateLossChart(lossData, "test_out.png")
	// cnn.GenerateAllLossChart(lossData, testLossData, "test_out.png")
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
	inputs := make([][]float64, 0)
	expects := make([][]float64, 0)
	for {
		x, y := randPoint(1)
		if x-y < 0.05 && x-y > -0.05 {
			continue
		}
		inputs = append(inputs, []float64{x, y})
		if y > x {
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

func randPoint(m float64) (float64, float64) {
	return round2(rand.Float64() * m), round2(rand.Float64() * m)
}

func round2(v float64) float64 {
	v, _ = strconv.ParseFloat(fmt.Sprintf("%.02f", v), 0)
	return v
}
