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

func main() {
	// k := &cnn.Kernel{
	// 	Step: 1,
	// 	Size: 3,
	// 	Value: []float64{
	// 		-1, -1, -1,
	// 		-1, 8, -1,
	// 		-1, -1, -1,
	// 	},
	// }

	// p := &cnn.MaxPooling{
	// 	BasePooling: cnn.BasePooling{Size: 2},
	// }

	// data := [][]float64{
	// 	{0, 1, 0, 0, 0, 0, 0, 0},
	// 	{0, 1, 0, 0, 1, 1, 1, 0},
	// 	{0, 1, 0, 0, 0, 0, 1, 0},
	// 	{0, 1, 0, 0, 0, 0, 1, 0},
	// 	{0, 1, 0, 0, 0, 0, 1, 0},
	// 	{0, 1, 0, 0, 0, 0, 1, 0},
	// }

	// r := k.ExtractAll(data)

	// for _, v := range r {
	// 	fmt.Println(v)
	// }

	// r2 := p.PoolingAll(p, r)
	// for _, v := range r2 {
	// 	fmt.Println(v)
	// }

	n := cnn.NewNeuralNetwork([]int64{2, 1}, []cnn.IActive{cnn.Sigmoid}, cnn.LogisticDiff)

	// str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"0-0":-7.249935669704905,"0-1":7.90453879704736,"weight":-0.6715934689739514}]]` // 0.17254570314572118
	// str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"0-0":-14.98969899799228,"0-1":13.857857520509198,"weight":0.17208973994279142}]]`  // 0.07207974243123158
	// str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"0-0":-18.250244719453068,"0-1":17.060935792694767,"weight":0.14705368810981495}]]` // 0.05114406571725615
	str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"0-0":-18.202112009529372,"0-1":17.093395741488663,"weight":0.6617362893934389}]]` // 0.04642070943222921
	var wm cnn.WeightMap
	err := json.Unmarshal([]byte(str), &wm)
	if err != nil {
		panic(err)
	}
	n.ApplyWeight(wm)

	data := generateData(500)
	// saveData(data)
	// data := loadData()
	NewPointChart(data)

	n.Calculate(data.Inputs)

	// lossData := n.Train(data.Inputs, data.Expects, 100, 16, 0.01, 0.01)

	// ws := n.ExportWeight()
	// b, _ := json.Marshal(ws)
	// fmt.Println("ws -> ", string(b))

	// NewLineChart(lossData)
	// cnn.GenerateLossChart(lossData, "test_out.png")
}

func NewLineChart(data []float64) {
	if len(data) == 0 {
		return
	}
	series := make([]chart.Series, 0)
	xValue := make([]float64, 0, len(data))
	for i, _ := range data {
		xValue = append(xValue, float64(i))
	}

	series = append(series, chart.ContinuousSeries{
		Name:    "text",
		XValues: xValue,
		YValues: data,
	})
	graph := chart.Chart{Series: series}
	f, err := os.Create("test_out.png")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	err = graph.Render(chart.PNG, f)
	if err != nil {
		panic(err)
	}
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
		XValues: []float64{0, 0.008886, 1},
		YValues: []float64{-0.009462, 0, 1.055400},
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
