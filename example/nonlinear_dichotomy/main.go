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

func getNY(x float64, n, i int, wm cnn.WeightMap) float64 {
	return getY(x, wm[n][i][fmt.Sprintf("%d-0", n-1)], wm[n][i][fmt.Sprintf("%d-1", n-1)], wm[n][i]["weight"])
}

func getY(x, w1, w2, b float64) float64 {
	return (0.5 - b - w1) * x / w2
}

func main() {
	n := cnn.NewNeuralNetwork([]int64{2, 4, 4, 1}, []cnn.IActive{cnn.ReLU, cnn.ReLU, cnn.Sigmoid}, cnn.LogisticDiff)

	// str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"0-0":0.8968232542163553,"0-1":0.8807586146183405,"weight":-0.5577256636400991},{"0-0":1.0020757858022826,"0-1":0.8075301729062754,"weight":-0.5511510746106212},{"0-0":0.6259485602306175,"0-1":1.1643488034052096,"weight":-0.5576018003123591},{"0-0":0.7743938498576542,"0-1":1.016100907235981,"weight":-0.5513924786248734}],[{"1-0":1.0163027748265099,"1-1":1.096018077974827,"1-2":1.0536176198907243,"1-3":1.0694203093101808,"weight":-2.510490907195784}]]`
	// var wm cnn.WeightMap
	// json.Unmarshal([]byte(str), &wm)
	// n.ApplyWeight(wm)

	// data := generateData(500)
	// saveData(data)
	data := loadData()
	NewPointChart(data)

	lossData, testLossData := n.TrainWithTest(data.Inputs[0:400], data.Expects[0:400], data.Inputs[400:], data.Expects[400:], 10000, 32, 0.01, 0.1)

	ws := n.ExportWeight()
	b, _ := json.Marshal(ws)
	fmt.Println("ws -> ", string(b))

	fmt.Println(len(lossData), len(testLossData))
	// cnn.GenerateAllLossChart(lossData, testLossData, "test_out.png")

	// var count int
	// for i, input := range data.Inputs {
	// 	outs := n.Compute(input...)
	// 	if (outs[0] > 0.5 && data.Expects[i][0] > 0.5) || (outs[0] < 0.5 && data.Expects[i][0] < 0.5) {
	// 		count += 1
	// 	}
	// }
	// fmt.Println(float64(count) / float64(len(data.Inputs)))
}

func randPoint(m float64) (float64, float64) {
	return round2(rand.Float64() * m), round2(rand.Float64() * m)
}

func round2(v float64) float64 {
	v, _ = strconv.ParseFloat(fmt.Sprintf("%.02f", v), 10)
	return v
}

type Data struct {
	Inputs  [][]float64 `json:"inputs"`
	Expects [][]float64 `json:"expects"`
}

func generateData(size int) *Data {
	r := 1.0
	inputs := make([][]float64, 0)
	expects := make([][]float64, 0)
	for {
		x, y := randPoint(r)
		if (x-y < r*0.05 && x-y > -r*0.05) || (1-x-y < r*0.05 && 1-x-y > -r*0.05) {
			continue
		}
		inputs = append(inputs, []float64{x, y})
		if y > x && r-x < y {
			expects = append(expects, []float64{0.99})
		} else {
			expects = append(expects, []float64{0.01})
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

func NewPointChart(data *Data) {
	series := make([]chart.Series, 0)
	x1Value := make([]float64, 0)
	x2Value := make([]float64, 0)
	y1Value := make([]float64, 0)
	y2Value := make([]float64, 0)

	for i, _ := range data.Inputs {
		if data.Expects[i][0] > 0.5 {
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

	// for i, _ := range wm[1] {
	// 	series = append(series, chart.ContinuousSeries{
	// 		Style: chart.Style{
	// 			StrokeColor: chart.GetDefaultColor(1),
	// 		},
	// 		XValues: []float64{0, 1},
	// 		YValues: []float64{getNY(0, 1, i, wm), getNY(1, 1, i, wm)},
	// 	})
	// }
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
